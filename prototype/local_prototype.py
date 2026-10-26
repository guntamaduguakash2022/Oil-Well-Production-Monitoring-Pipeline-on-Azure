"""
Step 2: local prototype of the cleansing + anomaly detection logic (PySpark).

Reads  <data>/raw/production/date=*/wells_*.csv   (from generate_wells.py)
Writes <data>/curated/daily_well_metrics.csv      (for Power BI prototyping)
       <data>/curated/anomalies.csv
Compares detections to <data>/ground_truth/anomaly_labels.csv at the end.

Each function maps to a Databricks notebook step later:
  cleanse()         -> raw  -> cleansed zone
  hourly_rollup(), flag_hours(), consolidate()   -> anomaly detection
  daily_metrics()   -> curated zone

Usage:
  pip install pyspark pandas      (needs Java 17+)
  python local_prototype.py --data ./data
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

SENSOR_COLS = ["tubing_pressure_psi", "casing_pressure_psi", "oil_rate_bbl_d",
               "gas_rate_mcf_d", "water_cut_pct", "temperature_f"]
VALID_RANGES = {  # values outside these are treated as bad sensor data
    "tubing_pressure_psi": (0, 5000), "casing_pressure_psi": (0, 6000),
    "oil_rate_bbl_d": (0, 2000), "gas_rate_mcf_d": (0, 5000),
    "water_cut_pct": (0, 100), "temperature_f": (50, 350),
}

# ---- detection settings: tune these ----
BASELINE_DAYS = 7        # how much history forms the baseline
BASELINE_LAG_HOURS = 24  # skip the most recent day so an ongoing event can't pollute its own baseline
Z_THRESHOLD = 3.0        # how many std-devs below baseline counts as unusual
MIN_DROP_PCT = 0.10      # ...and it must also be at least this far below baseline
MIN_EVENT_HOURS = 2      # ignore blips shorter than this
MERGE_GAP_HOURS = 3      # flagged hours closer than this belong to one event


# ---------------------------------------------------------------- reading
def read_raw(spark, prod_dir, reader):
    """Load all daily CSVs as strings (typing happens in cleanse()).

    reader="spark":  spark.read.csv  -> what you'll use on Databricks.
    reader="pandas": read with pandas, then hand to Spark. Needed on Windows, where Spark's
                     file reader fails without Hadoop's winutils.exe / hadoop.dll.
    """
    if reader == "spark":
        return (spark.read.option("header", True).option("recursiveFileLookup", True)
                .csv(str(prod_dir)))
    files = sorted(Path(prod_dir).glob("date=*/wells_*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {prod_dir}")
    pdf = pd.concat((pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
                     for f in files), ignore_index=True)
    rows = [tuple(None if pd.isna(v) else v for v in r)
            for r in pdf.itertuples(index=False, name=None)]
    schema = StructType([StructField(c, StringType(), True) for c in pdf.columns])
    return spark.createDataFrame(rows, schema)


# ---------------------------------------------------------------- cleansing
def cleanse(raw):
    """Cast types, drop duplicates, null out impossible values. Returns (clean_df, stats)."""
    df = raw.withColumn("timestamp", F.to_timestamp("timestamp"))
    df = df.withColumn("choke_size", F.col("choke_size").cast("int"))
    for c in SENSOR_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))

    stats = {"raw_rows": df.count()}
    df = df.dropDuplicates(["well_id", "timestamp"]).cache()
    stats["rows_after_dedupe"] = df.count()
    stats["duplicates_removed"] = stats["raw_rows"] - stats["rows_after_dedupe"]

    oor = [F.sum(((F.col(c) < lo) | (F.col(c) > hi)).cast("int")).alias(c)
           for c, (lo, hi) in VALID_RANGES.items()]
    stats["out_of_range"] = {k: int(v or 0) for k, v in df.agg(*oor).first().asDict().items()}

    for c, (lo, hi) in VALID_RANGES.items():
        df = df.withColumn(c, F.when(F.col(c).between(lo, hi), F.col(c)))  # else NULL

    nulls = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in SENSOR_COLS]
    stats["nulls_after_cleaning"] = {k: int(v) for k, v in df.agg(*nulls).first().asDict().items()}
    return df, stats


def count_missing_rows(clean, freq_min):
    r = clean.agg(F.countDistinct("well_id").alias("wells"),
                  F.min("timestamp").cast("long").alias("t0"),
                  F.max("timestamp").cast("long").alias("t1")).first()
    expected = r["wells"] * ((r["t1"] - r["t0"]) // (freq_min * 60) + 1)
    return int(expected - clean.count())


# ---------------------------------------------------------------- detection
def hourly_rollup(clean):
    """15-min readings -> one row per well-hour. Values come from PRODUCING readings only."""
    producing = F.col("status") == "PRODUCING"
    return (clean.withColumn("hour_ts", F.date_trunc("hour", "timestamp"))
            .groupBy("well_id", "hour_ts")
            .agg(F.avg(F.when(producing, F.col("tubing_pressure_psi"))).alias("tubing_pressure_psi"),
                 F.avg(F.when(producing, F.col("oil_rate_bbl_d"))).alias("oil_rate_bbl_d"),
                 F.count("*").alias("n_readings"),
                 F.sum((F.col("status") == "SHUT_IN").cast("int")).alias("n_shut_in"))
            .withColumn("is_shut_in", F.col("n_shut_in") / F.col("n_readings") > 0.5))


def flag_hours(hourly):
    """Compare each hour to a trailing per-well baseline (mean/std of the previous days)."""
    w = (Window.partitionBy("well_id").orderBy(F.col("hour_ts").cast("long"))
         .rangeBetween(-(BASELINE_DAYS * 86400 + BASELINE_LAG_HOURS * 3600),
                       -BASELINE_LAG_HOURS * 3600))
    df = hourly
    for c, short in [("tubing_pressure_psi", "tp"), ("oil_rate_bbl_d", "oil")]:
        df = (df.withColumn(f"{short}_base", F.avg(c).over(w))
                .withColumn(f"{short}_std", F.stddev(c).over(w))
                .withColumn(f"{short}_n", F.count(c).over(w))
                .withColumn(f"{short}_drop", (F.col(f"{short}_base") - F.col(c)) / F.col(f"{short}_base"))
                .withColumn(f"{short}_z", (F.col(c) - F.col(f"{short}_base")) / F.col(f"{short}_std"))
                .withColumn(f"{short}_flag",
                            (~F.col("is_shut_in")) & (F.col(f"{short}_n") >= 48)
                            & (F.col(f"{short}_z") <= -Z_THRESHOLD)
                            & (F.col(f"{short}_drop") >= MIN_DROP_PCT)))
    # a pressure drop drags flow down too; call that one event, explained by pressure
    return df.withColumn("flow_flag", F.col("oil_flag") & ~F.col("tp_flag"))


def consolidate(hours, flag_col, severity_col, event_type):
    """Turn flagged hours into events: merge neighbours, compute start/end/severity."""
    h = hours.filter(F.col(flag_col)).select("well_id", "hour_ts", F.col(severity_col).alias("sev"))
    w = Window.partitionBy("well_id").orderBy("hour_ts")
    gap = F.col("hour_ts").cast("long") - F.col("prev_ts").cast("long")
    h = (h.withColumn("prev_ts", F.lag("hour_ts").over(w))
          .withColumn("new_event", (F.col("prev_ts").isNull() | (gap > MERGE_GAP_HOURS * 3600)).cast("int"))
          .withColumn("event_no", F.sum("new_event").over(w)))
    return (h.groupBy("well_id", "event_no")
            .agg(F.min("hour_ts").alias("start_ts"),
                 (F.max("hour_ts") + F.expr("INTERVAL 1 HOUR")).alias("end_ts"),
                 F.count("*").alias("duration_hours"),
                 F.max("sev").alias("max_drop_pct"))
            .filter(F.col("duration_hours") >= MIN_EVENT_HOURS)
            .withColumn("anomaly_type", F.lit(event_type))
            .drop("event_no"))


def detect_events(hours):
    hours = hours.withColumn("shut_in_sev", F.lit(1.0))
    ev = (consolidate(hours, "tp_flag", "tp_drop", "pressure_drop")
          .unionByName(consolidate(hours, "flow_flag", "oil_drop", "flow_decline"))
          .unionByName(consolidate(hours, "is_shut_in", "shut_in_sev", "shut_in")))
    ev = ev.withColumn("severity", F.when(F.col("max_drop_pct") >= 0.35, "High")
                                    .when(F.col("max_drop_pct") >= 0.20, "Medium").otherwise("Low"))
    return ev.withColumn("anomaly_id", F.concat(F.lit("D"), F.format_string("%04d", F.row_number().over(
        Window.orderBy("start_ts", "well_id")))))


# ---------------------------------------------------------------- curated metrics
def daily_metrics(clean, events):
    producing = (F.col("status") == "PRODUCING").cast("int")
    daily = (clean.withColumn("date", F.to_date("timestamp"))
             .groupBy("well_id", "date")
             .agg(F.avg("tubing_pressure_psi").alias("avg_tubing_pressure_psi"),
                  F.min("tubing_pressure_psi").alias("min_tubing_pressure_psi"),
                  F.avg("oil_rate_bbl_d").alias("daily_oil_bbl"),
                  F.avg("gas_rate_mcf_d").alias("daily_gas_mcf"),
                  F.avg("water_cut_pct").alias("avg_water_cut_pct"),
                  (F.sum(producing) / F.count("*") * 24).alias("uptime_hours"))
             .withColumn("downtime_hours", 24 - F.col("uptime_hours")))
    w7 = Window.partitionBy("well_id").orderBy("date").rowsBetween(-6, 0)
    daily = daily.withColumn("oil_bbl_7d_avg", F.avg("daily_oil_bbl").over(w7))

    ev_days = (events.withColumn("date", F.explode(F.sequence(
                   F.to_date("start_ts"), F.to_date(F.col("end_ts") - F.expr("INTERVAL 1 SECOND")))))
               .groupBy("well_id", "date").agg(F.count("*").alias("anomaly_count")))
    return (daily.join(ev_days, ["well_id", "date"], "left")
            .withColumn("anomaly_count", F.coalesce("anomaly_count", F.lit(0)))
            .orderBy("well_id", "date"))


# ---------------------------------------------------------------- evaluation
def evaluate(pred, truth):
    """Event-level scoring: a true event is 'caught' if any detection overlaps it on the same well."""
    pred, truth = pred.copy(), truth.copy()
    for d in (pred, truth):
        d["start_ts"] = pd.to_datetime(d["start_ts"], utc=True)
        d["end_ts"] = pd.to_datetime(d["end_ts"], utc=True)
    truth["end_ts"] = truth["end_ts"] + pd.Timedelta(minutes=15)

    m = truth.merge(pred, on="well_id", suffixes=("_t", "_p"))
    m = m[(m.start_ts_p < m.end_ts_t) & (m.end_ts_p > m.start_ts_t)]
    m["delay_h"] = (m.start_ts_p - m.start_ts_t).dt.total_seconds() / 3600

    print("\nEVENT-LEVEL RESULTS vs ground truth")
    print(f"{'type':15s}{'true':>6s}{'caught':>8s}{'recall':>8s}{'right type':>12s}{'avg delay (h)':>15s}")
    for t in ["pressure_drop", "flow_decline", "shut_in"]:
        tt = truth[truth.anomaly_type == t]
        mm = m[m.anomaly_type_t == t]
        caught = mm.anomaly_id_t.nunique()
        right = mm[mm.anomaly_type_p == t].anomaly_id_t.nunique()
        delay = mm.sort_values("delay_h").groupby("anomaly_id_t").delay_h.first().mean()
        print(f"{t:15s}{len(tt):6d}{caught:8d}{caught / max(len(tt), 1):8.0%}{right:12d}{delay:15.1f}")

    matched_pred = m.anomaly_id_p.nunique()
    precision = matched_pred / max(len(pred), 1)
    print(f"\nDetections: {len(pred)} | matched a real event: {matched_pred} | precision: {precision:.0%}")
    false_pos = pred[~pred.anomaly_id.isin(m.anomaly_id_p)]
    if len(false_pos):
        print("False positives:\n", false_pos[["anomaly_id", "well_id", "anomaly_type", "start_ts", "duration_hours"]]
              .to_string(index=False))
    return precision


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--freq-min", type=int, default=15)
    ap.add_argument("--reader", choices=["spark", "pandas"],
                    default="pandas" if os.name == "nt" else "spark",
                    help="how to load the CSVs (default: pandas on Windows, spark elsewhere)")
    args = ap.parse_args()
    data = Path(args.data)

    spark = (SparkSession.builder.master("local[*]").appName("oil-well-prototype")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.execution.arrow.pyspark.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    raw = read_raw(spark, data / "raw" / "production", args.reader)
    clean, stats = cleanse(raw)
    stats["missing_readings"] = count_missing_rows(clean, args.freq_min)

    injected = {}
    qpath = data / "ground_truth" / "quality_issues.json"
    if qpath.exists():
        injected = json.loads(qpath.read_text())
    print("DATA QUALITY REPORT (found vs injected)")
    print(f"  raw rows ............ {stats['raw_rows']:,}")
    print(f"  duplicates removed .. {stats['duplicates_removed']:,}  (injected: {injected.get('duplicates_injected', '?')})")
    print(f"  missing readings .... {stats['missing_readings']:,}  (injected: {injected.get('rows_dropped', '?')})")
    print(f"  out-of-range values . {sum(stats['out_of_range'].values()):,}  (injected: {injected.get('out_of_range_injected', '?')})")
    print(f"  nulls per column .... {list(stats['nulls_after_cleaning'].values())}  "
          f"(injected: {injected.get('nulls_injected', {}).get('oil_rate_bbl_d', '?')} each, before cleaning added out-of-range)")

    hours = flag_hours(hourly_rollup(clean)).cache()
    events = detect_events(hours).cache()
    pred = events.toPandas()

    truth = pd.read_csv(data / "ground_truth" / "anomaly_labels.csv")
    evaluate(pred, truth)

    curated = data / "curated"
    curated.mkdir(exist_ok=True)
    daily_metrics(clean, events).toPandas().round(2).to_csv(curated / "daily_well_metrics.csv", index=False)
    pred.sort_values(["start_ts", "well_id"]).to_csv(curated / "anomalies.csv", index=False)
    print(f"\nWrote curated CSVs to {curated.resolve()}")
    spark.stop()


if __name__ == "__main__":
    main()