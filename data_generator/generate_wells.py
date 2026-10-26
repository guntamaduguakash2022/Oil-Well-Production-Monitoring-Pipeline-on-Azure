"""
Simulated oil well production data generator.

Writes (under --out):
  raw/reference/well_master.csv
  raw/production/date=YYYY-MM-DD/wells_YYYYMMDD.csv   one file per day
  ground_truth/anomaly_labels.csv    keep OUT of the pipeline; used only to evaluate your detector
  ground_truth/quality_issues.json   what data-quality problems were injected

Usage:
  pip install pandas numpy
  python generate_wells.py --wells 55 --days 90 --freq-min 15 --seed 42 --out ./data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["North Ridge", "Eastern Basin", "South Flats"]
SENSOR_COLS = ["tubing_pressure_psi", "casing_pressure_psi", "oil_rate_bbl_d",
               "gas_rate_mcf_d", "water_cut_pct", "temperature_f"]
# anomaly type -> (min hours, max hours)
EVENT_TYPES = {"pressure_drop": (4, 48), "flow_decline": (48, 120), "shut_in": (6, 72)}
WARMUP_DAYS = 7  # keep the first week clean so rolling baselines have history


def build_well_master(n_wells, rng, start):
    fields = rng.choice(FIELDS, n_wells)
    install = start - pd.to_timedelta(rng.integers(180, 1800, n_wells), unit="D")
    return pd.DataFrame({
        "well_id": [f"W{i:03d}" for i in range(1, n_wells + 1)],
        "well_name": [f"{f.replace(' ', '_').upper()}_{i:02d}" for i, f in enumerate(fields, 1)],
        "field": fields,
        "well_type": rng.choice(["Vertical", "Horizontal"], n_wells, p=[0.4, 0.6]),
        "install_date": install.strftime("%Y-%m-%d"),
        "latitude": rng.uniform(31.5, 32.5, n_wells).round(5),
        "longitude": rng.uniform(-103.5, -102.0, n_wells).round(5),
    })


def well_params(rng):
    annual_decline = rng.uniform(0.15, 0.35)
    return {
        "base_tp": rng.uniform(800, 2200),
        "csg_ratio": rng.uniform(1.1, 1.4),
        "base_oil": rng.uniform(150, 900),
        "gor": rng.uniform(500, 2000),
        "base_wc": rng.uniform(5, 35),
        "wc_growth": rng.uniform(0.02, 0.10),
        "base_temp": rng.uniform(140, 220),
        "choke": int(rng.choice(range(16, 49, 4))),
        "decline_d": -np.log(1 - annual_decline) / 365,
    }


def make_events(n_steps, steps_per_hour, rng):
    """Pick 0-3 non-overlapping anomaly events as (type, start_idx, end_idx)."""
    warmup = WARMUP_DAYS * 24 * steps_per_hour
    pad = 24 * steps_per_hour
    occupied = np.zeros(n_steps, dtype=bool)
    events = []
    for _ in range(rng.choice([0, 1, 2, 3], p=[0.2, 0.35, 0.3, 0.15])):
        etype = str(rng.choice(list(EVENT_TYPES)))
        lo, hi = EVENT_TYPES[etype]
        dur = int(rng.integers(lo, hi + 1)) * steps_per_hour
        if n_steps - dur - 1 <= warmup:
            continue
        for _ in range(20):
            start = int(rng.integers(warmup, n_steps - dur - 1))
            if not occupied[max(0, start - pad): start + dur + pad].any():
                occupied[start:start + dur] = True
                events.append((etype, start, start + dur))
                break
    return events


def simulate_well(p, ts, events, steps_per_hour, rng):
    n = len(ts)
    days = np.arange(n) / (24 * steps_per_hour)
    decline = np.exp(-p["decline_d"] * days)

    tp = p["base_tp"] * np.exp(-0.3 * p["decline_d"] * days) * (1 + rng.normal(0, 0.015, n))
    oil_mult = np.ones(n)
    shut = np.zeros(n, dtype=bool)

    for etype, s, e in events:
        if etype == "pressure_drop":
            tp[s:e] *= 1 - rng.uniform(0.25, 0.50)
        elif etype == "flow_decline":
            oil_mult[s:e] *= np.linspace(1, 1 - rng.uniform(0.20, 0.40), e - s)
        else:
            shut[s:e] = True

    # flow follows pressure, so a pressure drop also drags production down
    oil = (p["base_oil"] * decline * (tp / p["base_tp"]) ** 0.8
           * oil_mult * (1 + rng.normal(0, 0.03, n)))
    gas = oil * p["gor"] / 1000
    csg = tp * p["csg_ratio"] * (1 + rng.normal(0, 0.01, n))
    wc = np.clip(p["base_wc"] + p["wc_growth"] * days + rng.normal(0, 0.5, n), 0, 95)
    hour = ts.hour.values + ts.minute.values / 60
    temp = p["base_temp"] + 2 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.6, n)

    # shut-in: no flow, pressure builds up
    oil[shut] = 0
    gas[shut] = 0
    tp[shut] *= 1.15
    csg[shut] *= 1.05

    return pd.DataFrame({
        "timestamp": ts,
        "tubing_pressure_psi": tp.round(1),
        "casing_pressure_psi": csg.round(1),
        "oil_rate_bbl_d": oil.round(2),
        "gas_rate_mcf_d": gas.round(2),
        "water_cut_pct": wc.round(2),
        "temperature_f": temp.round(1),
        "choke_size": p["choke"],
        "status": np.where(shut, "SHUT_IN", "PRODUCING"),
    })


def inject_quality_issues(df, rng):
    """Add realistic mess for the cleansing step to find. Returns (df, summary)."""
    summary = {}

    # missing readings (rows never arrived)
    drop = rng.choice(len(df), int(len(df) * 0.002), replace=False)
    df = df.drop(df.index[drop]).reset_index(drop=True)
    summary["rows_dropped"] = int(len(drop))

    # random nulls in sensor columns
    summary["nulls_injected"] = {}
    for col in SENSOR_COLS:
        idx = rng.choice(len(df), int(len(df) * 0.003), replace=False)
        df.loc[idx, col] = np.nan
        summary["nulls_injected"][col] = int(len(idx))

    # out-of-range values (sentinel pressure, absurd flow rate)
    idx = rng.choice(len(df), int(len(df) * 0.0005), replace=False)
    half = len(idx) // 2
    df.loc[idx[:half], "tubing_pressure_psi"] = -999
    df.loc[idx[half:], "oil_rate_bbl_d"] = df.loc[idx[half:], "oil_rate_bbl_d"] * 50
    summary["out_of_range_injected"] = int(len(idx))

    # exact duplicate rows
    dup = rng.choice(len(df), int(len(df) * 0.001), replace=False)
    df = pd.concat([df, df.iloc[dup]], ignore_index=True)
    summary["duplicates_injected"] = int(len(dup))

    df = df.sort_values(["timestamp", "well_id"], kind="stable").reset_index(drop=True)
    return df, summary


def main():
    ap = argparse.ArgumentParser(description="Generate simulated oil well data")
    ap.add_argument("--wells", type=int, default=55)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--freq-min", type=int, default=15, choices=[1, 5, 10, 15, 30, 60])
    ap.add_argument("--start-date", default="2026-07-01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./data")
    args = ap.parse_args()

    out = Path(args.out)
    steps_per_hour = 60 // args.freq_min
    start = pd.Timestamp(args.start_date)
    ts = pd.date_range(start, periods=args.days * 24 * steps_per_hour, freq=f"{args.freq_min}min")
    rng = np.random.default_rng(args.seed)

    master = build_well_master(args.wells, rng, start)
    frames, labels = [], []
    for wid in master["well_id"]:
        p = well_params(rng)
        events = make_events(len(ts), steps_per_hour, rng)
        w = simulate_well(p, ts, events, steps_per_hour, rng)
        w.insert(0, "well_id", wid)
        frames.append(w)
        for etype, s, e in events:
            labels.append({"well_id": wid, "anomaly_type": etype,
                           "start_ts": ts[s].strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "end_ts": ts[e - 1].strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "duration_hours": (e - s) / steps_per_hour})

    df = pd.concat(frames, ignore_index=True)
    df, quality = inject_quality_issues(df, rng)

    # write reference + ground truth
    (out / "raw" / "reference").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    master.to_csv(out / "raw" / "reference" / "well_master.csv", index=False)
    labels_df = pd.DataFrame(labels).sort_values(["start_ts", "well_id"]).reset_index(drop=True)
    labels_df.insert(0, "anomaly_id", [f"A{i:04d}" for i in range(1, len(labels_df) + 1)])
    labels_df.to_csv(out / "ground_truth" / "anomaly_labels.csv", index=False)
    (out / "ground_truth" / "quality_issues.json").write_text(json.dumps(quality, indent=2))

    # one raw file per day, in date-partitioned folders (mirrors the ADLS raw zone)
    day = df["timestamp"].dt.strftime("%Y-%m-%d")
    for d, g in df.groupby(day):
        folder = out / "raw" / "production" / f"date={d}"
        folder.mkdir(parents=True, exist_ok=True)
        g.to_csv(folder / f"wells_{d.replace('-', '')}.csv", index=False,
                 date_format="%Y-%m-%dT%H:%M:%SZ")

    print(f"Wells: {args.wells} | days: {args.days} | rows: {len(df):,} | daily files: {day.nunique()}")
    print("Injected anomalies:", labels_df["anomaly_type"].value_counts().to_dict())
    print("Injected quality issues:", json.dumps(quality))
    print(f"Output written to: {out.resolve()}")


if __name__ == "__main__":
    main()  