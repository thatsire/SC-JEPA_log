"""
Build train/val/test CSVs from the raw Azure Predictive Maintenance files
(PdM_telemetry.csv, PdM_machines.csv, PdM_failures.csv).

Split is temporal and identical for every machine, so every model sees the same
machines in training and is tested on a later period:
    train : 2015-01-01 .. 2015-08-31
    val   : 2015-09-01 .. 2015-10-31
    test  : 2015-11-01 .. 2016-01-01
Windows are built inside each file (see data/datasets.py), so no window crosses
a split boundary and there is no leakage between splits.

Output columns: datetime, machineID, volt, rotate, pressure, vibration, age,
                model (int 0..3), fail_comp1..fail_comp4 (1 at the failure hour)
"""
import argparse
import os

import pandas as pd

COMPS = ["comp1", "comp2", "comp3", "comp4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="directory with PdM_*.csv")
    ap.add_argument("--out_dir", default="dataset")
    ap.add_argument("--val_start", default="2015-09-01")
    ap.add_argument("--test_start", default="2015-11-01")
    args = ap.parse_args()

    tel = pd.read_csv(os.path.join(args.raw_dir, "PdM_telemetry.csv"), parse_dates=["datetime"])
    fail = pd.read_csv(os.path.join(args.raw_dir, "PdM_failures.csv"), parse_dates=["datetime"])
    mach = pd.read_csv(os.path.join(args.raw_dir, "PdM_machines.csv"))

    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    mach["model"] = mach["model"].str.replace("model", "", regex=False).astype(int) - 1  # 0..3
    df = tel.merge(mach, on="machineID", how="left")

    for c in COMPS:
        f = (fail[fail.failure == c][["datetime", "machineID"]]
             .drop_duplicates().assign(**{f"fail_{c}": 1}))
        df = df.merge(f, on=["datetime", "machineID"], how="left")
        df[f"fail_{c}"] = df[f"fail_{c}"].fillna(0).astype(int)

    os.makedirs(args.out_dir, exist_ok=True)
    splits = {
        "train": df[df.datetime < args.val_start],
        "val": df[(df.datetime >= args.val_start) & (df.datetime < args.test_start)],
        "test": df[df.datetime >= args.test_start],
    }
    fail_cols = [f"fail_{c}" for c in COMPS]
    for name, part in splits.items():
        part = part.sort_values(["machineID", "datetime"])
        part.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        n_fail = int(part[fail_cols].max(axis=1).sum())
        print(f"{name:5s}: rows={len(part):7d} machines={part.machineID.nunique():3d} "
              f"failure-hours={n_fail:3d}  {part.datetime.min()} -> {part.datetime.max()}")


if __name__ == "__main__":
    main()
