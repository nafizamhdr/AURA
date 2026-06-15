"""Continuous Training (CT) for the AURA RUL regressor, with a quality gate.

Flow:
  load data -> build 18-feature matrix -> fit scaler + RandomForest ->
  evaluate (MAE/R2 + fairness per Type) -> compare against gate thresholds.

Artifacts are written ONLY when the model passes the gate AND --promote is
passed, so an accidental run never overwrites the shipped model. Without
--promote it is a dry-run that just reports metrics and the gate decision.

Examples:
  python -m ct.train --nrows 20000            # fast dry-run
  python -m ct.train --promote                # full retrain + save if gate passes
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Src"))

from Preprocessing_pipeline import PreprocessingPipeline  # noqa: E402
from ct.evaluate import regression_metrics, mae_by_group  # noqa: E402


def load_xy(nrows=None):
    df = pd.read_csv(ROOT / "Data" / "predictive_maintenance.csv", nrows=nrows)
    X = PreprocessingPipeline().transform_batch(df.to_dict("records"))
    y = df["RUL_hours"].to_numpy()
    return df, X, y


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="AURA continuous training with quality gate")
    ap.add_argument("--nrows", type=int, default=None, help="limit rows (for a fast run)")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-mae-days", type=float, default=10.0, help="gate: max test MAE in days")
    ap.add_argument("--min-r2", type=float, default=0.95, help="gate: min test R2")
    ap.add_argument("--max-type-gap-days", type=float, default=5.0,
                    help="gate: max MAE gap (days) across machine Type H/M/L")
    ap.add_argument("--promote", action="store_true",
                    help="write artifacts if the gate passes (otherwise dry-run)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    df, X, y = load_xy(args.nrows)
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, df, test_size=0.2, random_state=42
    )

    scaler = StandardScaler().fit(X_tr)
    model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
    model.fit(scaler.transform(X_tr), y_tr)

    pred = model.predict(scaler.transform(X_te))
    perf = regression_metrics(y_te, pred)
    mae_days = perf["mae"] / 24
    per_type, gap_hours = mae_by_group(df_te["Type"].to_numpy(), y_te, pred)
    gap_days = gap_hours / 24

    print(f"Test MAE : {mae_days:6.2f} days")
    print(f"Test R2  : {perf['r2']:.4f}")
    print("MAE per Type (days):",
          {k: round(v / 24, 2) for k, v in sorted(per_type.items())})
    print(f"Fairness gap across Type: {gap_days:.2f} days")

    checks = {
        "mae_days <= max": mae_days <= args.max_mae_days,
        "r2 >= min": perf["r2"] >= args.min_r2,
        "type_gap_days <= max": gap_days <= args.max_type_gap_days,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print("QUALITY GATE:", "PASS" if passed else "FAIL")

    if not args.promote:
        # Evaluation / dry-run: report the decision but never fail the run.
        # In continuous training, a failing gate simply means "do not promote"
        # (keep the current production model) -- that is a normal outcome, not
        # a build error.
        print("Dry-run: model eligible for promotion."
              if passed else
              "Dry-run: gate failed -> model would NOT be promoted.")
        return 0

    # --promote: a failing gate blocks the write and is signalled as an error.
    if not passed:
        print("Gate failed -> not promoting; keeping the current model.")
        return 1

    # Gate passed and --promote requested: write the new artifacts.
    joblib.dump(scaler, ROOT / "Model" / "scaler.pkl")
    joblib.dump(model, ROOT / "Model" / "rul_model.pkl")
    meta_path = ROOT / "Model" / "model_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "retrained_at": datetime.now().isoformat(),
        "performance": {
            "test_mae_hours": perf["mae"],
            "test_mae_days": mae_days,
            "test_r2": perf["r2"],
            "fairness_type_gap_days": gap_days,
        },
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Promoted: saved scaler.pkl, rul_model.pkl, updated model_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
