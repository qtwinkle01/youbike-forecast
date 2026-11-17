"""Baselines vs LightGBM with a time-ordered split.

Run:  docker compose run --rm ml python train.py            # test = last 24 h
      docker compose run --rm ml python train.py --test-hours 48

Split (by the time being PREDICTED, so nothing from the test period leaks back):

    |------------- train -------------|-- valid --|  1 h gap  |---- test ----|
     oldest data                        last 15 %               newest N hours

  * test  : rows whose slot is in the last N hours.
  * train : rows whose TARGET time (slot + 1 h) is before the test period starts.
            The 1-hour gap exists because a row at 09:30 "knows" the 10:30 answer.
  * valid : the newest 15 % of train, used only to decide when to stop adding trees.

Baselines
  now  : "one hour from now it will be the same as now"
  yday : "it will be the same as yesterday at that time"
The model predicts the CHANGE over the next hour and adds it to the current count.

Outputs
  out/model.txt        trained on train only  -> the honest scores in out/metrics.json
  out/model_full.txt   same settings, refit on every row -> used by explain.py and the app
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

OUT = Path("out")
FEATURES = [
    "bikes_now", "docks_free_now", "fill_now", "capacity",
    "delta_15m", "delta_60m",
    "bikes_yday_now", "bikes_yday_target", "yday_delta_60m",
    "hour", "dow", "is_weekend",
    "cur_temp", "cur_precip", "cur_humidity", "cur_wind",
    "fc_temp", "fc_precip", "fc_precip_prob",
    "nbr_bikes_mean", "nbr_fill_mean", "nbr_delta_60m", "nbr_km_mean",
    "lat", "lng", "is_taipei",
]
PARAMS = dict(objective="regression_l1", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)


def split(df, test_hours):
    end = df.slot_utc.max()
    test_start = end - pd.Timedelta(hours=test_hours) + pd.Timedelta(minutes=5)
    test = df[df.slot_utc >= test_start]
    train_all = df[df.slot_utc + pd.Timedelta(hours=1) < test_start]
    slots = np.sort(train_all.slot_utc.unique())
    valid_start = slots[int(len(slots) * 0.85)]
    valid = train_all[train_all.slot_utc >= valid_start]
    train = train_all[train_all.slot_utc + pd.Timedelta(hours=1) < valid_start]
    return train, valid, test


def scores(y, pred):
    err = pred - y
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}


def event_scores(actual, flagged):
    tp = int((actual & flagged).sum())
    return {"events": int(actual.sum()), "flagged": int(flagged.sum()),
            "precision": tp / max(int(flagged.sum()), 1), "recall": tp / max(int(actual.sum()), 1)}


def evaluate(test, preds):
    """preds: name -> predicted bikes. Returns metrics on all rows and on rows that have 'yesterday'."""
    y = test.y_bikes.to_numpy()
    has_yday = test.bikes_yday_target.notna().to_numpy()
    empty = y == 0
    full = test.y_docks_free.to_numpy() == 0
    out = {}
    for name, p in preds.items():
        ok = ~np.isnan(p)
        m = {"rows": int(ok.sum()), **scores(y[ok], p[ok])}
        both = ok & has_yday
        m["same_rows_as_yday"] = {"rows": int(both.sum()), **scores(y[both], p[both])}
        # "will it be empty / full in an hour?"  predicted free docks = free now - predicted change
        free_pred = test.docks_free_now.to_numpy() - (p - test.bikes_now.to_numpy())
        m["empty"] = event_scores(empty[ok], (p < 0.5)[ok])
        m["full"] = event_scores(full[ok], (free_pred < 0.5)[ok])
        out[name] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-hours", type=int, default=24)
    args = ap.parse_args()

    df = pd.read_parquet(OUT / "model_table.parquet")
    train, valid, test = split(df, args.test_hours)
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        print(f"{name:5s} {len(part):>9,} rows  {part.slot_local.min()} -> {part.slot_local.max()}")

    def xy(part):
        return part[FEATURES], (part.y_bikes - part.bikes_now)      # target = change in 1 h

    model = lgb.train(
        PARAMS, lgb.Dataset(*xy(train)), num_boost_round=2000,
        valid_sets=[lgb.Dataset(*xy(valid))],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    print("trees used:", model.best_iteration)

    cap = test.bikes_now.to_numpy() + np.maximum(test.docks_free_now.to_numpy(), 0)
    pred = test.bikes_now.to_numpy() + model.predict(test[FEATURES], num_iteration=model.best_iteration)
    pred = np.clip(pred, 0, np.maximum(cap, test.capacity.to_numpy()))   # cannot go below 0 or above the docks
    preds = {
        "baseline_now": test.bikes_now.to_numpy(dtype="float64"),
        "baseline_yday": test.bikes_yday_target.to_numpy(dtype="float64"),
        "lightgbm": pred,
    }
    metrics = evaluate(test, preds)

    base = metrics["baseline_now"]["mae"]
    print(f"\n{'model':15s} {'MAE':>7s} {'RMSE':>7s} {'vs now':>8s} | {'empty P/R':>12s} | {'full P/R':>12s}")
    for name, m in metrics.items():
        print(f"{name:15s} {m['mae']:7.3f} {m['rmse']:7.3f} {100 * (m['mae'] / base - 1):+7.1f}% | "
              f"{m['empty']['precision']:.2f} / {m['empty']['recall']:.2f}  | "
              f"{m['full']['precision']:.2f} / {m['full']['recall']:.2f}")
    print("\nOn the rows where 'yesterday' exists:")
    for name, m in metrics.items():
        s = m["same_rows_as_yday"]
        print(f"{name:15s} MAE {s['mae']:.3f}  ({s['rows']:,} rows)")

    # Where does the model help? Stations that barely move are easy for everyone.
    moved = np.abs(test.y_bikes - test.bikes_now).to_numpy() >= 3
    print(f"\nRows where the count changed by 3+ bikes ({moved.mean():.1%} of test):")
    for name, p in preds.items():
        ok = moved & ~np.isnan(p)
        print(f"{name:15s} MAE {np.abs(p[ok] - test.y_bikes.to_numpy()[ok]).mean():.3f}")

    gain = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    gain = (gain / gain.sum()).round(4)
    print("\nFeature importance (share of total gain):")
    print(gain.head(12).to_string())

    model.save_model(str(OUT / "model.txt"), num_iteration=model.best_iteration)

    # The scores above come from the split model. For the app and the explanations we
    # refit on ALL rows with the same number of trees, so the newest days are not wasted.
    full = lgb.train(PARAMS, lgb.Dataset(*xy(df)), num_boost_round=model.best_iteration)
    full.save_model(str(OUT / "model_full.txt"))
    print(f"\nrefit on all {len(df):,} rows -> out/model_full.txt")
    out = test[["slot_utc", "slot_local", "sno", "bikes_now", "y_bikes", "y_docks_free"]].copy()
    out["pred"] = pred
    out.to_parquet(OUT / "test_predictions.parquet", index=False)
    (OUT / "metrics.json").write_text(json.dumps({
        "test_hours": args.test_hours,
        "periods_local": {n: [str(p.slot_local.min()), str(p.slot_local.max())]
                          for n, p in (("train", train), ("valid", valid), ("test", test))},
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "trees": model.best_iteration, "features": FEATURES,
        "metrics": metrics, "importance_gain": gain.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
