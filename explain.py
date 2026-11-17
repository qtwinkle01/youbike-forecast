"""What drives the prediction, and what-if scenarios.

Run:  docker compose run --rm ml python explain.py      (after train.py)

Part 1  Drivers.  SHAP values split every single prediction into "how many bikes
        each feature added or removed". Averaging their size over many predictions
        tells us which factors the model leans on. Features are also grouped into
        families (current state, time, weather, ...) because that is what people ask about.
Part 2  What-if.  Take real rows, change ONE thing, predict again, compare.
          A. Same weekday 08:00 / 18:00 situation, but pretend it is a Sunday.
          B. Same situation, but pretend it rains 5 mm/h.  (Flagged: we barely saw rain.)
          C. Same situation, but a truck adds 10 bikes now. How many are left in an hour?

Uses out/model_full.txt (trained on every row). Writes out/explain.json.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train import FEATURES

OUT = Path("out")
FAMILIES = {
    "current state": ["bikes_now", "docks_free_now", "fill_now", "capacity"],
    "recent trend": ["delta_15m", "delta_60m"],
    "yesterday": ["bikes_yday_now", "bikes_yday_target", "yday_delta_60m"],
    "time": ["hour", "dow", "is_weekend"],
    "weather": ["cur_temp", "cur_precip", "cur_humidity", "cur_wind",
                "fc_temp", "fc_precip", "fc_precip_prob"],
    "neighbours": ["nbr_bikes_mean", "nbr_fill_mean", "nbr_delta_60m", "nbr_km_mean"],
    "location": ["lat", "lng", "is_taipei"],
}
SAMPLE = 200_000


def drivers(model, df):
    x = df[FEATURES].sample(min(SAMPLE, len(df)), random_state=0)
    shap = model.predict(x, pred_contrib=True)[:, :-1]          # last column = base value
    per_feature = pd.Series(np.abs(shap).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    idx = {f: i for i, f in enumerate(FEATURES)}
    fam = {name: float(np.abs(shap[:, [idx[f] for f in cols]].sum(axis=1)).mean())
           for name, cols in FAMILIES.items()}
    fam = pd.Series(fam).sort_values(ascending=False)

    print("Drivers: average size of each factor's push on the 1-hour change (bikes)")
    for name, v in fam.items():
        print(f"  {name:14s} {v:5.2f}  {'#' * int(round(v * 40))}")
    print("\nTop single features (bikes):")
    print(per_feature.head(10).round(3).to_string())

    # when does "time" matter most?
    t = np.abs(shap[:, [idx[f] for f in FAMILIES["time"]]].sum(axis=1))
    hours = df.loc[x.index, "hour"].astype(int)
    wk = df.loc[x.index, "is_weekend"].map({0: "weekday", 1: "weekend"})
    by_hour = pd.Series(t, index=x.index).groupby([wk, hours]).mean().unstack(0).round(2)
    print("\nSize of the 'time' push by hour of day (bikes):")
    print(by_hour.to_string())

    # direction of the strongest single feature: fuller stations lose bikes, emptier ones gain
    fill_shap = pd.Series(shap[:, idx["fill_now"]], index=x.index)
    bins = pd.cut(x["fill_now"], [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 5])
    direction = fill_shap.groupby(bins, observed=True).mean().round(2)
    print("\nPush from 'how full the station is now' (bikes, + means gains bikes):")
    print(direction.to_string())
    return {"families": fam.round(4).to_dict(), "features": per_feature.round(4).to_dict(),
            "time_push_by_hour": {k: v.dropna().to_dict() for k, v in by_hour.items()},
            "fill_direction": {str(k): float(v) for k, v in direction.items()}}


def rows_at(df, hour, weekend):
    """Real rows at a given local hour (±10 min) from the latest matching day."""
    m = (df.is_weekend == int(weekend)) & ((df.hour - hour).abs() <= 10 / 60)
    part = df[m]
    last_day = part.slot_local.dt.date.max()
    part = part[part.slot_local.dt.date == last_day]
    return part.sort_values("slot_local").groupby("sno").tail(1), last_day


def what_if_daytype(model, df, stations, hour):
    base, day = rows_at(df, hour, weekend=False)
    alt = base.copy()
    alt["dow"], alt["is_weekend"] = 6, 1                      # pretend it is a Sunday
    p_base = model.predict(base[FEATURES])
    p_alt = model.predict(alt[FEATURES])
    diff = pd.Series(p_base - p_alt, index=base.sno.to_numpy())    # weekday minus Sunday
    res = pd.DataFrame({"sno": base.sno.to_numpy(), "bikes_now": base.bikes_now.to_numpy(),
                        "weekday": p_base, "as_sunday": p_alt, "diff": diff.to_numpy()})
    res = res.merge(stations[["sno", "name", "district"]], on="sno")
    print(f"\nWhat-if A  {day} {hour:02d}:00, {len(res):,} stations: same situation, weekday vs 'pretend Sunday'")
    print(f"  predicted 1-hour change, average over stations: weekday {p_base.mean():+.2f}, Sunday {p_alt.mean():+.2f}")
    print(f"  stations where the day type moves the forecast by 1+ bike: {(diff.abs() >= 1).mean():.1%}"
          f"   by 3+ bikes: {(diff.abs() >= 3).mean():.1%}")
    cols = ["name", "district", "bikes_now", "weekday", "as_sunday", "diff"]
    lose = res.nsmallest(5, "diff")[cols].round(1)
    gain = res.nlargest(5, "diff")[cols].round(1)
    print("  loses more bikes on a weekday:"); print(lose.to_string(index=False))
    print("  gains more bikes on a weekday:"); print(gain.to_string(index=False))
    return {"day": str(day), "hour": hour, "stations": len(res),
            "mean_change_weekday": float(p_base.mean()), "mean_change_as_sunday": float(p_alt.mean()),
            "share_moved_1plus": float((diff.abs() >= 1).mean()),
            "share_moved_3plus": float((diff.abs() >= 3).mean()),
            "top_lose": lose.to_dict("records"), "top_gain": gain.to_dict("records")}


def what_if_rain(model, df, hour, mm=5.0):
    base, day = rows_at(df, hour, weekend=False)
    dry, wet = base.copy(), base.copy()
    dry[["cur_precip", "fc_precip", "fc_precip_prob"]] = [0.0, 0.0, 0.0]
    wet[["cur_precip", "fc_precip", "fc_precip_prob"]] = [mm, mm, 100.0]
    d = model.predict(wet[FEATURES]) - model.predict(dry[FEATURES])
    seen = float(df.fc_precip.max())
    print(f"\nWhat-if B  {day} {hour:02d}:00: dry vs {mm} mm/h rain")
    print(f"  average change in forecast: {d.mean():+.2f} bikes; average size {np.abs(d).mean():.2f}")
    print(f"  CAUTION: heaviest rain in the data is {seen:.1f} mm/h and "
          f"{(df.cur_precip > 0).mean():.0%} of rows have any rain, so {mm} mm/h is outside what the model saw.")
    return {"day": str(day), "hour": hour, "mm": mm, "mean_diff": float(d.mean()),
            "mean_abs_diff": float(np.abs(d).mean()), "max_precip_seen": seen,
            "share_rows_with_rain": float((df.cur_precip > 0).mean())}


def what_if_refill(model, df, stations, hour, k=10):
    """A truck drops k bikes at the station right now. How many extra bikes are there in an hour?"""
    base, day = rows_at(df, hour, weekend=False)
    base = base[base.docks_free_now >= k]                       # only where k bikes fit
    more = base.copy()
    more["bikes_now"] += k
    more["docks_free_now"] -= k
    more["fill_now"] = more.bikes_now / np.maximum(more.capacity, 1)
    more[["delta_15m", "delta_60m"]] += k                       # the jump is visible in the recent trend too
    with_k = np.clip(more.bikes_now.to_numpy() + model.predict(more[FEATURES]), 0, None)
    without = np.clip(base.bikes_now.to_numpy() + model.predict(base[FEATURES]), 0, None)
    kept = pd.Series(with_k - without, index=base.sno.to_numpy())
    res = pd.DataFrame({"sno": base.sno.to_numpy(), "bikes_now": base.bikes_now.to_numpy(),
                        "in_1h_without": without, "in_1h_with": with_k, "kept": kept.to_numpy()})
    res = res.merge(stations[["sno", "name", "district"]], on="sno")
    print(f"\nWhat-if C  {day} {hour:02d}:00, {len(res):,} stations: a truck adds {k} bikes now")
    print(f"  extra bikes still there one hour later: average {kept.mean():.1f} of {k}"
          f"   (10% of stations keep under {kept.quantile(0.1):.1f}, 10% keep over {kept.quantile(0.9):.1f})")
    cols = ["name", "district", "bikes_now", "in_1h_without", "in_1h_with", "kept"]
    worst = res.nsmallest(5, "kept")[cols].round(1)
    print("  stations where the added bikes disappear fastest:"); print(worst.to_string(index=False))
    return {"day": str(day), "hour": hour, "k": k, "stations": len(res),
            "kept_mean": float(kept.mean()), "kept_p10": float(kept.quantile(0.1)),
            "kept_p90": float(kept.quantile(0.9)), "fastest_drain": worst.to_dict("records")}


def main():
    model = lgb.Booster(model_file=str(OUT / "model_full.txt"))
    df = pd.read_parquet(OUT / "model_table.parquet")
    stations = pd.read_parquet(OUT / "stations.parquet")
    print(f"rows {len(df):,}  {df.slot_local.min()} -> {df.slot_local.max()}")
    print("rows per weekday (0=Mon):", df.groupby("dow").size().to_dict(), "\n")
    out = {"drivers": drivers(model, df),
           "what_if_daytype": [what_if_daytype(model, df, stations, h) for h in (8, 18)],
           "what_if_rain": what_if_rain(model, df, 8),
           "what_if_refill": [what_if_refill(model, df, stations, h) for h in (8, 18)]}
    (OUT / "explain.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
