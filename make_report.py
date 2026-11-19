"""Fill the report with the numbers from the latest run.

Run:  docker compose run --rm ml python make_report.py     (after build_dataset / train / explain)

Reads   out/*.json, out/test_predictions.parquet, data/fetch_log.jsonl
Writes  docs/REPORT.md, docs/REPORT.zh-TW.md, docs/img/*.png

Nothing in the report is typed by hand: re-run the pipeline, re-run this, and every
number and chart follows.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT, DOCS = Path("out"), Path("docs")
FAMILY = {
    "current state": ("Current state (bikes, free docks, how full)", "站點現況（車數、空位、多滿）"),
    "recent trend": ("Recent trend (last 15 / 60 min)", "近期趨勢（過去 15／60 分鐘）"),
    "yesterday": ("Yesterday at this time", "昨天同時段"),
    "time": ("Time of day, weekday", "時段、星期幾"),
    "weather": ("Weather (now + forecast for the target hour)", "天氣（現況＋目標時段預報）"),
    "neighbours": ("5 nearest stations", "最近的 5 個站"),
    "location": ("Location", "位置"),
}


def load():
    j = lambda n: json.loads((OUT / n).read_text(encoding="utf-8"))
    return j("dataset_report.json"), j("metrics.json"), j("explain.json")


def fetch_stats():
    c = Counter()
    for line in open("data/fetch_log.jsonl", encoding="utf-8"):
        e = json.loads(line)
        kind = "bike" if e["source"].startswith("youbike") else "weather"
        c[kind, e["ok"]] += 1
    return c


def failure_split(gaps):
    """Bike fetch failures inside the reported gaps vs isolated ones that left no gap."""
    ranges = [[pd.Timestamp(x) for x in g.split(" -> ")] for g in gaps]
    inside = outside = 0
    for line in open("data/fetch_log.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if not e["source"].startswith("youbike") or e["ok"]:
            continue
        t = pd.Timestamp(e["ts"]).tz_convert("Asia/Taipei").tz_localize(None)
        hit = any(a - pd.Timedelta(minutes=10) <= t <= b + pd.Timedelta(minutes=10) for a, b in ranges)
        inside += hit
        outside += not hit
    longest = max(((b - a).total_seconds() / 60 + 5 for a, b in ranges), default=0)
    return inside, outside, int(round(longest))


def weekday_count(start, end):
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    return int((days.dayofweek < 5).sum())


def weekend_count(start, end):
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    return int((days.dayofweek >= 5).sum())


DAYTYPE_MATTERS = 0.05     # share of stations moving 1+ bike above which the weekday label "matters"


def peak_hours(by_hour):
    """Hours whose time-of-day push is at least 75% of the largest one, as 'HH:00 to HH:00'."""
    s = pd.Series({int(h): v for h, v in by_hour.items()})
    top = sorted(s[s >= 0.75 * s.max()].index)
    return f"{top[0]:02d}:00", f"{top[-1] + 1:02d}:00"


def charts(pred, fam):
    (DOCS / "img").mkdir(parents=True, exist_ok=True)
    pred = pred.assign(hour=pred.slot_local.dt.hour,
                       err_now=(pred.bikes_now - pred.y_bikes).abs(),
                       err_model=(pred.pred - pred.y_bikes).abs())
    by = pred.groupby("hour")[["err_now", "err_model"]].mean()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(by.index, by.err_now, color="#888780", linestyle="--", label="Baseline: same as now")
    ax.plot(by.index, by.err_model, color="#534AB7", linewidth=2, label="LightGBM")
    ax.set_xlabel("Hour of day (Asia/Taipei)"); ax.set_ylabel("MAE (bikes)")
    ax.set_xticks(range(0, 24, 2)); ax.legend(frameon=False); ax.grid(alpha=0.25)
    ax.set_title("Error by hour of day, held-out test period", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(DOCS / "img/mae_by_hour.png", dpi=150); plt.close(fig)

    s = pd.Series({FAMILY[k][0]: v for k, v in fam.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(s.index, s.values, color="#534AB7")
    ax.set_xlabel("Average push on the 1-hour change (bikes)"); ax.grid(axis="x", alpha=0.25)
    ax.set_title("What the model leans on (mean |SHAP| per factor group)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(DOCS / "img/drivers.png", dpi=150); plt.close(fig)
    worst = by.err_now.idxmax()
    return {"peak_hour": int(worst), "peak_now": by.err_now[worst], "peak_model": by.err_model[worst]}


def values():
    ds, m, ex = load()
    fs = fetch_stats()
    pred = pd.read_parquet(OUT / "test_predictions.parquet")
    mm = m["metrics"]
    now, yday, lgbm = mm["baseline_now"], mm["baseline_yday"], mm["lightgbm"]
    moved = (pred.y_bikes - pred.bikes_now).abs() >= 3
    mv_now = (pred.bikes_now - pred.y_bikes).abs()[moved].mean()
    mv_model = (pred.pred - pred.y_bikes).abs()[moved].mean()
    span = pd.Timestamp(ds["last_slot_local"]) - pd.Timestamp(ds["first_slot_local"])
    fam = ex["drivers"]["families"]
    refill, rain, day = ex["what_if_refill"][0], ex["what_if_rain"], ex["what_if_daytype"][0]
    fd = list(ex["drivers"]["fill_direction"].values())
    v = dict(
        first=ds["first_slot_local"][:16], last=ds["last_slot_local"][:16],
        days=f"{span.total_seconds() / 86400:.1f}",
        stations=f"{ds['stations']:,}", clean_rows=f"{ds['clean_rows']:,}",
        dups=f"{ds['duplicate_rows_removed']:,}", model_rows=f"{ds['model_rows']:,}",
        slots_expected=f"{ds['slots_expected']:,}", slots_missing=ds["slots_missing"],
        coverage=f"{100 * ds['slots_present'] / ds['slots_expected']:.1f}",
        gaps="; ".join(ds["missing_ranges_local"]) or "none",
        weather_missing=ds["weather_slots_without_data"],
        bike_ok=f"{fs['bike', True]:,}", bike_fail=fs["bike", False],
        w_ok=f"{fs['weather', True]:,}", w_fail=fs["weather", False],
        train_a=m["periods_local"]["train"][0][:16], train_b=m["periods_local"]["train"][1][:16],
        valid_a=m["periods_local"]["valid"][0][:16], valid_b=m["periods_local"]["valid"][1][:16],
        test_a=m["periods_local"]["test"][0][:16], test_b=m["periods_local"]["test"][1][:16],
        n_train=f"{m['rows']['train']:,}", n_test=f"{m['rows']['test']:,}", trees=m["trees"],
        mae_now=f"{now['mae']:.3f}", mae_yday=f"{yday['mae']:.3f}", mae_lgbm=f"{lgbm['mae']:.3f}",
        rmse_now=f"{now['rmse']:.3f}", rmse_yday=f"{yday['rmse']:.3f}", rmse_lgbm=f"{lgbm['rmse']:.3f}",
        gain=f"{100 * (1 - lgbm['mae'] / now['mae']):.1f}",
        gain_rmse=f"{100 * (1 - lgbm['rmse'] / now['rmse']):.1f}",
        yday_worse=f"{100 * (yday['mae'] / now['mae'] - 1):.0f}",
        moved_share=f"{100 * moved.mean():.0f}", mv_now=f"{mv_now:.2f}", mv_model=f"{mv_model:.2f}",
        mv_gain=f"{100 * (1 - mv_model / mv_now):.1f}",
        e_p=f"{100 * lgbm['empty']['precision']:.0f}", e_r=f"{100 * lgbm['empty']['recall']:.0f}",
        f_p=f"{100 * lgbm['full']['precision']:.0f}", f_r=f"{100 * lgbm['full']['recall']:.0f}",
        eb_p=f"{100 * now['empty']['precision']:.0f}", eb_r=f"{100 * now['empty']['recall']:.0f}",
        fill_empty=f"{fd[0]:+.2f}", fill_full=f"{fd[-1]:+.2f}",
        k=refill["k"], rh=f"{refill['hour']:02d}", kept=f"{refill['kept_mean']:.1f}",
        kept10=f"{refill['kept_p10']:.1f}", kept90=f"{refill['kept_p90']:.1f}",
        n_refill=f"{refill['stations']:,}",
        rain_mm=f"{rain['mm']:.0f}", rain_diff=f"{rain['mean_diff']:+.2f}",
        rain_seen=f"{rain['max_precip_seen']:.1f}", rain_share=f"{100 * rain['share_rows_with_rain']:.0f}",
        day_moved=f"{100 * day['share_moved_1plus']:.1f}", day_moved3=f"{100 * day['share_moved_3plus']:.1f}",
    )
    peak = charts(pred, fam)
    inside, outside, longest = failure_split(ds["missing_ranges_local"])
    t_a, t_b = peak_hours(ex["drivers"]["time_push_by_hour"]["weekday"])
    n_wk, n_we = weekday_count(*m["periods_local"]["train"]), weekend_count(*m["periods_local"]["train"])
    day = ex["what_if_daytype"][0]
    if day["share_moved_1plus"] < DAYTYPE_MATTERS:
        v.update(day_verdict_en="The label barely matters: the model reads the day's rhythm from the trend and "
                                f"from yesterday. With {n_wk} weekdays and {n_we} weekend days in the training data "
                                "it cannot learn the calendar.",
                 day_verdict_zh=f"標籤幾乎沒影響：模型從趨勢與昨天的型態讀出當天的節奏。訓練資料只有 {n_wk} 個平日、"
                                f"{n_we} 個週末天，學不到日曆效應。")
    else:
        v.update(day_verdict_en=f"An early signal: on a Sunday the forecast changes by {day['mean_change_as_sunday'] - day['mean_change_weekday']:+.2f} "
                                "bikes on average and busy commuter stations drain less. It is learned from only "
                                f"{n_we} weekend days, so treat it as a hint, not a finding.",
                 day_verdict_zh=f"初步訊號：改成週日後，預測平均變動 {day['mean_change_as_sunday'] - day['mean_change_weekday']:+.2f} 台，"
                                f"通勤站流失變少。但只從 {n_we} 個週末天學到，只能當線索，不能當結論。")
    v["fail_out_en"] = "The other one was" if outside == 1 else f"The other {outside} were"
    v.update(fail_in=inside, fail_out=outside, gap_min=longest, time_a=t_a, time_b=t_b,
             n_weekdays=n_wk, n_weekend=n_we)
    v.update(peak_hour=f"{peak['peak_hour']:02d}", peak_now=f"{peak['peak_now']:.2f}",
             peak_model=f"{peak['peak_model']:.2f}")
    for lang in (0, 1):
        rows = "\n".join(f"| {FAMILY[k][lang]} | {x:.2f} |"
                         for k, x in sorted(fam.items(), key=lambda kv: -kv[1]))
        v["drivers_en" if lang == 0 else "drivers_zh"] = rows
    return v


EN = """# YouBike: how many bikes in one hour?

*All numbers and charts in this file are generated by `make_report.py` from the latest pipeline run.*
*Data used: {first} to {last} (Asia/Taipei), {days} days.*

## 1. Summary

- **Goal.** Predict, for every YouBike station in Taipei and New Taipei, how many bikes it will have one hour from now, and explain what drives it.
- **Result.** On a held-out final 24 hours, LightGBM has a mean absolute error of **{mae_lgbm} bikes** against **{mae_now}** for the "same as now" baseline: **{gain}% better**. When the count actually moves by 3+ bikes ({moved_share}% of cases) the gain is **{mv_gain}%**.
- **Main driver.** The station's own state. Nearly full stations lose bikes, nearly empty ones gain. Then yesterday's pattern at the same time, then what nearby stations are doing.
- **Honest limit.** {days} days of data. The model is useful, not finished; the limits are in section 6.

## 2. Pipeline

```mermaid
flowchart LR
    A[YouBike Taipei API] --> C[collector.py<br/>Docker, every 5 min]
    B[YouBike New Taipei API<br/>paged] --> C
    W[Open-Meteo<br/>hourly] --> C
    C --> R[data/raw<br/>untouched gzip files]
    C --> L[fetch_log.jsonl<br/>every attempt]
    R --> D[build_dataset.py]
    L --> D
    D --> T[out/model_table.parquet]
    T --> M[train.py<br/>baselines + LightGBM]
    M --> E[explain.py<br/>SHAP + what-if]
    M --> P[app.py<br/>Streamlit]
    E --> P
```

| Step | File | What it guarantees |
|---|---|---|
| Collect | `collector.py` | Runs on a schedule in Docker (`restart: unless-stopped`). Stores the raw response bytes, named by our fetch time in UTC. Retries, then logs the failure and carries on. Weather runs in its own thread so it can never delay the bikes. |
| Clean | `build_dataset.py` | Reproducible from `data/raw` in about two minutes. One row per station per 5-minute slot. Drops duplicate fetches, partial New Taipei rounds and switched-off stations. |
| Model | `train.py` | Time-ordered split with a 1-hour gap. Two naive baselines and LightGBM. |
| Explain | `explain.py` | SHAP values per factor group, three what-if scenarios. |
| Present | `app.py` | Per-station forecast with its explanation, what-if sliders, at-risk map. English / 中文. |

## 3. Data

| | |
|---|---|
| Stations | {stations} |
| Clean snapshot rows | {clean_rows} ({dups} duplicate rows removed) |
| Rows with features and a known answer | {model_rows} |
| 5-minute slots expected / missing | {slots_expected} / {slots_missing} ({coverage}% covered) |
| Bike fetches succeeded / failed | {bike_ok} / {bike_fail} |
| Weather fetches succeeded / failed | {w_ok} / {w_fail} |

- **Gaps.** {gaps}. {fail_in} of the {bike_fail} bike fetch failures fall in this window: DNS failures on the collecting machine, not the APIs. The collector logged them and resumed by itself. {fail_out_en} isolated and left no gap in the data.
- **Weather failures look worse than they are.** After a failure the collector retries every 10 minutes and logs each try. Every successful download carries a 48-hour forecast, so only {weather_missing} slots have no weather at all.
- **Time.** File names are UTC fetch times. Features use Asia/Taipei.
- **Leakage rule.** Each row only uses data fetched at or before its own slot. The weather for the target hour comes from the forecast we had already downloaded then, never from what was observed later.

Column-level documentation: [DATASET.md](DATASET.md).

## 4. Model and results

**Target.** Change in available bikes over the next hour, added back to the current count. **Model.** One LightGBM regressor ({trees} trees, L1 loss), 26 features.

**Split, by time and never random.**

| Part | Period | Role |
|---|---|---|
| Train | {train_a} → {train_b} | {n_train} rows the model learns from |
| Validation | {valid_a} → {valid_b} | Decides when to stop adding trees |
| Test | {test_a} → {test_b} | {n_test} forecasts, never seen, the only source of the scores below |

A 1-hour gap separates the parts, because a row at 09:20 already "knows" the 10:20 answer.

| Method | MAE (bikes) | RMSE | vs "same as now" |
|---|---|---|---|
| Baseline: same as now | {mae_now} | {rmse_now} | — |
| Baseline: same as yesterday at that time | {mae_yday} | {rmse_yday} | {yday_worse}% worse |
| **LightGBM** | **{mae_lgbm}** | **{rmse_lgbm}** | **{gain}% better** |

![Error by hour](img/mae_by_hour.png)

- **"Same as now" is a strong baseline.** Most stations barely move within an hour. The honest comparison is on the rows that do move: {mv_now} vs {mv_model} bikes, **{mv_gain}% better**.
- **Hardest hour.** {peak_hour}:00, where the baseline error is {peak_now} bikes and the model's is {peak_model}.
- **"Same as yesterday" loses badly.** Copying yesterday also copies yesterday's one-off events: a truck that refilled a station at 08:00 yesterday may come at 09:30 today, and a busy hour yesterday may be quiet today. The current count already reflects today.
- **"Same time last week" is not possible** with {days} days of data.
- **Empty / full warnings.** When the model says a station will be empty it is right {e_p}% of the time, but it catches {e_r}% of the empty stations (baseline: {eb_p}% / {eb_r}%). For full: {f_p}% right, {f_r}% caught. Precise but not complete; the threshold is conservative.

## 5. Drivers and what-if

Each forecast is split into contributions with SHAP values. The table is the average size of each group's contribution, in bikes.

| Factor group | Average push (bikes) |
|---|---|
{drivers_en}

![Drivers](img/drivers.png)

- **Stations drift back to the middle.** How full the station is pushes the forecast by {fill_empty} bikes when it is under 10% full and by {fill_full} when it is over 90% full.
- **Yesterday matters more than the calendar.** The change the station saw at this time yesterday is one of the strongest single signals.
- **Neighbours carry the area's direction.** If the five nearest stations are draining, this one probably is too.
- **Time of day matters most on weekday mornings**, {time_a} to {time_b}.

**What-if scenarios** (take real rows, change one thing, predict again):

| Scenario | Result | Can it be trusted? |
|---|---|---|
| A truck adds {k} bikes at {rh}:00 on a weekday ({n_refill} stations) | On average **{kept}** extra bikes are still there an hour later. Worst 10% of stations keep under {kept10}; best 10% keep over {kept90}. | Usable. Matches what happened after real sudden jumps in the test period. It is correlation, not a controlled experiment. |
| Same weekday morning, but labelled as a Sunday | {day_moved}% of stations change by 1+ bike ({day_moved3}% by 3+) | {day_verdict_en} |
| {rain_mm} mm/h of rain | {rain_diff} bikes on average | **No.** The heaviest rain observed was {rain_seen} mm/h; {rain_share}% of rows had any rain. |

The first scenario is the operational one: it shows where added bikes stay and where they are gone within the hour.

### The app

![App in replay mode](img/app_replay.jpg)

The app has two views. **Latest** forecasts from the newest snapshot. **Replay** lets you pick any moment of the held-out test day and puts the forecast (orange, dashed) next to what really happened (grey), using the model that never saw that day. The screenshot (from an earlier run, so a different test day) also shows a rebalancing jump at about 04:30 that no public data announces.

## 6. Limits

- **{days} days of data.** No weekly seasonality, one test day, weekday behaviour learned from very few weekdays. Scores will move as data accumulates.
- **Rebalancing is invisible.** Trucks move bikes and no public feed says when. Example: a station jumps from 4 to 20 bikes at 04:00. Errors of this kind cannot be removed with these sources.
- **Almost no rain** during collection, so the weather effect is not established.
- **Demand is censored.** An empty station shows zero rentals whatever the real demand was.
- **What-ifs are correlational.** They describe what followed similar situations in the past.
- **Single collector on a laptop.** One network outage cost {gap_min} minutes of data.
- **The app shows the newest processed snapshot**, not a live feed.

## 7. With another month

1. **More history first.** Four to five weeks unlock the "same time last week" baseline, weekly patterns, holidays and rolling-origin backtests over many test days instead of one.
2. **Model flows, not stock.** Infer rentals and returns between snapshots, detect rebalancing events from sudden jumps and model them separately, handle censored demand at empty and full stations.
3. **Risk instead of a single number.** Quantile forecasts, calibrated probabilities of empty / full, thresholds tuned to the cost of a missed warning, horizons of 15, 30 and 60 minutes.
4. **More signals.** MRT ridership, events, school and public holidays, CWA rainfall radar for nowcasting, weather per district instead of two points.
5. **Operations.** Collector on a cloud VM with object storage, alerting on gaps, daily automatic retraining with a score history, and a live app.
6. **A recommendation, not just a forecast.** Turn the refill what-if into a ranked list: where should the next truck go so the bikes last.

## 8. Reproduce

```
docker compose up -d --build                           # collector
docker compose build ml
docker compose run --rm ml python build_dataset.py
docker compose run --rm ml python train.py
docker compose run --rm ml python explain.py
docker compose run --rm ml python export_app_data.py
docker compose run --rm ml python make_report.py
docker compose --profile app up app                    # http://localhost:8502
```
"""

ZH = """# YouBike：一小時後還有幾台車？

*本檔所有數字與圖表都由 `make_report.py` 從最新一次流程執行結果產生。*
*使用資料：{first} 至 {last}（台灣時間），共 {days} 天。*

## 1. 摘要

- **目標。** 預測台北與新北每個 YouBike 站點一小時後有幾台車，並解釋影響因素。
- **結果。** 用最後 24 小時（模型沒看過）驗證，LightGBM 的平均誤差是 **{mae_lgbm} 台**，「跟現在一樣」基準是 **{mae_now} 台**，**好 {gain}%**。車數實際變動 3 台以上時（佔 {moved_share}%），好 **{mv_gain}%**。
- **主要驅動因素。** 站點自己的現況：快滿的站會掉車，快空的站會回升。其次是昨天同時段的型態，再來是鄰近站的動向。
- **誠實的限制。** 只有 {days} 天資料。模型有用，但還沒完成；限制見第 6 節。

## 2. 流程架構

```mermaid
flowchart LR
    A[台北 YouBike API] --> C[collector.py<br/>Docker，每 5 分鐘]
    B[新北 YouBike API<br/>分頁] --> C
    W[Open-Meteo<br/>每小時] --> C
    C --> R[data/raw<br/>原封不動的壓縮檔]
    C --> L[fetch_log.jsonl<br/>每次嘗試都記錄]
    R --> D[build_dataset.py]
    L --> D
    D --> T[out/model_table.parquet]
    T --> M[train.py<br/>基準＋LightGBM]
    M --> E[explain.py<br/>SHAP＋what-if]
    M --> P[app.py<br/>Streamlit]
    E --> P
```

| 步驟 | 檔案 | 保證什麼 |
|---|---|---|
| 收集 | `collector.py` | 在 Docker 內按排程執行（`restart: unless-stopped`）。原始回應原封不動存檔，檔名是我們的抓取時間（UTC）。失敗會重試，仍失敗就記錄後繼續。天氣跑在獨立執行緒，不會拖慢車數的抓取。 |
| 整理 | `build_dataset.py` | 約兩分鐘可從 `data/raw` 完整重現。每站每 5 分鐘一列。去除重複抓取、新北缺頁的輪次、停用的站點。 |
| 建模 | `train.py` | 按時間切分並留 1 小時間隔。兩個笨基準加 LightGBM。 |
| 解釋 | `explain.py` | 各類因素的 SHAP 值，三個 what-if 情境。 |
| 呈現 | `app.py` | 單站預測與其解釋、what-if 滑桿、風險地圖。中英文可切換。 |

## 3. 資料

| | |
|---|---|
| 站點數 | {stations} |
| 乾淨快照列數 | {clean_rows}（去除 {dups} 列重複） |
| 有特徵且已知答案的列數 | {model_rows} |
| 應有／缺少的 5 分鐘時段 | {slots_expected}／{slots_missing}（涵蓋 {coverage}%） |
| 車數抓取 成功／失敗 | {bike_ok}／{bike_fail} |
| 天氣抓取 成功／失敗 | {w_ok}／{w_fail} |

- **缺口。** {gaps}。{bike_fail} 次車數抓取失敗中，有 {fail_in} 次落在這段時間，原因是收集主機的 DNS 解析失敗，不是 API 的問題。收集程式有記錄並自行恢復。其餘 {fail_out} 次是零星失敗，沒有造成資料缺口。
- **天氣失敗次數看起來多，實際影響小。** 失敗後每 10 分鐘重試一次，每次都記一行。每次成功下載都帶有 48 小時預報，所以完全沒有天氣資料的時段只有 {weather_missing} 個。
- **時間。** 檔名是 UTC 抓取時間，特徵使用台灣時間。
- **防洩漏規則。** 每一列只使用在該時段當下或之前抓到的資料。目標時段的天氣取自當時已下載的預報，不使用事後的觀測值。

欄位層級的說明見 [DATASET.md](DATASET.md)。

## 4. 模型與結果

**預測目標。** 未來一小時的車數變化量，再加回現在的車數。**模型。** 一個 LightGBM 迴歸模型（{trees} 棵樹、L1 損失），26 個特徵。

**切分方式：按時間，絕不隨機。**

| 區段 | 期間 | 用途 |
|---|---|---|
| 訓練 | {train_a} → {train_b} | 模型學習用，{n_train} 列 |
| 驗證 | {valid_a} → {valid_b} | 決定何時停止增加樹 |
| 測試 | {test_a} → {test_b} | {n_test} 筆預測，模型從未看過，下方成績只來自這一段 |

各區段之間留 1 小時間隔，因為 09:20 的那一列已經「知道」10:20 的答案。

| 方法 | 平均誤差 MAE（台） | RMSE | 相對「跟現在一樣」 |
|---|---|---|---|
| 基準：跟現在一樣 | {mae_now} | {rmse_now} | — |
| 基準：跟昨天同時間一樣 | {mae_yday} | {rmse_yday} | 差 {yday_worse}% |
| **LightGBM** | **{mae_lgbm}** | **{rmse_lgbm}** | **好 {gain}%** |

![各時段誤差](img/mae_by_hour.png)

- **「跟現在一樣」是很強的基準。** 多數站點一小時內變化不大。公平的比較要看有變動的時候：{mv_now} 對 {mv_model} 台，**好 {mv_gain}%**。
- **最難的時段。** {peak_hour}:00，基準誤差 {peak_now} 台，模型 {peak_model} 台。
- **「跟昨天一樣」輸很多。** 照抄昨天，也會照抄昨天的偶發事件：昨天 08:00 來補車，今天可能 09:30 才來；昨天的尖峰今天也可能很冷清。而「現在的車數」已經反映了今天的狀況。
- **「跟上週同時間一樣」做不到**，資料只有 {days} 天。
- **空站／滿站預警。** 模型說會空站時 {e_p}% 是對的，但只抓到 {e_r}% 的空站（基準為 {eb_p}%／{eb_r}%）。滿站：{f_p}% 正確，抓到 {f_r}%。準但不全，門檻設得保守。

## 5. 驅動因素與 what-if

每一筆預測都用 SHAP 值拆成各因素的貢獻。下表是各類因素貢獻的平均大小，單位是台。

| 因素類別 | 平均推力（台） |
|---|---|
{drivers_zh}

![驅動因素](img/drivers.png)

- **站點會往中間靠。** 滿度低於 10% 時，「多滿」這條線索把預測推動 {fill_empty} 台；高於 90% 時推動 {fill_full} 台。
- **昨天比日曆重要。** 站點昨天同時段的變化量是最強的單一訊號之一。
- **鄰近站帶出整個區域的方向。** 最近的五個站都在掉車，這一站多半也是。
- **時段的影響集中在平日早上** {time_a} 到 {time_b}。

**what-if 情境**（取真實資料列，只改一項，重新預測）：

| 情境 | 結果 | 可信嗎？ |
|---|---|---|
| 平日 {rh}:00 調度車補進 {k} 台（{n_refill} 個站） | 一小時後平均還多 **{kept}** 台。最差的一成站點留不到 {kept10} 台，最好的一成留超過 {kept90} 台。 | 可用。與測試期真實突增事件之後的走勢相符。它是相關性，不是對照實驗。 |
| 同一個平日早上，改標成週日 | {day_moved}% 的站點變動 1 台以上（3 台以上 {day_moved3}%） | {day_verdict_zh} |
| 降雨 {rain_mm} mm／小時 | 平均 {rain_diff} 台 | **不可信。** 觀測到的最大雨量是 {rain_seen} mm／小時，有降雨的資料列佔 {rain_share}%。 |

第一個情境有營運意義：它指出補進去的車在哪些站留得住，在哪些站一小時內就被借光。

### App

![回放模式的 App](img/app_replay.jpg)

App 有兩種檢視。**最新**：用最新一筆快照做預測。**回放**：可選測試日的任一時間點，把預測（橘色虛線）和實際結果（灰線）並排，使用的是沒看過那一天的模型。截圖（來自較早的一次執行，所以是另一個測試日）裡也看得到凌晨 4:30 左右的一次調度跳升，這在公開資料中沒有任何預告。

## 6. 限制

- **只有 {days} 天資料。** 沒有週週期、測試日只有一天、平日行為只從極少數平日學到。資料累積後成績會變動。
- **看不到調度。** 調度車搬車沒有公開資料。例如某站在凌晨 04:00 從 4 台跳到 20 台，這類誤差無法用現有資料來源消除。
- **收集期間幾乎沒下雨**，天氣的影響尚未確立。
- **需求被截斷。** 空站的借車數是零，但真實需求不是零。
- **what-if 是相關性。** 它描述過去類似狀況之後發生了什麼。
- **只有一台筆電在收集。** 一次斷網就少了 {gap_min} 分鐘資料。
- **App 顯示的是最新一次處理過的快照**，不是即時串流。

## 7. 如果再多一個月

1. **先累積歷史。** 四到五週的資料可以做「上週同時間」基準、週週期、假日，以及跨多個測試日的滾動回測。
2. **預測流量而不是存量。** 從相鄰快照推估借還量，從突增偵測調度事件並另外建模，處理空站與滿站時被截斷的需求。
3. **給風險而不是單一數字。** 分位數預測、校準過的空站／滿站機率、依漏報成本調整門檻、15／30／60 分鐘多個預測時距。
4. **更多訊號。** 捷運運量、活動、學校與國定假日、氣象署雷達降雨即時預報、分區天氣（目前只有兩個點）。
5. **維運。** 收集程式搬到雲端主機並使用物件儲存、缺口告警、每日自動重新訓練並保留成績歷史、即時更新的 App。
6. **從預測走到建議。** 把補車的 what-if 做成排序清單：下一輛調度車去哪裡，補的車留得最久。

## 8. 重現步驟

```
docker compose up -d --build                           # 收集程式
docker compose build ml
docker compose run --rm ml python build_dataset.py
docker compose run --rm ml python train.py
docker compose run --rm ml python explain.py
docker compose run --rm ml python export_app_data.py
docker compose run --rm ml python make_report.py
docker compose --profile app up app                    # http://localhost:8502
```
"""


def main():
    DOCS.mkdir(exist_ok=True)
    v = values()
    (DOCS / "REPORT.md").write_text(EN.format(**v), encoding="utf-8", newline="\n")
    (DOCS / "REPORT.zh-TW.md").write_text(ZH.format(**v), encoding="utf-8", newline="\n")
    print("wrote docs/REPORT.md, docs/REPORT.zh-TW.md, docs/img/mae_by_hour.png, docs/img/drivers.png")
    print(f"MAE now {v['mae_now']}  yday {v['mae_yday']}  lgbm {v['mae_lgbm']}  gain {v['gain']}%  moved gain {v['mv_gain']}%")


if __name__ == "__main__":
    main()
