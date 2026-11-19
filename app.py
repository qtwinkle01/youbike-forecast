"""YouBike 1-hour forecast app (English / 中文).

Run:  docker compose --profile app up app      then open http://localhost:8502
Reads app_data/ (made by export_app_data.py). Two modes:
  Latest  forecast from the newest snapshot (model trained on everything)
  Replay  pick a moment in the held-out test day and compare the forecast with what
          really happened (model that never saw that day)
"""
import json
import time
from pathlib import Path

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from explain import FAMILIES
from train import FEATURES

APP = Path("app_data")

# Every piece of UI text lives here: key -> (English, 中文). {placeholders} are filled by t().
TEXT = {
    "title": ("YouBike: how many bikes in one hour?", "YouBike：一小時後還有幾台車？"),
    "subtitle": ("Taipei + New Taipei · {n:,} active stations · data as of **{asof}** (Asia/Taipei) · forecast for **{target}**",
                 "台北＋新北 · {n:,} 個營運中站點 · 資料時間 **{asof}**（台灣時間）· 預測時間 **{target}**"),
    "missing": ("Missing files in app_data/: {files}. Run build_dataset.py, train.py, explain.py and export_app_data.py first (see README).",
                "app_data/ 缺少檔案：{files}。請先執行 build_dataset.py、train.py、explain.py、export_app_data.py（見 README）。"),
    "mode": ("View", "檢視模式"),
    "mode_live": ("Latest", "最新"),
    "mode_replay": ("Replay with real outcome", "回放（對照實際結果）"),
    "replay_time": ("Moment to replay", "回放的時間點"),
    "replay_note": ("Any 5-minute moment of the held-out test day.", "可選測試日的任一個 5 分鐘時間點。"),
    "replay_banner": ("**Replay.** The forecast below was made with a model that never saw this day. The real outcome one hour later is shown next to it.",
                      "**回放模式。** 下方的預測來自沒看過這一天的模型，旁邊列出一小時後實際發生的結果。"),
    "update": ("Update to the newest data", "更新到最新資料"),
    "updating": ("Reading the last 26 hours of raw files (about a minute and a half)…", "正在讀取最近 26 小時的原始檔（約一分半鐘）…"),
    "stale": ("This snapshot is **{h:.1f} hours old**. Live forecasts need the collector running; then press \"Update to the newest data\". Or use Replay to see forecasts against real outcomes.",
              "這份快照已經是 **{h:.1f} 小時前**的資料。即時預測需要收集程式持續執行，再按「更新到最新資料」。也可以切到回放模式，對照預測與實際結果。"),
    "ended": ("**Data collection has ended.** This is the last snapshot ({asof}), so the forecast is for one hour after it. Use **Replay** to see forecasts next to what really happened.",
              "**資料收集已結束。** 這是最後一筆快照（{asof}），預測的是它的一小時後。切到**回放**可以看預測與實際結果的對照。"),
    "station_hdr": ("Station", "站點"),
    "city": ("City", "城市"),
    "taipei": ("Taipei", "台北"),
    "newtaipei": ("New Taipei", "新北"),
    "district": ("District", "行政區"),
    "station": ("Station", "站點"),
    "tab1": ("Station forecast", "單站預測"),
    "tab2": ("Stations at risk", "可能空站／滿站"),
    "tab3": ("Model, drivers and limits", "模型、驅動因素與限制"),
    "bikes_now": ("Bikes now", "現在可借"),
    "bikes_then": ("Bikes at {asof}", "{asof} 可借"),
    "free_now": ("Free docks", "空位"),
    "forecast_at": ("Forecast for {target}", "{target} 預測"),
    "actual_at": ("Actually at {target}", "{target} 實際"),
    "delta_bikes": ("{d:+.1f} bikes", "{d:+.1f} 台"),
    "err": ("forecast off by {e:.1f}", "預測差 {e:.1f} 台"),
    "status": ("Status in 1 hour", "一小時後狀態"),
    "st_empty": ("Likely EMPTY", "可能空站"),
    "st_full": ("Likely FULL", "可能滿站"),
    "st_ok": ("OK", "正常"),
    "y_bikes": ("Bikes available", "可借車數"),
    "chart_note": ("Blue: last 24 hours (one point every 5 minutes). Orange dashed: the forecast for the next hour.",
                   "藍線：過去 24 小時（每 5 分鐘一點）。橘色虛線：未來一小時的預測。"),
    "chart_note_replay": ("Blue: the 6 hours before. Orange dashed: the forecast. Grey: what really happened in the following hour.",
                          "藍線：之前的 6 小時。橘色虛線：預測。灰線：接下來一小時實際發生的變化。"),
    "replay_score": ("This case: the model was off by **{m:.1f}** bikes; \"same as now\" would have been off by **{b:.0f}**.",
                     "這一筆：模型差 **{m:.1f}** 台；「跟現在一樣」會差 **{b:.0f}** 台。"),
    "why": ("Why this forecast?", "為什麼是這個預測？"),
    "push_axis": ("Push on the 1-hour change (bikes)", "對一小時變化量的推力（台）"),
    "why_note": ("Each bar is how many bikes that group of signals adds (green) or removes (orange) from the forecast. They add up to the predicted change (SHAP values).",
                 "每一條代表這類線索讓預測增加（綠）或減少（橘）幾台，全部加起來就是預測的變化量（SHAP 值）。"),
    "whatif": ("What if…", "如果…"),
    "truck": ("A truck adds (+) or removes (−) bikes right now", "調度車現在補進（+）或移走（−）幾台"),
    "rain_on": ("Set the rain myself", "自己設定降雨"),
    "rain": ("Rain (mm per hour)", "降雨量（mm／小時）"),
    "fc_asis": ("Forecast, as is", "原本的預測"),
    "fc_whatif": ("Forecast, what-if", "假設情境的預測"),
    "kept": ("Of the {n} bikes added, about **{k:.1f}** would still be there in an hour.",
             "補進的 {n} 台，一小時後大約還留著 **{k:.1f}** 台。"),
    "removed": ("Removing {n} bikes leaves about **{k:.1f}** fewer in an hour.",
                "移走 {n} 台，一小時後大約少 **{k:.1f}** 台。"),
    "rain_warn": ("Low confidence: the heaviest rain during data collection was {mm:.1f} mm/h, so the model has almost no evidence of how rain changes demand.",
                  "可信度低：資料收集期間最大雨量只有 {mm:.1f} mm／小時，模型幾乎沒有「下雨如何影響需求」的證據。"),
    "risk_title": ("Stations likely to be empty or full at {target}", "{target} 可能空站或滿站的站點"),
    "likely_empty": ("Likely empty", "可能空站"),
    "likely_full": ("Likely full", "可能滿站"),
    "checked": ("Stations checked", "檢查站數"),
    "tip_empty": ("likely empty", "可能空站"),
    "tip_full": ("likely full", "可能滿站"),
    "no_risk": ("No station is forecast to be empty or full in the next hour.", "未來一小時沒有站點被預測為空站或滿站。"),
    "map_note": ("Orange: likely empty (forecast under 1 bike). Blue: likely full (forecast under 1 free dock).",
                 "橘色：可能空站（預測不到 1 台車）。藍色：可能滿站（預測不到 1 個空位）。"),
    "risk_hits": ("What really happened: **{eh} of {en}** stations flagged empty had 0 or 1 bike an hour later; **{fh} of {fn}** flagged full had 0 or 1 free dock.",
                  "實際結果：被標為可能空站的 {en} 站中，有 **{eh}** 站一小時後只剩 0 或 1 台；被標為可能滿站的 {fn} 站中，有 **{fh}** 站只剩 0 或 1 個空位。"),
    "col_docks": ("Docks", "車格數"),
    "col_forecast": ("Forecast", "預測"),
    "col_actual": ("Actual", "實際"),
    "trust": ("How far to trust these flags (whole held-out test): when the model says empty it is right {ep:.0%} of the time but it catches only {er:.0%} of empty stations; for full, {fp:.0%} and {fr:.0%}. A short list of confident warnings, not a complete one.",
              "這份清單可信到什麼程度（整個測試期）：模型說會空站時有 {ep:.0%} 是對的，但只抓到 {er:.0%} 的空站；滿站則是 {fp:.0%} 與 {fr:.0%}。它是一份「少而準」的警示，不是完整名單。"),
    "beat": ("Does it beat the naive baselines?", "有沒有贏過笨基準？"),
    "split": ("Trained on {a} → {b}, tested on the following unseen period {c} → {d} ({n:,} forecasts). No random split: the future never leaks into training.",
              "訓練期 {a} → {b}，測試期是之後沒看過的 {c} → {d}（共 {n:,} 筆預測）。不做隨機切分，未來的資料不會洩漏進訓練。"),
    "method": ("Method", "方法"),
    "mae": ("MAE (bikes)", "平均誤差 MAE（台）"),
    "vs_now": ("vs 'same as now'", "相對「跟現在一樣」"),
    "b_now": ("Baseline: same as now", "基準：跟現在一樣"),
    "b_yday": ("Baseline: same as yesterday at that time", "基準：跟昨天同時間一樣"),
    "drives": ("What drives the forecast?", "什麼在推動預測？"),
    "avg_push": ("Average push on the 1-hour change (bikes)", "對一小時變化量的平均推力（台）"),
    "findings": (
        "- **Stations drift back to the middle.** Nearly full stations lose about 1 bike more than average in the next hour; nearly empty ones gain.\n"
        "- **What-if, rebalancing:** adding {k} bikes at {h:02d}:00 on a weekday leaves on average **{kept:.1f}** extra bikes an hour later; in the worst 10% of stations under {p10:.1f}.\n"
        "- **What-if, weekday vs Sunday:** changing only the day label moves forecasts by under 1 bike. The model reads the day's rhythm from the current trend and from yesterday, not from the calendar.",
        "- **站點會往中間靠。** 快滿的站接下來一小時平均多掉約 1 台，快空的站則會回升。\n"
        "- **假設情境：調度補車。** 平日 {h:02d}:00 補進 {k} 台，一小時後平均還多 **{kept:.1f}** 台；最差的一成站點留不到 {p10:.1f} 台。\n"
        "- **假設情境：平日換成週日。** 只改星期幾的標籤，預測變動不到 1 台。模型是從當下趨勢與昨天的型態判斷今天的節奏，不是看日曆。"),
    "limits": ("Limits", "限制"),
    "limits_body": (
        "- **Only a few days of data.** No 'same time last week' baseline is possible, and weekday patterns are learned from one or two weekdays.\n"
        "- **Rebalancing trucks are invisible.** Sudden jumps (for example +8 bikes at 4 am) cannot be predicted from public data.\n"
        "- **Almost no rain was observed**, so the rain what-if is not reliable.\n"
        "- **Correlation, not cause.** What-ifs show what the model expects from similar past situations.\n"
        "- **No events, MRT or holiday data** yet.",
        "- **只有幾天的資料。** 做不了「上週同時間」的基準，平日型態也只從一兩個平日學到。\n"
        "- **看不到調度車。** 突然的跳升（例如凌晨 4 點多 8 台）無法從公開資料預測。\n"
        "- **幾乎沒遇到下雨**，所以降雨的假設情境不可靠。\n"
        "- **是相關，不是因果。** 假設情境呈現的是「過去類似狀況後來怎樣」。\n"
        "- **還沒有活動、捷運、假日資料。**"),
}
FAMILY_LABEL = {
    "current state": ("Current state (bikes, free docks)", "站點現況（車數、空位）"),
    "recent trend": ("Recent trend (last 15 / 60 min)", "近期趨勢（過去 15／60 分鐘）"),
    "yesterday": ("Yesterday at this time", "昨天同時段"),
    "time": ("Time of day / weekday", "時段／星期幾"),
    "weather": ("Weather", "天氣"),
    "neighbours": ("Nearby stations", "鄰近站"),
    "location": ("Location", "位置"),
}
WEEKDAY_ZH = "一二三四五六日"
TPE = pd.Timedelta(hours=8)

st.set_page_config(page_title="YouBike 1-hour forecast", page_icon="🚲", layout="wide")
with st.sidebar:
    lang = st.radio("Language / 語言", ["English", "中文"], horizontal=True, key="lang")
L = 0 if lang == "English" else 1


def t(key, **kw):
    s = TEXT[key][L]
    return s.format(**kw) if kw else s


def fam(key):
    return FAMILY_LABEL[key][L]


def when(ts):
    if L == 0:
        return f"{ts:%a %d %b %H:%M}"
    return f"{ts.month}/{ts.day}（週{WEEKDAY_ZH[ts.dayofweek]}）{ts:%H:%M}"


needed = ["model.txt", "model_full.txt", "latest.parquet", "replay.parquet", "history.parquet",
          "stations.parquet", "metrics.json", "explain.json"]
missing = [f for f in needed if not (APP / f).exists()]
if missing:
    st.error(t("missing", files=", ".join(missing)))
    st.stop()


@st.cache_resource
def load_models():
    """(trained on everything, trained without the test day)."""
    return (lgb.Booster(model_file=str(APP / "model_full.txt")),
            lgb.Booster(model_file=str(APP / "model.txt")))


@st.cache_data
def load_static():
    stations = pd.read_parquet(APP / "stations.parquet")
    stations["label"] = stations["name"].str.replace("YouBike2.0_", "", regex=False)
    metrics = json.loads((APP / "metrics.json").read_text())
    explain = json.loads((APP / "explain.json").read_text(encoding="utf-8"))
    slots = pd.read_parquet(APP / "replay.parquet", columns=["slot_utc"]).slot_utc
    return stations, metrics, explain, sorted(slots.unique())


@st.cache_data
def load_latest():
    return pd.read_parquet(APP / "latest.parquet")


@st.cache_data
def load_replay(slot_utc):
    return pd.read_parquet(APP / "replay.parquet", filters=[("slot_utc", "==", slot_utc)])


@st.cache_data
def history(sno):
    parts = [pd.read_parquet(APP / "history.parquet", filters=[("sno", "==", sno)])]
    live = APP / "history_live.parquet"
    if live.exists():
        parts.append(pd.read_parquet(live, filters=[("sno", "==", sno)]))
    h = pd.concat(parts).drop_duplicates("slot_local", keep="last")
    return h[h.act == 1].sort_values("slot_local")


def predict(model, rows):
    """Predicted bikes one hour ahead, kept between 0 and the docks that exist."""
    raw = rows.bikes_now.to_numpy() + model.predict(rows[FEATURES])
    room = rows.bikes_now.to_numpy() + np.maximum(rows.docks_free_now.to_numpy(), 0)
    return np.clip(raw, 0, np.maximum(room, rows.capacity.to_numpy()))


def scenario(row, add_bikes, rain_mm):
    """Copy of one row with a what-if applied. Nothing else is changed."""
    s = row.copy()
    if add_bikes:
        s["bikes_now"] += add_bikes
        s["docks_free_now"] -= add_bikes
        s["fill_now"] = s.bikes_now / np.maximum(s.capacity, 1)
        s[["delta_15m", "delta_60m"]] += add_bikes
    if rain_mm is not None:
        s[["cur_precip", "fc_precip"]] = rain_mm
        s["fc_precip_prob"] = 100.0 if rain_mm > 0 else 0.0
    return s


def family_push(model, row):
    c = model.predict(row[FEATURES], pred_contrib=True)[0][:-1]
    idx = {f: i for i, f in enumerate(FEATURES)}
    return pd.DataFrame([{"factor": fam(k), "bikes": float(sum(c[idx[f]] for f in cols))}
                         for k, cols in FAMILIES.items()])


def collector_running():
    """The collector touches data/heartbeat every round. No recent heartbeat (for example on a
    hosted copy of the app) means there is nothing newer to load, so the button is hidden."""
    hb = Path("data/heartbeat")
    return hb.exists() and time.time() - hb.stat().st_mtime < 900


model_full, model_split = load_models()
stations, metrics, explain, replay_slots = load_static()

# ------------------------------------------------------------- sidebar: mode
with st.sidebar:
    mode = st.radio(t("mode"), ["live", "replay"], format_func=lambda m: t("mode_" + m), key="mode")
    if mode == "replay":
        eight = [s for s in replay_slots if (pd.Timestamp(s) + TPE).strftime("%H:%M") == "08:00"]
        pick = st.select_slider(t("replay_time"), options=replay_slots,
                                value=eight[-1] if eight else replay_slots[len(replay_slots) // 2],
                                format_func=lambda s: when(pd.Timestamp(s) + TPE), key="replay_slot")
        st.caption(t("replay_note"))
    elif collector_running() and st.button(t("update"), use_container_width=True):
        with st.spinner(t("updating")):
            import build_dataset
            build_dataset.build_latest(APP)
        st.cache_data.clear()
        st.rerun()

replay = mode == "replay"
model = model_split if replay else model_full
latest = (load_replay(pd.Timestamp(pick)) if replay else load_latest()).merge(
    stations[["sno", "label", "district"]], on="sno")
as_of = latest.slot_local.max()
target_time = as_of + pd.Timedelta(hours=1)
target = f"{target_time:%H:%M}"

st.title(t("title"))
st.caption(t("subtitle", n=len(latest), asof=when(as_of), target=target))
if replay:
    st.info(t("replay_banner"))
else:
    age = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None) - as_of
    if age > pd.Timedelta(minutes=15) and collector_running():
        st.warning(t("stale", h=age.total_seconds() / 3600))
    elif age > pd.Timedelta(minutes=15):
        st.info(t("ended", asof=when(as_of)))

# ---------------------------------------------------------- sidebar: station
with st.sidebar:
    st.header(t("station_hdr"))
    city = st.radio(t("city"), ["taipei", "newtaipei"], format_func=t, horizontal=True, key="city")
    pool = latest[latest.city == city]
    districts = sorted(pool.district.unique())
    district = st.selectbox(t("district"), districts, key=f"district_{city}",
                            index=districts.index("大安區") if "大安區" in districts else 0)
    names = sorted(pool[pool.district == district].label.unique())
    default = names.index("捷運科技大樓站") if "捷運科技大樓站" in names else 0
    name = st.selectbox(t("station"), names, index=default, key=f"station_{city}_{district}")

row = pool[(pool.district == district) & (pool.label == name)].head(1)
sno = row.sno.iloc[0]
now = int(row.bikes_now.iloc[0])
free = int(row.docks_free_now.iloc[0])
pred = float(predict(model, row)[0])

tab1, tab2, tab3 = st.tabs([t("tab1"), t("tab2"), t("tab3")])

# ------------------------------------------------------------ tab 1: station
with tab1:
    st.subheader(f"{name} · {district}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("bikes_then", asof=f"{as_of:%H:%M}") if replay else t("bikes_now"), now)
    c2.metric(t("free_now"), free)
    c3.metric(t("forecast_at", target=target), f"{pred:.1f}", t("delta_bikes", d=pred - now))
    if replay:
        actual = int(row.y_bikes.iloc[0])
        c4.metric(t("actual_at", target=target), actual, t("err", e=abs(pred - actual)), delta_color="off")
    else:
        free_pred = free - (pred - now)
        c4.metric(t("status"), t("st_empty") if pred < 1 else t("st_full") if free_pred < 1 else t("st_ok"))

    h = history(sno)
    past = h[(h.slot_local >= as_of - pd.Timedelta(hours=6 if replay else 24)) & (h.slot_local <= as_of)]
    x, y = alt.X("slot_local:T", title=None), alt.Y("bikes:Q", title=t("y_bikes"))
    layers = [alt.Chart(past).mark_line().encode(x=x, y=y)]
    if replay:
        after = h[(h.slot_local >= as_of) & (h.slot_local <= target_time)]
        layers.append(alt.Chart(after).mark_line(color="#888780", strokeWidth=3).encode(x=x, y=y))
    point = pd.DataFrame({"slot_local": [as_of, target_time], "bikes": [now, pred]})
    layers.append(alt.Chart(point).mark_line(strokeDash=[5, 4], color="#D85A30", point=True).encode(x=x, y=y))
    st.altair_chart(alt.layer(*layers), use_container_width=True)
    st.caption(t("chart_note_replay") if replay else t("chart_note"))
    if replay:
        st.markdown(t("replay_score", m=abs(pred - actual), b=abs(now - actual)))

    left, right = st.columns(2)
    with left:
        st.markdown("#### " + t("why"))
        push = family_push(model, row)
        bars = alt.Chart(push).mark_bar().encode(
            x=alt.X("bikes:Q", title=t("push_axis")),
            y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.condition(alt.datum.bikes > 0, alt.value("#1D9E75"), alt.value("#D85A30")),
            tooltip=["factor", alt.Tooltip("bikes:Q", format="+.2f")])
        st.altair_chart(bars, use_container_width=True)
        st.caption(t("why_note"))

    with right:
        st.markdown("#### " + t("whatif"))
        lo, hi = -min(now, 15), min(free, 15)
        add = st.slider(t("truck"), lo, hi, 0, key=f"truck_{sno}_{as_of}") if hi > lo else 0
        use_rain = st.checkbox(t("rain_on"), key="rain_on")
        rain = st.slider(t("rain"), 0.0, 10.0, 0.0, 0.5, key="rain_mm") if use_rain else None
        s_pred = float(predict(model, scenario(row, add, rain))[0])
        a, b = st.columns(2)
        a.metric(t("fc_asis"), f"{pred:.1f}")
        b.metric(t("fc_whatif"), f"{s_pred:.1f}", t("delta_bikes", d=s_pred - pred))
        if add > 0:
            st.info(t("kept", n=add, k=max(s_pred - pred, 0)))
        elif add < 0:
            st.info(t("removed", n=-add, k=max(pred - s_pred, 0)))
        if use_rain:
            st.warning(t("rain_warn", mm=explain["what_if_rain"]["max_precip_seen"]))

# ------------------------------------------------------------ tab 2: at risk
with tab2:
    allp = latest.copy()
    allp["forecast"] = predict(model, allp)
    allp["free_forecast"] = allp.docks_free_now - (allp.forecast - allp.bikes_now)
    allp["risk"] = np.where(allp.forecast < 1, "empty", np.where(allp.free_forecast < 1, "full", "ok"))
    risky = allp[allp.risk != "ok"]
    st.subheader(t("risk_title", target=target))
    a, b, c = st.columns(3)
    a.metric(t("likely_empty"), int((risky.risk == "empty").sum()))
    b.metric(t("likely_full"), int((risky.risk == "full").sum()))
    c.metric(t("checked"), len(allp))
    if replay:
        e_, f_ = risky[risky.risk == "empty"], risky[risky.risk == "full"]
        st.markdown(t("risk_hits", en=len(e_), eh=int((e_.y_bikes <= 1).sum()),
                      fn=len(f_), fh=int((f_.y_docks_free <= 1).sum())))
    pts = [{"lat": float(la), "lon": float(lo_), "name": n, "tip": t("tip_" + r),
            "color": [216, 90, 48] if r == "empty" else [24, 95, 165]}
           for la, lo_, n, r in zip(risky.lat, risky.lng, risky.label, risky.risk)]
    if pts:
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=float(np.mean([p["lat"] for p in pts])),
                                             longitude=float(np.mean([p["lon"] for p in pts])), zoom=10.5),
            layers=[pdk.Layer("ScatterplotLayer", data=pts, get_position="[lon, lat]", get_fill_color="color",
                              get_radius=120, radius_min_pixels=4, pickable=True)],
            tooltip={"text": "{name}: {tip}"}))
    else:
        st.success(t("no_risk"))
    st.caption(t("map_note"))
    view = risky.assign(city=risky.city.map({"taipei": t("taipei"), "newtaipei": t("newtaipei")}))
    show = ["label", "district", "city", "capacity", "bikes_now", "forecast"] + (["y_bikes"] if replay else [])
    cols = {"label": t("station"), "district": t("district"), "city": t("city"),
            "capacity": t("col_docks"), "bikes_now": t("bikes_now"), "forecast": t("col_forecast"),
            "y_bikes": t("col_actual")}
    left, right = st.columns(2)
    left.markdown(f"**{t('likely_empty')}**")
    left.dataframe(view[view.risk == "empty"].sort_values("bikes_now", ascending=False)[show]
                   .rename(columns=cols).round(1), hide_index=True, use_container_width=True)
    right.markdown(f"**{t('likely_full')}**")
    right.dataframe(view[view.risk == "full"].sort_values("bikes_now")[show]
                    .rename(columns=cols).round(1), hide_index=True, use_container_width=True)
    e, f = metrics["metrics"]["lightgbm"]["empty"], metrics["metrics"]["lightgbm"]["full"]
    st.caption(t("trust", ep=e["precision"], er=e["recall"], fp=f["precision"], fr=f["recall"]))

# ----------------------------------------------------- tab 3: model and limits
with tab3:
    per = metrics["periods_local"]
    st.subheader(t("beat"))
    st.caption(t("split", a=per["train"][0][:16], b=per["train"][1][:16],
                 c=per["test"][0][:16], d=per["test"][1][:16], n=metrics["rows"]["test"]))
    mm = metrics["metrics"]
    base = mm["baseline_now"]["mae"]
    tbl = pd.DataFrame([
        {t("method"): label, t("mae"): mm[k]["mae"], "RMSE": mm[k]["rmse"],
         t("vs_now"): f"{100 * (mm[k]['mae'] / base - 1):+.1f}%"}
        for k, label in (("baseline_now", t("b_now")), ("baseline_yday", t("b_yday")),
                         ("lightgbm", "LightGBM"))])
    st.dataframe(tbl.round(3), hide_index=True, use_container_width=True)

    st.subheader(t("drives"))
    fam_df = pd.DataFrame([{"factor": fam(k), "bikes": v}
                           for k, v in explain["drivers"]["families"].items()])
    st.altair_chart(alt.Chart(fam_df).mark_bar(color="#534AB7").encode(
        x=alt.X("bikes:Q", title=t("avg_push")),
        y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260))), use_container_width=True)
    r = explain["what_if_refill"][0]
    st.markdown(t("findings", k=r["k"], h=r["hour"], kept=r["kept_mean"], p10=r["kept_p10"]))

    st.subheader(t("limits"))
    st.markdown(t("limits_body"))
