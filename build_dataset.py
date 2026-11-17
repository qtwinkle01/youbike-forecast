"""Raw snapshots -> clean tables -> model table.

Run:  docker compose run --rm ml python build_dataset.py

Reads   data/raw/**, data/fetch_log.jsonl          (never modified)
Writes  out/snapshots.parquet    one row per station per 5-minute slot (clean, long)
        out/stations.parquet     one row per station (static info)
        out/weather.parquet      one row per city per 5-minute slot (what we knew then)
        out/model_table.parquet  features + target, ready for train.py
        out/dataset_report.json  counts, gaps, dropped rounds

Time handling
  * File names carry OUR fetch time in UTC. Each fetch is assigned to the nearest
    5-minute slot. If several fetches land in one slot, the closest one wins.
  * `slot_utc` is UTC; `slot_local` is Asia/Taipei (UTC+8, no DST) and is what
    hour / weekday features are computed from.
Leakage rule
  * Every feature of a row uses only data fetched at or before that row's slot.
    Weather for the target hour comes from the forecast we had already downloaded.
"""
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("out")
SLOT = 300                    # seconds
HORIZON = 12                  # slots ahead = 1 hour
DAY = 288                     # slots per day
K_NEIGHBORS = 5
WEATHER_MAX_AGE = pd.Timedelta(hours=6)   # older weather than this is treated as missing
TPE = timedelta(hours=8)


def stamp_to_dt(stamp):
    return datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc)


def to_slot(dt):
    """Nearest 5-minute slot, plus distance to it in seconds."""
    epoch = dt.timestamp()
    slot = round(epoch / SLOT) * SLOT
    return datetime.fromtimestamp(slot, timezone.utc).replace(tzinfo=None), abs(epoch - slot)


def load(path):
    with gzip.open(path) as f:
        return json.loads(f.read())


def failed_rounds():
    """Fetch-time stamps of rounds the collector logged as failed (may be partial)."""
    bad = {}
    log = Path("data/fetch_log.jsonl")
    if not log.exists():
        return bad
    for line in log.open(encoding="utf-8"):
        e = json.loads(line)
        if not e["ok"]:
            stamp = datetime.fromisoformat(e["ts"]).strftime("%Y-%m-%dT%H-%M-%SZ")
            bad.setdefault(e["source"], set()).add(stamp)
    return bad


# ---------------------------------------------------------------- snapshots
def read_snapshots(report):
    bad = failed_rounds()
    rows = []          # slot, dist, sno, city, bikes, docks_free, capacity, act, station_ts
    static = {}        # sno -> (city, name, district, lat, lng)   (latest wins)
    dropped = []

    for path in sorted((RAW / "youbike_taipei").rglob("*.json.gz")):
        stamp = path.name[:20]
        if stamp in bad.get("youbike_taipei", ()):
            dropped.append(path.name)
            continue
        slot, dist = to_slot(stamp_to_dt(stamp))
        for r in load(path):
            rows.append((slot, dist, r["sno"], "taipei", r["available_rent_bikes"],
                         r["available_return_bikes"], r["Quantity"], int(r["act"]),
                         r["mday"]))
            static[r["sno"]] = ("taipei", r["sna"], r["sarea"], r["latitude"], r["longitude"])

    for path in sorted((RAW / "youbike_newtaipei").rglob("*.json.gz")):
        stamp = path.name[:20]
        if stamp in bad.get("youbike_newtaipei", ()):   # partial round: some pages missing
            dropped.append(path.name)
            continue
        slot, dist = to_slot(stamp_to_dt(stamp))
        for r in load(path):
            m = r["mday"]
            rows.append((slot, dist, r["sno"], "newtaipei", int(r["sbi_quantity"]),
                         int(r["bemp"]), int(r["tot_quantity"]), int(r["act"]),
                         f"{m[:4]}-{m[4:6]}-{m[6:8]} {m[9:11]}:{m[11:13]}:{m[13:15]}"))
            static[r["sno"]] = ("newtaipei", r["sna"], r["sarea"], float(r["lat"]), float(r["lng"]))

    df = pd.DataFrame(rows, columns=["slot_utc", "dist", "sno", "city", "bikes", "docks_free",
                                     "capacity", "act", "station_ts"])
    n_raw = len(df)
    # several fetches in one slot (manual test runs, restarts): keep the closest one
    df = df.sort_values("dist").drop_duplicates(["slot_utc", "sno"], keep="first")
    df = df.drop(columns="dist").sort_values(["slot_utc", "sno"]).reset_index(drop=True)
    df["station_ts"] = pd.to_datetime(df["station_ts"])          # station's own clock, local time
    df["slot_local"] = df["slot_utc"] + TPE
    for c in ("bikes", "docks_free", "capacity"):
        df[c] = df[c].astype("int16")
    df["act"] = df["act"].astype("int8")

    stations = pd.DataFrame(
        [(k, *v) for k, v in static.items()],
        columns=["sno", "city", "name", "district", "lat", "lng"]).sort_values("sno")
    stations["capacity"] = stations["sno"].map(df.groupby("sno")["capacity"].median()).astype("int16")
    stations = stations[(stations.lat > 0) & (stations.lng > 0)].reset_index(drop=True)

    report.update(raw_rows=n_raw, clean_rows=len(df),
                  duplicate_rows_removed=n_raw - len(df),
                  rounds_dropped_as_failed=dropped, stations=len(stations))
    return df, stations


# ------------------------------------------------------------------ weather
def read_weather(slots):
    """For every slot and city: the newest weather download made at or before the slot."""
    cur, fc = [], []
    for source, city in (("weather_taipei", "taipei"), ("weather_newtaipei", "newtaipei")):
        for path in sorted((RAW / source).rglob("*.json.gz")):
            fetched = stamp_to_dt(path.name[:20]).replace(tzinfo=None)
            w = load(path)
            c, h = w["current"], w["hourly"]
            cur.append((city, fetched, c["temperature_2m"], c["precipitation"],
                        c["relative_humidity_2m"], c["wind_speed_10m"]))
            for i, t in enumerate(h["time"]):                      # local (Asia/Taipei) hours
                fc.append((city, fetched, pd.Timestamp(t), h["temperature_2m"][i],
                           h["precipitation"][i], h["precipitation_probability"][i]))
    cur = pd.DataFrame(cur, columns=["city", "fetched_utc", "cur_temp", "cur_precip",
                                     "cur_humidity", "cur_wind"])
    fc = pd.DataFrame(fc, columns=["city", "fetched_utc", "valid_local", "fc_temp",
                                   "fc_precip", "fc_precip_prob"])
    out = []
    for city in ("taipei", "newtaipei"):
        base = pd.DataFrame({"slot_utc": slots, "city": city})
        m = pd.merge_asof(base, cur[cur.city == city].drop(columns="city").sort_values("fetched_utc"),
                          left_on="slot_utc", right_on="fetched_utc",
                          direction="backward", tolerance=WEATHER_MAX_AGE)
        # forecast for the hour we are predicting, taken from that same download
        m["valid_local"] = (m["slot_utc"] + TPE + pd.Timedelta(hours=1)).dt.floor("h")
        m = m.merge(fc[fc.city == city].drop(columns="city"),
                    on=["fetched_utc", "valid_local"], how="left")
        out.append(m)
    w = pd.concat(out, ignore_index=True)
    w["weather_age_min"] = (w["slot_utc"] - w["fetched_utc"]).dt.total_seconds() / 60
    return w


# ----------------------------------------------------------------- features
def nearest_neighbors(lat, lng, k):
    la, lo = np.radians(lat)[:, None], np.radians(lng)[:, None]
    d = 2 * 6371 * np.arcsin(np.sqrt(
        np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2))
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    return idx, np.take_along_axis(d, idx, axis=1).mean(axis=1)


def shift(a, n):
    """Row r of the result holds a[r - n]; rows without a source are NaN."""
    out = np.full_like(a, np.nan)
    if n > 0:
        out[n:] = a[:-n]
    elif n < 0:
        out[:n] = a[-n:]
    else:
        out[:] = a
    return out


def build_model_table(snap, stations, weather):
    grid = pd.date_range(snap.slot_utc.min(), snap.slot_utc.max(), freq="5min")
    snos = stations.sno.to_numpy()

    def wide(col):
        return (snap.pivot(index="slot_utc", columns="sno", values=col)
                .reindex(index=grid, columns=snos).to_numpy(dtype="float32"))

    B, D, A = wide("bikes"), wide("docks_free"), wide("act")
    B[A != 1] = np.nan            # station switched off: its numbers are not real demand
    D[A != 1] = np.nan
    cap = stations.capacity.to_numpy(dtype="float32")[None, :]
    fill = B / np.maximum(cap, 1)

    nbr, nbr_km = nearest_neighbors(stations.lat.to_numpy(), stations.lng.to_numpy(), K_NEIGHBORS)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        nbr_bikes = np.nanmean(B[:, nbr], axis=2)
        nbr_fill = np.nanmean(fill[:, nbr], axis=2)

    T, S = B.shape
    local = grid + TPE
    hour = (local.hour + local.minute / 60).to_numpy(dtype="float32")
    cols = {
        "bikes_now": B,
        "docks_free_now": D,
        "fill_now": fill,
        "delta_15m": B - shift(B, 3),
        "delta_60m": B - shift(B, HORIZON),
        "bikes_yday_now": shift(B, DAY),                     # 24 h ago
        "bikes_yday_target": shift(B, DAY - HORIZON),        # yesterday, at the time we predict
        "nbr_bikes_mean": nbr_bikes,
        "nbr_fill_mean": nbr_fill,
        "nbr_delta_60m": nbr_bikes - shift(nbr_bikes, HORIZON),
        # ---- targets (the future; never used as a feature)
        "y_bikes": shift(B, -HORIZON),
        "y_docks_free": shift(D, -HORIZON),
    }
    cols["yday_delta_60m"] = cols["bikes_yday_target"] - cols["bikes_yday_now"]

    df = pd.DataFrame({k: v.ravel() for k, v in cols.items()})
    df.insert(0, "slot_utc", np.repeat(grid.to_numpy(), S))
    df.insert(1, "slot_local", np.repeat(local.to_numpy(), S))
    df.insert(2, "sno", np.tile(snos, T))
    df["hour"] = np.repeat(hour, S)
    df["dow"] = np.repeat(local.dayofweek.to_numpy(dtype="int8"), S)      # 0 = Monday
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")   # no public holiday in the collection window
    for c, name in (("capacity", "capacity"), ("lat", "lat"), ("lng", "lng")):
        df[name] = np.tile(stations[c].to_numpy(dtype="float32"), T)
    df["nbr_km_mean"] = np.tile(nbr_km.astype("float32"), T)
    df["city"] = np.tile(stations.city.to_numpy(), T)

    wcols = ["cur_temp", "cur_precip", "cur_humidity", "cur_wind",
             "fc_temp", "fc_precip", "fc_precip_prob", "weather_age_min"]
    df = df.merge(weather[["slot_utc", "city"] + wcols], on=["slot_utc", "city"], how="left")
    df["is_taipei"] = (df["city"] == "taipei").astype("int8")
    for c in wcols:
        df[c] = df[c].astype("float32")

    # keep rows where we know both "now" and the answer one hour later
    df = df[df.bikes_now.notna() & df.y_bikes.notna()].reset_index(drop=True)
    return df, grid


def main():
    OUT.mkdir(exist_ok=True)
    report = {}
    snap, stations = read_snapshots(report)
    slots = pd.Series(sorted(snap.slot_utc.unique()))
    grid_all = pd.date_range(slots.iloc[0], slots.iloc[-1], freq="5min")
    missing = grid_all.difference(pd.DatetimeIndex(slots))
    weather = read_weather(grid_all.to_series().reset_index(drop=True))
    table, grid = build_model_table(snap, stations, weather)

    snap.to_parquet(OUT / "snapshots.parquet", index=False)
    stations.to_parquet(OUT / "stations.parquet", index=False)
    weather.to_parquet(OUT / "weather.parquet", index=False)
    table.to_parquet(OUT / "model_table.parquet", index=False)

    report.update(
        first_slot_local=str(slots.iloc[0] + TPE), last_slot_local=str(slots.iloc[-1] + TPE),
        slots_expected=len(grid_all), slots_present=len(slots), slots_missing=len(missing),
        missing_ranges_local=_ranges(missing + TPE),
        inactive_station_rows=int((snap.act != 1).sum()),
        weather_slots_without_data=int(weather.cur_temp.isna().sum()),
        model_rows=len(table),
        model_rows_with_yesterday=int(table.bikes_yday_target.notna().sum()),
    )
    (OUT / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _ranges(idx):
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for t in idx[1:]:
        if t - prev > pd.Timedelta(minutes=5):
            out.append(f"{start} -> {prev}")
            start = t
        prev = t
    out.append(f"{start} -> {prev}")
    return out


if __name__ == "__main__":
    main()
