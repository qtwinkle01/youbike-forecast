# Dataset documentation

Everything under `out/` is derived by `build_dataset.py` from `data/raw/` and `data/fetch_log.jsonl`.
Delete `out/` and re-run the script to get the same tables again.

## Raw layer (never modified)

```
data/raw/<source>/YYYY/MM/DD/<fetch time UTC>[_pN].json.gz
data/fetch_log.jsonl
```

| Source | Endpoint | Cadence | Notes |
|---|---|---|---|
| `youbike_taipei` | `tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json` | 5 min | One file, all stations |
| `youbike_newtaipei` | `data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json` | 5 min | Paged (`page`, `size=1000`); one file per page, `_p0`, `_p1`, … |
| `weather_taipei`, `weather_newtaipei` | `api.open-meteo.com/v1/forecast` | 60 min | Current conditions + 48 hourly forecast values, one point per city |

- The file name is **our fetch time in UTC**, not a time reported by the API.
- Files are written to a `.tmp` name and renamed, so a half-written file never exists.
- `fetch_log.jsonl` has one line per attempt: `ts`, `source`, `ok`, `rows`, `bytes`, `pages`, `error`, `elapsed_s`.

### Field mapping between the two cities

| Meaning | Taipei | New Taipei |
|---|---|---|
| Station id | `sno` | `sno` |
| Bikes available to rent (**the target**) | `available_rent_bikes` (int) | `sbi_quantity` (string) |
| Free docks | `available_return_bikes` | `bemp` |
| Total docks | `Quantity` | `tot_quantity` |
| Station's own update time (local) | `mday` `YYYY-MM-DD HH:MM:SS` | `mday` `YYYYMMDDTHHMMSS` |
| Position | `latitude`, `longitude` | `lat`, `lng` (strings) |
| In service | `act` (`1` / `0`) | `act` |

Station ids do not overlap between the cities. Bikes + free docks is often less than total docks
(broken docks or bikes), so "full" is defined as free docks = 0, never as bikes = total docks.

## Cleaning rules

1. **Slot.** Each fetch is assigned to the nearest 5-minute slot. If several fetches fall in one slot (manual test runs, restarts), the one closest to the slot time is kept.
2. **Failed rounds.** A round logged as failed in `fetch_log.jsonl` is dropped entirely. For New Taipei this removes rounds where only some pages were saved.
3. **Inactive stations.** Rows with `act != 1` are kept in `snapshots.parquet` but excluded from features and targets.
4. **Gaps.** Lags and targets are looked up by time on a complete 5-minute grid, so a missing slot yields a missing value rather than a value from the wrong time.
5. **Time zone.** `slot_utc` is UTC, `slot_local` is Asia/Taipei (UTC+8, no daylight saving).

## out/snapshots.parquet — one row per station per slot

| Column | Meaning |
|---|---|
| `slot_utc`, `slot_local` | 5-minute slot |
| `sno`, `city` | Station id; `taipei` or `newtaipei` |
| `bikes` | Bikes available to rent |
| `docks_free` | Free docks |
| `capacity` | Total docks |
| `act` | 1 = in service |
| `station_ts` | The station's own last update time (local) |

## out/stations.parquet — one row per station

`sno`, `city`, `name`, `district`, `lat`, `lng`, `capacity` (median over the period).

## out/weather.parquet — one row per city per slot

What we knew at that slot: the newest weather download made at or before it (at most 6 hours old).

| Column | Meaning |
|---|---|
| `fetched_utc`, `weather_age_min` | Which download was used and how old it was |
| `cur_temp`, `cur_precip`, `cur_humidity`, `cur_wind` | Current conditions in that download (°C, mm, %, km/h) |
| `valid_local` | The hour being predicted (slot + 1 h, floored) |
| `fc_temp`, `fc_precip`, `fc_precip_prob` | That download's forecast for `valid_local` |

## out/model_table.parquet — features and target

One row per station per slot where both "now" and "one hour later" are known.

| Group | Columns | Meaning |
|---|---|---|
| Keys | `slot_utc`, `slot_local`, `sno`, `city` | |
| Current state | `bikes_now`, `docks_free_now`, `fill_now`, `capacity` | `fill_now` = bikes / capacity |
| Recent trend | `delta_15m`, `delta_60m` | Change over the last 15 and 60 minutes |
| Yesterday | `bikes_yday_now`, `bikes_yday_target`, `yday_delta_60m` | 24 h ago; yesterday at the time being predicted; the difference |
| Time | `hour`, `dow`, `is_weekend` | Local hour (with minutes as a fraction), 0 = Monday |
| Weather | `cur_*`, `fc_*`, `weather_age_min` | See weather table |
| Neighbours | `nbr_bikes_mean`, `nbr_fill_mean`, `nbr_delta_60m`, `nbr_km_mean` | Over the 5 nearest stations (haversine), across city borders |
| Location | `lat`, `lng`, `is_taipei` | |
| **Target** | `y_bikes`, `y_docks_free` | Values at slot + 60 min. Never used as a feature. |

`out/latest.parquet` has the same feature columns for the newest slot only (no target yet); the app predicts from it.

## Known issues

- Public holidays: none fell in the collection window, so `is_weekend` is the only day-type flag.
- Weather is one point per city, not per station.
- The station feed lags our fetch by roughly 2 minutes (Taipei) and 6 minutes (New Taipei).
