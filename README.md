# YouBike: how many bikes in one hour?

Predicts how many bikes each YouBike station in Taipei and New Taipei will have one hour from now,
and explains what drives the forecast.

- **Report (results, drivers, limits, next steps):** [docs/REPORT.md](docs/REPORT.md) · [中文版](docs/REPORT.zh-TW.md)
- **Dataset documentation:** [docs/DATASET.md](docs/DATASET.md)

## What is in this repository

| File | Role |
|---|---|
| `collector.py` | Fetches both YouBike feeds every 5 minutes and Open-Meteo every hour; stores raw responses |
| `build_dataset.py` | Raw files → clean tables → model table (reproducible) |
| `train.py` | Two naive baselines vs LightGBM, time-ordered split |
| `explain.py` | Drivers (SHAP) and what-if scenarios |
| `app.py` | Streamlit app, English / 中文 |
| `export_app_data.py` | Packs what the app needs into `app_data/` (committed, so the app runs without the pipeline) |
| `make_report.py` | Regenerates the report and its charts from the latest run |
| `backup.ps1` | One-command backup of `data/` to this repository |
| `data/raw/`, `data/fetch_log.jsonl` | The collected history (committed, so everything can be reproduced) |

## Run it

Requires Docker. Nothing else needs to be installed.

**1. Collector** (already has several days of data in `data/raw/`; start it to keep collecting)

    docker compose up -d --build
    docker compose logs -f

**2. Dataset, model, explanations, report** (about 15 minutes in total)

    docker compose build ml
    docker compose run --rm ml python build_dataset.py   # raw -> out/*.parquet
    docker compose run --rm ml python train.py           # baselines vs LightGBM, test = last 24 h
    docker compose run --rm ml python explain.py         # drivers + what-if
    docker compose run --rm ml python export_app_data.py # app_data/ (what the app reads)
    docker compose run --rm ml python make_report.py     # docs/REPORT*.md + charts

**3. App**

    docker compose --profile app up app                  # then open http://localhost:8502

Three tabs: one station's forecast with its explanation and what-if sliders; stations likely to be
empty or full in an hour (map + lists); model scores, drivers and limits.

The app reads `app_data/`, which is committed, so step 3 works straight after cloning.
It has two views: **Latest** (newest snapshot; with the collector running, press "Update to the
newest data", about a minute and a half) and **Replay** (any moment of the held-out test day,
forecast next to the real outcome). `out/` holds the full derived tables and is not committed.

## Checking the collector

    tail -n 5 data/fetch_log.jsonl        # every attempt, ok/error, rows, elapsed
    docker ps                             # STATUS shows (healthy)

    data/raw/<source>/YYYY/MM/DD/<fetch time UTC>[_pN].json.gz   # raw, untouched
    data/fetch_log.jsonl                                          # fetch history
    data/heartbeat                                                # touched on each successful round

Design notes live at the top of each script: time and leakage rules in `build_dataset.py`,
the train / validation / test split in `train.py`.
