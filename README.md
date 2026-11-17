# YouBike collector

## Run
    docker compose up -d --build
    docker compose logs -f

## Test once first (fetch everything one time, then exit)
    docker compose run --rm -e RUN_ONCE=1 collector
    find data -type f | head

## Check
    tail -n 5 data/fetch_log.jsonl        # every attempt, ok/error, rows, elapsed
    docker ps                              # STATUS shows (healthy)
    du -sh data

## Layout
    data/raw/<source>/YYYY/MM/DD/<fetch time UTC>[_pN].json.gz   # raw, untouched
    data/fetch_log.jsonl                                          # fetch history
    data/heartbeat                                                # touched on each successful round

## Dataset and model (reproducible from data/raw)
    docker compose build ml
    docker compose run --rm ml python build_dataset.py   # raw -> out/*.parquet (+ dataset_report.json)
    docker compose run --rm ml python train.py           # baselines vs LightGBM, test = last 24 h

`out/` is derived and not committed. Time rules and the leakage rule are documented at the top of
`build_dataset.py`; the train / valid / test split is documented at the top of `train.py`.

Sources: youbike_taipei, youbike_newtaipei (one file per page), weather_taipei, weather_newtaipei.
Back up `data/` regularly (e.g. rsync to another machine).
