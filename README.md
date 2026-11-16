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

Sources: youbike_taipei, youbike_newtaipei (one file per page), weather_taipei, weather_newtaipei.
Back up `data/` regularly (e.g. rsync to another machine).
