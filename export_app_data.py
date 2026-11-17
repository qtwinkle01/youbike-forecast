"""Pack what the app needs into app_data/ (small enough to commit and to host).

Run:  docker compose run --rm ml python export_app_data.py     (after train.py and explain.py)

out/ holds the full derived tables and is not committed. The app only needs:
  latest.parquet     newest slot, for the live forecast
  replay.parquet     the held-out test period with the real outcome, for replay
  history.parquet    bike counts around the test period, for the charts
  model.txt          trained WITHOUT the test period  -> used for replay (honest)
  model_full.txt     trained on everything            -> used for the live forecast
  stations.parquet, metrics.json, explain.json
"""
import json
import shutil
from pathlib import Path

import pandas as pd

OUT, APP = Path("out"), Path("app_data")


def main():
    APP.mkdir(exist_ok=True)
    metrics = json.loads((OUT / "metrics.json").read_text())
    test_start = pd.Timestamp(metrics["periods_local"]["test"][0])

    table = pd.read_parquet(OUT / "model_table.parquet")
    replay = table[table.slot_local >= test_start].sort_values(["slot_utc", "sno"])
    replay.to_parquet(APP / "replay.parquet", index=False, row_group_size=20_000)

    snap = pd.read_parquet(OUT / "snapshots.parquet", columns=["slot_local", "sno", "bikes", "act"])
    hist = snap[snap.slot_local >= test_start - pd.Timedelta(hours=24)].sort_values(["sno", "slot_local"])
    hist.to_parquet(APP / "history.parquet", index=False, row_group_size=50_000)

    for name in ("latest.parquet", "stations.parquet", "model.txt", "model_full.txt",
                 "metrics.json", "explain.json"):
        shutil.copy(OUT / name, APP / name)
    (APP / "history_live.parquet").unlink(missing_ok=True)      # belongs to an older quick refresh

    size = sum(f.stat().st_size for f in APP.iterdir()) / 1e6
    print(f"app_data/: {len(replay):,} replay rows from {test_start}, {len(hist):,} history rows, {size:.1f} MB")


if __name__ == "__main__":
    main()
