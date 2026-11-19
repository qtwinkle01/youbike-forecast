"""Checks that the promises in the report still hold for the latest run.

Run:  docker compose run --rm ml python check_outputs.py     (after make_report.py)

Every promise we make in the report (and in the private slides, when present) is checked here against the data,
so a rerun cannot silently break one. Exit code 1 if any check fails.

If a check fails: fix the pipeline or the wording of the report. Do NOT loosen a
threshold or delete a check to make it pass; a red check means a promise is false.
"""
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

OUT, APP, DOCS = Path("out"), Path("app_data"), Path("docs")
FETCH_LOG = Path("data/fetch_log.jsonl")
MIN_DAYS = 5
SAMPLE = 200_000


def check_collection(log_rows, report):
    """Assignment: at least 5 days of timestamped history."""
    errs = []
    for src in ("youbike_taipei", "youbike_newtaipei"):
        ts = sorted(datetime.fromisoformat(r["ts"]) for r in log_rows if r["source"] == src and r["ok"])
        if len(ts) < 200:
            errs.append(f"{src}: only {len(ts)} successful fetches in fetch_log; the path or source name is "
                        "probably wrong (this guard stops the check from passing on empty input).")
            continue
        span = ts[-1] - ts[0]
        if span < timedelta(days=MIN_DAYS):
            errs.append(f"{src}: successful fetches span {span}, the assignment asks for {MIN_DAYS} days.")
    share = report["slots_missing"] / report["slots_expected"]
    if share > 0.05:
        errs.append(f"dataset_report: {share:.1%} of 5-minute slots missing (> 5%). The report calls coverage "
                    "near-complete; investigate the gaps in missing_ranges_local.")
    return errs


def check_split(metrics):
    """Time-ordered split with a 1-hour gap between parts; test = the newest period."""
    errs = []
    p = {k: [pd.Timestamp(v[0]), pd.Timestamp(v[1])] for k, v in metrics["periods_local"].items()}
    hour = pd.Timedelta(hours=1)
    if not p["train"][1] + hour < p["valid"][0]:
        errs.append(f"train ends {p['train'][1]}, valid starts {p['valid'][0]}: less than 1 hour apart, so "
                    "training targets overlap validation rows. Fix split() in train.py.")
    if not p["valid"][1] + hour < p["test"][0]:
        errs.append(f"valid ends {p['valid'][1]}, test starts {p['test'][0]}: less than 1 hour apart, so the "
                    "test answers leak into early stopping. Fix split() in train.py.")
    for name in ("train", "valid", "test"):
        if metrics["rows"][name] < 1000:
            errs.append(f"{name} has {metrics['rows'][name]} rows; the split is broken.")
    leaky = [f for f in metrics["features"] if f.startswith("y_") or "target" in f and "yday" not in f]
    if leaky:
        errs.append(f"features {leaky} look like targets. A feature must only use data known at the slot.")
    return errs


def check_leakage(table, snap):
    """Each row's target is the real count 1 h later; its features are from the past.

    Re-derives values from snapshots.parquet independently of build_dataset.py.
    """
    errs = []
    rows = table.sample(min(SAMPLE, len(table)), random_state=0)
    s = snap[snap.act == 1].set_index(["slot_local", "sno"])["bikes"]

    def lookup(offset):
        key = pd.MultiIndex.from_arrays([rows.slot_local + offset, rows.sno])
        return pd.Series(s.reindex(key).to_numpy(), index=rows.index)

    for col, offset, meaning in (("bikes_now", pd.Timedelta(0), "the count at the slot"),
                                 ("y_bikes", pd.Timedelta(hours=1), "the count 1 hour later"),
                                 ("bikes_yday_target", pd.Timedelta(hours=-23), "the count 23 hours earlier")):
        truth = lookup(offset)
        known = truth.notna() & rows[col].notna()
        if known.sum() < 1000:
            errs.append(f"{col}: only {known.sum()} sampled rows could be compared; the check would pass "
                        "vacuously. Check the slot_local / sno columns.")
            continue
        bad = (rows.loc[known, col] != truth[known]).sum()
        if bad:
            errs.append(f"{col}: {bad} of {known.sum()} sampled rows differ from {meaning} in "
                        "snapshots.parquet. The time shift in build_model_table() is wrong.")
    age = table["weather_age_min"].dropna()
    if (age < 0).any():
        errs.append(f"{(age < 0).sum()} rows use weather downloaded AFTER their slot (weather_age_min < 0). "
                    "That is leakage; merge_asof in read_weather() must look backward only.")
    if age.max() > 6 * 60:
        errs.append(f"weather up to {age.max():.0f} min old is used; WEATHER_MAX_AGE is 6 h.")
    return errs


def check_scores(metrics):
    """Assignment: beat a naive baseline."""
    m = metrics["metrics"]
    errs = []
    for base in ("baseline_now", "baseline_yday"):
        if not m["lightgbm"]["mae"] < m[base]["mae"]:
            errs.append(f"LightGBM MAE {m['lightgbm']['mae']:.3f} does not beat {base} {m[base]['mae']:.3f}. "
                        "Say so in the report; do not change the test period to find a better number.")
    return errs


def check_freshness(metrics, report):
    """App, report and model all come from the same, newest run."""
    errs = []
    if (APP / "metrics.json").read_text() != (OUT / "metrics.json").read_text():
        errs.append("app_data/metrics.json differs from out/metrics.json: run export_app_data.py.")
    test_end = pd.Timestamp(metrics["periods_local"]["test"][1])
    last = pd.Timestamp(report["last_slot_local"])
    if last - test_end > pd.Timedelta(hours=1):
        errs.append(f"test ends {test_end} but the data goes to {last}: train.py ran on an older dataset.")
    for doc in ("REPORT.md", "REPORT.zh-TW.md"):
        text = (DOCS / doc).read_text(encoding="utf-8")
        mae = f"{metrics['metrics']['lightgbm']['mae']:.3f}"
        if mae not in text:
            errs.append(f"docs/{doc} does not contain the current LightGBM MAE {mae}: run make_report.py.")
    return errs


def check_report_claims(log_rows, report):
    """Sentences in the report that state facts about the collection."""
    errs = []
    text = (DOCS / "REPORT.md").read_text(encoding="utf-8")
    gaps = report["missing_ranges_local"]
    bike_fail = [r for r in log_rows if r["source"].startswith("youbike") and not r["ok"]]
    in_gaps = 0
    for r in bike_fail:
        t = pd.Timestamp(r["ts"]).tz_convert("Asia/Taipei").tz_localize(None)
        if any(pd.Timestamp(a) - pd.Timedelta(minutes=10) <= t <= pd.Timestamp(b) + pd.Timedelta(minutes=10)
               for a, b in (g.split(" -> ") for g in gaps)):
            in_gaps += 1
    if re.search(r"failures all fall in", text) and in_gaps != len(bike_fail):
        errs.append(f"REPORT.md says all bike failures fall in the gap, but {len(bike_fail) - in_gaps} of "
                    f"{len(bike_fail)} are outside it. Compute that sentence in make_report.py.")
    claim = re.search(r"(\d+) of the (\d+) bike fetch failures fall in this window", text)
    if claim and (int(claim[1]), int(claim[2])) != (in_gaps, len(bike_fail)):
        errs.append(f"REPORT.md says {claim[1]} of {claim[2]} bike failures fall in the gap; fetch_log gives "
                    f"{in_gaps} of {len(bike_fail)}. Rerun make_report.py.")
    if not claim and not re.search(r"failures all fall in", text):
        errs.append("REPORT.md no longer states where the bike failures fall; this check found nothing to "
                    "verify. Update the pattern here to the new wording, do not delete the check.")
    day = json.loads((OUT / "explain.json").read_text(encoding="utf-8"))["what_if_daytype"][0]
    moved = day["share_moved_1plus"]
    if "label barely matters" in text and moved >= 0.05:
        errs.append(f"REPORT.md says the weekday label barely matters, but relabelling as Sunday moves "
                    f"{moved:.1%} of stations by 1+ bike. Rerun make_report.py (the verdict is computed).")
    if "An early signal" in text and moved < 0.05:
        errs.append(f"REPORT.md calls the Sunday what-if an early signal, but only {moved:.1%} of stations move.")
    for minutes in re.findall(r"cost (\d+) minutes", text):
        longest = max((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 60 + 5
                      for a, b in (g.split(" -> ") for g in gaps)) if gaps else 0
        if int(minutes) != round(longest):
            errs.append(f"REPORT.md says an outage cost {minutes} minutes, the longest gap is {longest:.0f}.")
    return errs


def check_slides(metrics):
    """The deck shows the current run, and no text box obviously overflows or leaves the slide."""
    from math import ceil

    from pptx import Presentation
    errs = []
    path = DOCS / "slides.pptx"
    if not Path("make_slides.py").exists():
        return None  # the deck is private and not in the public repository; nothing to check here
    if not path.exists():
        return ["docs/slides.pptx is missing: run make_slides.py."]
    prs = Presentation(path)
    if len(prs.slides) < 10:
        errs.append(f"slides.pptx has {len(prs.slides)} slides; expected the full deck (15).")
    words = " ".join(sh.text_frame.text for sl in prs.slides for sh in sl.shapes if sh.has_text_frame)
    mae = f"{metrics['metrics']['lightgbm']['mae']:.3f}"
    if mae not in words:
        errs.append(f"slides.pptx does not show the current LightGBM MAE {mae}: run make_slides.py.")
    W, H = prs.slide_width, prs.slide_height
    for i, sl in enumerate(prs.slides, 1):
        for sh in sl.shapes:
            if sh.left < 0 or sh.top < 0 or sh.left + sh.width > W + 9525 or sh.top + sh.height > H + 9525:
                errs.append(f"slide {i}: shape '{sh.name}' extends past the slide edge.")
            if not sh.has_text_frame or not sh.text_frame.text.strip():
                continue
            width_pt, need = sh.width / 12700, 0.0
            for p in sh.text_frame.paragraphs:
                size = max((r.font.size.pt for r in p.runs if r.font.size), default=18)
                # Arial averages ~0.52 em per Latin character; a CJK character is a full em
                em = sum(1.0 if "⺀" <= ch <= "鿿" or "＀" <= ch <= "￯" else 0.52 for ch in p.text)
                need += max(1, ceil(em * size / width_pt)) * size * 1.25
            if need > sh.height / 12700 * 1.25 + 6:
                errs.append(f"slide {i}: text '{sh.text_frame.text[:40]}' needs ~{need:.0f} pt but its box "
                            f"is {sh.height / 12700:.0f} pt, so it spills onto what is below. Shorten the "
                            "text or give it more room in make_slides.py.")
    return errs


def main():
    log_rows = [json.loads(line) for line in FETCH_LOG.read_text(encoding="utf-8").splitlines() if line]
    report = json.loads((OUT / "dataset_report.json").read_text())
    metrics = json.loads((OUT / "metrics.json").read_text())
    table = pd.read_parquet(OUT / "model_table.parquet",
                            columns=["slot_local", "sno", "bikes_now", "y_bikes", "bikes_yday_target",
                                     "weather_age_min"])
    snap = pd.read_parquet(OUT / "snapshots.parquet", columns=["slot_local", "sno", "bikes", "act"])

    checks = [("collection >= 5 days", check_collection(log_rows, report)),
              ("time split with 1 h gaps", check_split(metrics)),
              ("no leakage (re-derived)", check_leakage(table, snap)),
              ("beats naive baselines", check_scores(metrics)),
              ("app/report match latest run", check_freshness(metrics, report)),
              ("report facts match data", check_report_claims(log_rows, report)),
              ("slides match latest run", check_slides(metrics))]
    skipped = [name for name, errs in checks if errs is None]
    checks = [(name, errs) for name, errs in checks if errs is not None]
    for name in skipped:
        print(f"skip  {name} (private file not in this copy)")
    failed = 0
    for name, errs in checks:
        print(f"{'FAIL' if errs else 'ok  '}  {name}")
        for e in errs:
            print(f"      - {e}")
        failed += bool(errs)
    print(f"\n{len(checks) - failed} passed, {failed} failed, {len(skipped)} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
