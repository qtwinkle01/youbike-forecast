"""Smoke test: the app starts from app_data/ alone, in both languages and both modes.

Run:  docker compose run --rm ml python check_app.py

Promise checked: "clone the repo and the app runs". Exit code 1 on any exception.
If it fails, fix app.py or rerun export_app_data.py; do not catch the exception in app.py
just to make this pass.
"""
import sys

from streamlit.testing.v1 import AppTest

CASES = [("English", "live"), ("中文", "live"), ("English", "replay"), ("中文", "replay"),
         ("English", "replay_newtaipei")]


def run(lang, mode):
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    at.radio(key="lang").set_value(lang).run()
    if mode.startswith("replay"):
        at.radio(key="mode").set_value("replay").run()
    if mode.endswith("newtaipei"):
        at.radio(key="city").set_value("newtaipei").run()
    texts = len(at.markdown) + len(at.metric)
    return at.exception, texts


def main():
    failed = 0
    for lang, mode in CASES:
        exc, texts = run(lang, mode)
        if exc:
            failed += 1
            print(f"FAIL  {lang:8s} {mode}: {exc[0].message}")
        elif texts < 3:
            failed += 1
            print(f"FAIL  {lang:8s} {mode}: only {texts} text elements rendered; the page is probably empty.")
        else:
            print(f"ok    {lang:8s} {mode}: {texts} text elements, no exception")
    print(f"\n{len(CASES) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
