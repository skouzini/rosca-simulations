from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_with_defaults():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_app_runs_with_preset_and_rerun():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="preset").set_value("Plant layoff").run()
    at.button(key="run").click().run()
    assert not at.exception
    at.selectbox(key="preset").set_value("None").run()
    at.button(key="run").click().run()
    assert not at.exception
