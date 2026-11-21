import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import prepare

FIXTURES = Path(__file__).parent / "fixtures"


def test_daily_max_one_row_per_local_date():
    for name in ["auckland", "auckland_dst"]:
        raw = json.loads((FIXTURES / f"forecast_{name}.json").read_text())
        daily = prepare.daily_max(raw)
        assert daily["date"].is_unique
        assert len(daily) == 3
        assert "ecmwf_ifs025" in daily.columns  # prefix removed from column names
        hours = pd.Series(raw["hourly"]["temperature_2m_previous_day1_jma_seamless"][:24])
        assert daily["jma_seamless"].iloc[0] == hours.max()


def test_daily_max_is_missing_when_an_hour_is_missing():
    raw = json.loads((FIXTURES / "forecast_auckland.json").read_text())
    raw["hourly"]["temperature_2m_previous_day1_gfs_seamless"][5] = None
    daily = prepare.daily_max(raw)
    assert np.isnan(daily["gfs_seamless"].iloc[0])
    assert not np.isnan(daily["gfs_seamless"].iloc[1])


def test_build_table_joins_actuals(tmp_path):
    shutil.copy(FIXTURES / "forecast_auckland.json", tmp_path / "forecast_auckland.json")
    shutil.copy(FIXTURES / "actuals_auckland.json", tmp_path / "actuals_auckland.json")
    table = prepare.build_table(tmp_path)
    assert list(table["place"].unique()) == ["auckland"]
    assert table["actual"].notna().all() and len(table) == 3


def make_table(days=100):
    dates = pd.date_range("2024-01-01", periods=days).strftime("%Y-%m-%d")
    return pd.DataFrame({"place": "auckland", "date": dates, "a": 20.0, "b": 21.0, "actual": 20.5})


def test_select_models_drops_gappy_model():
    table = make_table()
    table.loc[:9, "b"] = np.nan  # b is missing 10% of days
    kept, dropped = prepare.select_models(table, ["a", "b"], min_coverage=0.95)
    assert kept == ["a"]
    assert "b" in dropped and "90.0%" in dropped["b"]


def test_check_flags_duplicates_and_range():
    table = make_table()
    assert prepare.check(table, ["a", "b"], rows_before=100) == []
    bad = pd.concat([table, table.iloc[[0]]])
    bad.loc[5, "a"] = 55.0
    problems = prepare.check(bad, ["a", "b"], rows_before=101)
    assert any("duplicate" in p for p in problems)
    assert any("outside" in p for p in problems)


def test_check_flags_too_many_rows_dropped():
    problems = prepare.check(make_table(80), ["a", "b"], rows_before=100)
    assert any("dropped" in p for p in problems)


def test_mark_test_last_n_days_by_time():
    table = make_table(100).sample(frac=1, random_state=0)  # shuffled on purpose
    marked = prepare.mark_test(table, test_days=30)
    test_dates = marked.loc[marked["is_test"], "date"]
    assert len(test_dates) == 30
    assert test_dates.min() == "2024-03-11" and test_dates.max() == "2024-04-09"
