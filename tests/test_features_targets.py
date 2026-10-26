import numpy as np
import pandas as pd

from fh_ml_addon.features import build_features
from fh_ml_addon.targets import make_target


def _frame(dates, values):
    return pd.DataFrame({"date": pd.to_datetime(dates), "sector": "S1", "m7_C": values, "m7_F": values})


def test_lag_is_exact_calendar_day_not_previous_row():
    frame = _frame(["2020-01-01", "2020-01-03", "2020-01-04"], [0.0, 3.0, 4.0])
    features, _ = build_features(frame)
    row_jan3 = features.index[features["date"].eq(pd.Timestamp("2020-01-03"))][0]
    row_jan4 = features.index[features["date"].eq(pd.Timestamp("2020-01-04"))][0]
    assert np.isnan(features.loc[row_jan3, "m7_C_lag_1d"])
    assert features.loc[row_jan4, "m7_C_lag_1d"] == 3.0


def test_past_features_do_not_change_when_future_values_change():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    values = np.arange(40, dtype=float) / 10
    a = pd.DataFrame({"date": dates, "sector": "S1", "m7_C": values, "m7_F": values})
    b = a.copy()
    b.loc[b["date"] > pd.Timestamp("2020-01-20"), "m7_C"] = 999.0
    fa, _ = build_features(a)
    fb, _ = build_features(b)
    cols = ["now_m7_C", "m7_C_lag_1d", "m7_C_rolling_mean_7d", "C_momentum_3d_vs_30d"]
    a_mask = fa["date"] <= pd.Timestamp("2020-01-20")
    pd.testing.assert_frame_equal(fa.loc[a_mask, cols].reset_index(drop=True), fb.loc[a_mask, cols].reset_index(drop=True))


def test_future_target_requires_complete_calendar_window():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    values = np.zeros(40, dtype=float)
    values[3] = 2.5  # Jan 4 will be a positive future day for Jan 1.
    frame = pd.DataFrame({"date": dates, "sector": "S1", "m7_C": values})
    y7, valid7, _ = make_target(frame, 7, c_threshold=2.0)
    assert bool(valid7.iloc[0]) and int(y7.iloc[0]) == 1
    assert not bool(valid7.iloc[-1])


def test_30_day_target_uses_configured_active_day_count():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    values = np.zeros(40, dtype=float)
    values[[2, 5, 10]] = 2.5
    frame = pd.DataFrame({"date": dates, "sector": "S1", "m7_C": values})
    y30, valid30, definition = make_target(frame, 30, c_threshold=2.0, elevated_min_active_days=3)
    assert bool(valid30.iloc[0]) and int(y30.iloc[0]) == 1
    assert "at least 3" in definition
