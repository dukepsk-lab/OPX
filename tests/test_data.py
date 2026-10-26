import pandas as pd
import pytest
from pathlib import Path
from opx.data import load_bars


def test_loads_epoch_ms_and_sorts(tmp_path):
    csv = tmp_path / "bars_epoch.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "120000,1.1002,1.1005,1.1001,1.1003,10\n"
        "0,1.1000,1.1005,1.0995,1.1001,12\n"
        "60000,1.1001,1.1004,1.1000,1.1002,15\n"
    )
    bars = load_bars(csv)
    assert list(bars.columns) == ["open", "high", "low", "close", "tick_volume"]
    assert bars.index.is_monotonic_increasing and str(bars.index.tz) == "UTC"
    assert bars.index[0] == pd.Timestamp("1970-01-01", tz="UTC")
    assert len(bars) == 3


def test_iso_timestamps(tmp_path):
    csv = tmp_path / "bars_iso.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02 00:00:00,1.1000,1.1005,1.0995,1.1001,12\n"
        "2024-01-02 00:01:00,1.1001,1.1004,1.1000,1.1002,15\n"
    )
    bars = load_bars(csv)
    assert bars.index[0] == pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert str(bars.index.tz) == "UTC"


def test_duplicate_timestamps_raise(tmp_path):
    csv = tmp_path / "bars_dup.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "60000,1.1000,1.1005,1.0995,1.1001,12\n"
        "60000,1.1001,1.1004,1.1000,1.1002,15\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_bars(csv)


def test_missing_column_raises(tmp_path):
    csv = tmp_path / "bars_missing.csv"
    csv.write_text(
        "timestamp,open,high,low,volume\n"
        "60000,1.1000,1.1005,1.0995,12\n"
    )
    with pytest.raises(ValueError, match="close"):
        load_bars(csv)
