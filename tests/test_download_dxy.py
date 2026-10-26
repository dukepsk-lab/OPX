import pandas as pd
import pytest
from scripts.download_dxy import format_dxy_dataframe


def test_format_dxy_dataframe():
    raw = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2023-01-01 00:00:00"),
                pd.Timestamp("2023-01-01 00:01:00"),
            ],
            "open": [103.50, 103.52],
            "high": [103.55, 103.58],
            "low": [103.48, 103.50],
            "close": [103.52, 103.57],
            "volume": [12.0, 15.0],
            "extra_col": [1, 2],
        }
    )
    formatted = format_dxy_dataframe(raw)
    assert list(formatted.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert formatted["timestamp"].dt.tz is not None
    assert str(formatted["timestamp"].dt.tz) == "UTC"
    assert len(formatted) == 2
