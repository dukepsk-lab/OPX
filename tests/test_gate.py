import pandas as pd
import pytest
from opx.gate import GateResult, go_nogo, main
from opx.walkforward import run_walk_forward

FAST = {"n_estimators": 80, "random_state": 42}


@pytest.fixture(scope="module")
def planted_oos(bars_factory):
    bars = bars_factory(30000, phi=-0.5)
    return run_walk_forward(bars, k=1, r=0.85, params=FAST)


@pytest.fixture(scope="module")
def noise_oos(bars_factory):
    bars = bars_factory(30000, phi=0.0)
    return run_walk_forward(bars, k=1, r=0.85, params=FAST)


def test_planted_signal_passes_all_checks(planted_oos):
    g = go_nogo(planted_oos, 0.85, buffers=[0.03], oos_stress=planted_oos, min_quarters=1)
    assert g.go, g.checks


def test_noise_fails(noise_oos):
    g = go_nogo(noise_oos, 0.85, buffers=[0.03], oos_stress=noise_oos, min_quarters=1)
    assert not g.go and not g.checks["lcb_vs_breakeven"]


def test_default_min_quarters_blocks_short_history(planted_oos):
    g = go_nogo(planted_oos, 0.85, buffers=[0.03], oos_stress=planted_oos)
    assert not g.checks["quarters"] and not g.go


def test_missing_stress_run_fails_lag_check(planted_oos):
    assert not go_nogo(planted_oos, 0.85, buffers=[0.03], min_quarters=1).checks["lag_stress"]


def test_multiple_buffers_compute_pbo(planted_oos):
    g = go_nogo(planted_oos, 0.85, buffers=[0.0, 0.02, 0.04, 0.06], oos_stress=planted_oos, min_quarters=1)
    assert 0.0 <= g.details["pbo"] <= 1.0
    assert g.checks["pbo"] == (g.details["pbo"] <= 0.2)


def test_multiple_buffers_with_too_few_days_fail_pbo(planted_oos):
    short = planted_oos.iloc[:3000]
    g = go_nogo(short, 0.85, buffers=[0.0, 0.02], min_quarters=1)
    assert not g.checks["pbo"]


def test_empty_oos_is_no_go_not_a_crash():
    empty = pd.DataFrame(columns=["time", "side", "p_cal", "y", "t1"])
    assert not go_nogo(empty, 0.85, buffers=[0.03]).go


def test_cli_exit_codes(tmp_path, bars_factory):
    path = tmp_path / "noise.parquet"
    bars = bars_factory(6000, phi=0.0)
    # Loader requires timestamp, open, high, low, close, volume
    df = bars.reset_index().rename(columns={"index": "timestamp", "tick_volume": "volume"})
    df.to_parquet(path)
    assert main(["--bars", str(path), "--k", "1", "--r", "0.85", "--buffers", "0.03"]) == 1
