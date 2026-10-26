import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from opx.model import fit_model, predict_win_prob, fit_calibrator, calibrate


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(42)
    N_tr = 20000
    N_te = 10000
    N_total = N_tr + N_te

    X = rng.normal(0, 1, (N_total, 5))
    prob = expit(1.2 * X[:, 0])
    y = (rng.random(N_total) < prob).astype(float)
    ynoise = (rng.random(N_total) < 0.5).astype(float)

    Xtr, Xte = X[:N_tr], X[N_tr:]
    ytr, yte = y[:N_tr], y[N_tr:]
    ynoise_tr, ynoise_te = ynoise[:N_tr], ynoise[N_tr:]

    # For validation / calibration split:
    # use last 5000 rows of train as val
    X_train_sub = Xtr[:15000]
    y_train_sub = ytr[:15000]
    X_val = Xtr[15000:]
    y_val = ytr[15000:]

    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "Xte": Xte,
        "yte": yte,
        "ynoise_tr": ynoise_tr,
        "ynoise_te": ynoise_te,
        "X_train_sub": X_train_sub,
        "y_train_sub": y_train_sub,
        "X_val": X_val,
        "y_val": y_val,
    }


def test_learns_planted_signal(synthetic_data):
    d = synthetic_data
    m = fit_model(d["Xtr"], d["ytr"], r=0.85, params={"n_estimators": 100})
    p = predict_win_prob(m, d["Xte"], r=0.85)
    assert roc_auc_score(d["yte"], p) > 0.65


def test_calibrated_mean_matches_base_rate(synthetic_data):
    d = synthetic_data
    # Fit model on train subset
    m = fit_model(d["X_train_sub"], d["y_train_sub"], r=0.85, params={"n_estimators": 100})
    p_val = predict_win_prob(m, d["X_val"], r=0.85)
    p_test = predict_win_prob(m, d["Xte"], r=0.85)

    cal = fit_calibrator(p_val, d["y_val"])
    p_cal = calibrate(cal, p_test)
    assert abs(p_cal.mean() - d["yte"].mean()) < 0.03


def test_noise_gives_no_confident_predictions(synthetic_data):
    d = synthetic_data
    m = fit_model(d["Xtr"], d["ynoise_tr"], r=0.85, params={"n_estimators": 100})
    p = predict_win_prob(m, d["Xte"], r=0.85)
    assert abs(p.mean() - d["ynoise_te"].mean()) < 0.05
