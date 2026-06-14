"""Python-side tests covering the binding, including edge and error paths.

Run with `pytest regressor-python/tests` after `maturin develop`.
"""
import numpy as np
import pytest

from regressor.model.linear_model import (
    LinearRegression, Ridge, LogisticRegression, Metric, Penalty,
)
from regressor.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# --- LinearRegression -------------------------------------------------------

def test_intercept_fit_automatically():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = np.array([1 + 2 * a + 3 * b for a, b in X])
    m = LinearRegression()
    m.fit(X, y)
    intercept, coefs = m.params()
    assert intercept == pytest.approx(1.0)
    assert coefs == pytest.approx([2.0, 3.0])


def test_single_feature_recovers_intercept():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([4.0, 7.0, 10.0, 13.0])  # 4 + 3x
    m = LinearRegression()
    m.fit(X, y)
    intercept, coefs = m.params()
    assert intercept == pytest.approx(4.0)
    assert coefs[0] == pytest.approx(3.0)


def test_no_intercept_reports_zero():
    # Caller supplies an explicit ones column; fit_intercept disabled.
    X = np.array([[1.0, 2.0], [1.0, 1.0], [1.0, 4.0], [1.0, 3.0]])
    y = np.array([1 + 3 * row[1] for row in X])
    m = LinearRegression(fit_intercept=False)
    m.fit(X, y)
    intercept, coefs = m.params()
    assert intercept == 0.0
    assert coefs == pytest.approx([1.0, 3.0])


def test_params_before_fit_raises():
    with pytest.raises(ValueError):
        LinearRegression().params()


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        LinearRegression().predict([[1.0, 2.0]])


def test_score_size_mismatch_raises():
    m = LinearRegression()
    m.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        m.score([1.0, 2.0], [1.0], Metric.R2)


# --- metrics ----------------------------------------------------------------

def test_mse_and_rmse_are_squared():
    m = LinearRegression()
    m.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 5.0]  # one error of 2 -> MSE = 4/3
    assert m.score(y_true, y_pred, Metric.MSE) == pytest.approx(4.0 / 3.0)
    assert m.score(y_true, y_pred, Metric.RMSE) == pytest.approx((4.0 / 3.0) ** 0.5)


# --- Ridge ------------------------------------------------------------------

def test_ridge_shrinks_relative_to_ols():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x
    ols_model = LinearRegression()
    ols_model.fit(X, y)
    _, ols = ols_model.params()
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X, y)
    _, ridge = ridge_model.params()
    assert 0.0 < ridge[0] < ols[0]


# --- LogisticRegression -----------------------------------------------------

def test_logistic_separates_toy_data():
    X = np.array([[x] for x in [-2, -1, -0.5, 0.5, 1, 2]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    clf = LogisticRegression(0.5, 2000)
    clf.set_penalty(Penalty.l2(0.01))
    clf.fit(X, y)
    assert clf.score(y, clf.predict(X)) == 1.0


# --- preprocessing ----------------------------------------------------------

def test_standard_scaler_roundtrip_stats():
    raw = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    z = StandardScaler().fit_transform(raw)
    assert np.allclose(z.mean(axis=0), 0.0)
    assert np.allclose(z.std(axis=0, ddof=1), 1.0)


def test_minmax_scaler_range():
    raw = np.array([[1.0], [3.0], [5.0]])
    mm = MinMaxScaler().fit_transform(raw)
    assert mm.min() == pytest.approx(0.0)
    assert mm.max() == pytest.approx(1.0)


def test_scaler_transform_before_fit_raises():
    with pytest.raises(ValueError):
        StandardScaler().transform([[1.0]])


def test_label_encoder_roundtrip():
    enc = LabelEncoder()
    codes = enc.fit_transform(["b", "a", "c", "a"])
    assert codes == [1.0, 0.0, 2.0, 0.0]
    assert enc.classes() == ["a", "b", "c"]
    assert enc.inverse_transform(codes) == ["b", "a", "c", "a"]


def test_label_encoder_unknown_label_raises():
    enc = LabelEncoder()
    enc.fit(["a", "b"])
    with pytest.raises(ValueError):
        enc.transform(["z"])
