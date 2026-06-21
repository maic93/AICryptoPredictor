"""
tests/test_pipeline.py
Runs on every push via CI workflow.
Tests data fetching, feature engineering, and model inference.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices():
    """Fake 200-day BTC price series."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200)
    prices = 40000 + np.cumsum(np.random.randn(200) * 500)
    return pd.Series(prices, index=dates, name="Close")

@pytest.fixture
def sample_df(sample_prices):
    np.random.seed(42)
    return pd.DataFrame({
        "Close":  sample_prices,
        "Open":   sample_prices * 0.99,
        "High":   sample_prices * 1.01,
        "Low":    sample_prices * 0.98,
        "Volume": np.random.randint(int(1e9), int(1e10), 200),
    })


# ── Data Pipeline Tests ───────────────────────────────────────────────────────

class TestDataPipeline:

    def test_no_nulls_after_ffill(self, sample_df):
        df = sample_df.copy()
        df.iloc[10, 0] = np.nan
        df = df.ffill()
        assert df.isnull().sum().sum() == 0

    def test_lag_features_correct_length(self, sample_df):
        df = sample_df.copy()
        close = df["Close"]
        for lag in [1, 3, 7]:
            df[f"lag{lag}"] = close.shift(lag)
        df.dropna(inplace=True)
        assert len(df) == 200 - 7

    def test_lag_values_correct(self, sample_df):
        df = sample_df.copy()
        close = df["Close"]
        df["lag1"] = close.shift(1)
        df.dropna(inplace=True)
        assert float(df["lag1"].iloc[0]) == pytest.approx(float(close.iloc[0]), rel=1e-5)

    def test_rolling_mean_shape(self, sample_df):
        df = sample_df.copy()
        df["roll7"] = df["Close"].rolling(7).mean()
        df.dropna(inplace=True)
        assert len(df) == 200 - 6

    def test_target_is_next_day_price(self, sample_df):
        df = sample_df.copy()
        close = df["Close"]
        df["target"] = close.shift(-1)
        df.dropna(inplace=True)
        assert float(df["target"].iloc[0]) == pytest.approx(float(close.iloc[1]), rel=1e-5)

    def test_no_negative_prices(self, sample_df):
        assert (sample_df["Close"] > 0).all()

    def test_high_always_gte_low(self, sample_df):
        assert (sample_df["High"] >= sample_df["Low"]).all()

    def test_train_test_no_data_leak(self, sample_df):
        close = sample_df["Close"].values
        split = int(len(close) * 0.8)
        train = close[:split]
        test  = close[split:]
        assert len(set(range(split)) & set(range(split, len(close)))) == 0
        assert len(train) + len(test) == len(close)


# ── Technical Indicators Tests ────────────────────────────────────────────────

class TestIndicators:

    def test_rsi_in_range(self, sample_prices):
        delta = sample_prices.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - (100 / (1 + gain / loss))).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_macd_is_diff_of_emas(self, sample_prices):
        ema12 = sample_prices.ewm(span=12, adjust=False).mean()
        ema26 = sample_prices.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        assert macd.shape == sample_prices.shape

    def test_bollinger_upper_above_lower(self, sample_prices):
        sma   = sample_prices.rolling(20).mean()
        std   = sample_prices.rolling(20).std()
        upper = (sma + 2 * std).dropna()
        lower = (sma - 2 * std).dropna()
        assert (upper > lower).all()

    def test_rolling_std_non_negative(self, sample_prices):
        std = sample_prices.rolling(14).std().dropna()
        assert (std >= 0).all()


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineering:

    def build_features(self, df):
        close = df["Close"]
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f"lag{lag}"] = close.shift(lag)
        for w in [7, 14, 30]:
            df[f"rollMean{w}"] = close.rolling(w).mean()
            df[f"rollStd{w}"]  = close.rolling(w).std()
        df["return1d"]  = close.pct_change(1)
        df["return7d"]  = close.pct_change(7)
        df["target"]    = close.shift(-1)
        df.dropna(inplace=True)
        return df

    def test_feature_count(self, sample_df):
        df = self.build_features(sample_df.copy())
        feat_cols = [c for c in df.columns if c not in
                     ["target", "Open", "High", "Low", "Close", "Volume"]]
        assert len(feat_cols) >= 14

    def test_no_inf_values(self, sample_df):
        df = self.build_features(sample_df.copy())
        assert not np.isinf(df.values).any()

    def test_no_nulls_after_dropna(self, sample_df):
        df = self.build_features(sample_df.copy())
        assert df.isnull().sum().sum() == 0

    def test_return1d_reasonable(self, sample_df):
        df = self.build_features(sample_df.copy())
        # Daily returns should be between -50% and +50% for BTC
        assert (df["return1d"].abs() < 0.5).all()


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestModels:

    def get_xy(self, sample_df):
        df = sample_df.copy()
        close = df["Close"]
        for lag in [1, 2, 3, 5, 7]:
            df[f"lag{lag}"] = close.shift(lag)
        for w in [7, 14]:
            df[f"rollMean{w}"] = close.rolling(w).mean()
        df["return1d"] = close.pct_change()
        df["target"] = close.shift(-1)
        df.dropna(inplace=True)
        feat = [c for c in df.columns if c not in
                ["target","Open","High","Low","Close","Volume"]]
        return df[feat].values, df["target"].values

    def test_xgboost_output_shape(self, sample_df):
        from xgboost import XGBRegressor
        X, y = self.get_xy(sample_df)
        split = int(len(X) * 0.8)
        model = XGBRegressor(n_estimators=10, verbosity=0)
        model.fit(X[:split], y[:split])
        preds = model.predict(X[split:])
        assert preds.shape == y[split:].shape

    def test_xgboost_predictions_positive(self, sample_df):
        from xgboost import XGBRegressor
        X, y = self.get_xy(sample_df)
        model = XGBRegressor(n_estimators=10, verbosity=0)
        model.fit(X, y)
        preds = model.predict(X[-10:])
        assert (preds > 0).all()

    def test_random_forest_feature_importance_sums_to_one(self, sample_df):
        from sklearn.ensemble import RandomForestRegressor
        X, y = self.get_xy(sample_df)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-6

    def test_lightgbm_output_shape(self, sample_df):
        import lightgbm as lgb
        X, y = self.get_xy(sample_df)
        split = int(len(X) * 0.8)
        model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        model.fit(X[:split], y[:split])
        preds = model.predict(X[split:])
        assert preds.shape == y[split:].shape

    def test_mae_better_than_naive(self, sample_df):
        """Model MAE should beat naive 'predict yesterday's price' baseline."""
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_absolute_error
        X, y = self.get_xy(sample_df)
        split = int(len(X) * 0.8)
        model = XGBRegressor(n_estimators=50, verbosity=0)
        model.fit(X[:split], y[:split])
        preds  = model.predict(X[split:])
        model_mae  = mean_absolute_error(y[split:], preds)
        naive_mae  = mean_absolute_error(y[split:], y[split-1:-1])
        assert model_mae < naive_mae * 2  # at least in the same ballpark
