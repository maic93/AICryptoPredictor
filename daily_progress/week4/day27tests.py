"""
Day 27 — Unit Tests for Data Pipeline & Model Inference
Week 4: Production
"""
import unittest
import numpy as np
import pandas as pd

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2023-01-01", periods=200)
        self.close = pd.Series(np.random.uniform(20000, 70000, 200), index=dates, name="Close")
        self.df = pd.DataFrame({"Close": self.close, "Volume": np.random.randint(1e9, 1e10, 200)})

    def test_no_nulls_after_ffill(self):
        df = self.df.copy()
        df.iloc[5, 0] = np.nan
        df = df.ffill()
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_lag_feature_shape(self):
        df = self.df.copy()
        close = df["Close"]
        for lag in [1, 3, 7]:
            df[f"lag{lag}"] = close.shift(lag)
        df.dropna(inplace=True)
        self.assertEqual(len(df), 200 - 7)

    def test_rsi_range(self):
        close = self.close
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + gain / loss))).dropna()
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())

    def test_target_is_next_day(self):
        df = self.df.copy()
        close = df["Close"]
        df["target"] = close.shift(-1)
        df.dropna(inplace=True)
        self.assertAlmostEqual(float(df["target"].iloc[0]), float(close.iloc[1]), places=2)

    def test_train_test_no_leak(self):
        X = np.arange(100).reshape(100, 1)
        split = 80
        X_train, X_test = X[:split], X[split:]
        self.assertFalse(any(x in X_test for x in X_train))


class TestModelInference(unittest.TestCase):
    def test_xgboost_prediction_shape(self):
        from xgboost import XGBRegressor
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test  = np.random.rand(20, 10)
        model = XGBRegressor(n_estimators=10, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        self.assertEqual(preds.shape, (20,))

    def test_prediction_is_positive(self):
        from xgboost import XGBRegressor
        X_train = np.random.rand(100, 5) * 50000
        y_train = np.random.rand(100) * 50000 + 10000
        model = XGBRegressor(n_estimators=10, verbosity=0)
        model.fit(X_train, y_train)
        pred = model.predict(X_train[-1:])[0]
        self.assertGreater(pred, 0)


def run():
    print("✅ Day 27 — Unit Tests")
    print("=" * 50)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestModelInference))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\n  {passed}/{result.testsRun} tests passed")
    print("\n✅ Day 27 complete!")

if __name__ == "__main__":
    run()
