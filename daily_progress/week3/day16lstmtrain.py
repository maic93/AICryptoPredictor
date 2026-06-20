"""
Day 16 — LSTM Model Training on BTC
Week 3: Deep Learning
"""
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def load_and_prepare(ticker="BTC-USD", lookback=60):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    prices = df["Close"].squeeze().dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    split = int(len(scaled) * 0.8)
    def seq(data):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X).reshape(-1, lookback, 1), np.array(y)
    X_train, y_train = seq(scaled[:split])
    X_test,  y_test  = seq(scaled[split - lookback:])
    return X_train, y_train, X_test, y_test, scaler, prices

def run():
    print("🔁 Day 16 — LSTM Training on BTC")
    print("=" * 50)
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        X_train, y_train, X_test, y_test, scaler, prices = load_and_prepare()

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.summary()

        es = EarlyStopping(patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                            validation_split=0.1, callbacks=[es], verbose=1)

        preds_scaled = model.predict(X_test)
        preds = scaler.inverse_transform(preds_scaled)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        mae = mean_absolute_error(actual, preds)
        print(f"\n  LSTM MAE : ${mae:,.2f}")
        model.save("models/lstm_btc.h5")
        print("  💾 Model saved → models/lstm_btc.h5")

    except ImportError:
        print("  ⚠️  TensorFlow not installed. Run: pip install tensorflow")
        print("  📋 Architecture that would be used:")
        print("     LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)")

    print("\n✅ Day 16 complete!")

if __name__ == "__main__":
    run()
