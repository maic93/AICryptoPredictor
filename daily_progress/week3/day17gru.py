"""
Day 17 — GRU Model
Week 3: Deep Learning
"""
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def load_sequences(ticker="BTC-USD", lookback=60):
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
    return seq(scaled[:split]), seq(scaled[split - lookback:]), scaler

def run():
    print("⚡ Day 17 — GRU Model (Gated Recurrent Unit)")
    print("=" * 50)
    print("  GRU is faster than LSTM with comparable accuracy.")
    print("  Uses 2 gates (reset, update) vs LSTM's 3 gates.\n")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        (X_train, y_train), (X_test, y_test), scaler = load_sequences()

        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.1, callbacks=[es], verbose=1)

        preds = scaler.inverse_transform(model.predict(X_test))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        print(f"\n  GRU MAE : ${mean_absolute_error(actual, preds):,.2f}")
        model.save("models/gru_btc.h5")
        print("  💾 Saved → models/gru_btc.h5")

    except ImportError:
        print("  ⚠️  TensorFlow not installed.")
        print("  📋 Architecture: GRU(64) → Dropout → GRU(32) → Dropout → Dense(1)")

    print("\n✅ Day 17 complete!")

if __name__ == "__main__":
    run()
