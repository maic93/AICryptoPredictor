"""
Day 18 — Bidirectional LSTM
Week 3: Deep Learning
"""
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def load_sequences(lookback=60):
    df = yf.download("BTC-USD", period="2y", progress=False, auto_adjust=True)
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
    print("↔️  Day 18 — Bidirectional LSTM")
    print("=" * 50)
    print("  BiLSTM reads sequences in BOTH directions,")
    print("  capturing patterns from past and future context.\n")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        (X_train, y_train), (X_test, y_test), scaler = load_sequences()

        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(60, 1)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=1)

        preds = scaler.inverse_transform(model.predict(X_test))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        print(f"\n  BiLSTM MAE : ${mean_absolute_error(actual, preds):,.2f}")
        model.save("models/bilstm_btc.h5")
        print("  💾 Saved → models/bilstm_btc.h5")

    except ImportError:
        print("  ⚠️  TensorFlow not installed.")
        print("  📋 Architecture: BiLSTM(64) → Dropout → BiLSTM(32) → Dropout → Dense(1)")

    print("\n✅ Day 18 complete!")

if __name__ == "__main__":
    run()
