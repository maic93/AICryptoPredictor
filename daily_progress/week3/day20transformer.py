"""
Day 20 — Transformer-Based Time Series Model
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
    print("🤖 Day 20 — Transformer for Crypto Price Prediction")
    print("=" * 50)
    print("  Transformers use multi-head self-attention — no recurrence.")
    print("  Same architecture powering GPT and BERT.\n")

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                             MultiHeadAttention, GlobalAveragePooling1D)
        from tensorflow.keras.callbacks import EarlyStopping

        (X_train, y_train), (X_test, y_test), scaler = load_sequences()

        def transformer_block(x, heads=4, ff_dim=64, dropout=0.1):
            attn = MultiHeadAttention(num_heads=heads, key_dim=32)(x, x)
            attn = Dropout(dropout)(attn)
            x = LayerNormalization(epsilon=1e-6)(x + attn)
            ff = Dense(ff_dim, activation="relu")(x)
            ff = Dense(x.shape[-1])(ff)
            ff = Dropout(dropout)(ff)
            return LayerNormalization(epsilon=1e-6)(x + ff)

        inputs = Input(shape=(60, 1))
        x = Dense(32)(inputs)
        x = transformer_block(x)
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.1)(x)
        output = Dense(1)(x)

        model = Model(inputs, output)
        model.compile(optimizer="adam", loss="mse")
        model.summary()

        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=1)

        preds = scaler.inverse_transform(model.predict(X_test))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        print(f"\n  Transformer MAE : ${mean_absolute_error(actual, preds):,.2f}")
        model.save("models/transformer_btc.h5")
        print("  💾 Saved → models/transformer_btc.h5")

    except ImportError:
        print("  ⚠️  TensorFlow not installed.")

    print("\n✅ Day 20 complete!")

if __name__ == "__main__":
    run()
