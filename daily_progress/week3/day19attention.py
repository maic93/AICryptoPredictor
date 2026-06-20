"""
Day 19 — Attention Mechanism on LSTM
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
    print("👁️  Day 19 — Attention Mechanism on LSTM")
    print("=" * 50)
    print("  Attention lets the model 'focus' on the most")
    print("  relevant timesteps when making predictions.\n")

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                             Permute, Multiply, Lambda, Flatten)
        from tensorflow.keras.callbacks import EarlyStopping

        (X_train, y_train), (X_test, y_test), scaler = load_sequences()

        # Simple attention block
        inputs = Input(shape=(60, 1))
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        # Attention weights
        attention = Dense(1, activation="tanh")(lstm_out)
        attention = Flatten()(attention)
        attention = tf.keras.layers.Activation("softmax")(attention)
        attention = tf.keras.layers.RepeatVector(64)(attention)
        attention = Permute([2, 1])(attention)
        context = Multiply()([lstm_out, attention])
        context = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
        dropout = Dropout(0.2)(context)
        output = Dense(1)(dropout)

        model = Model(inputs, output)
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=1)

        preds = scaler.inverse_transform(model.predict(X_test))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        print(f"\n  Attention-LSTM MAE : ${mean_absolute_error(actual, preds):,.2f}")
        model.save("models/attention_lstm.h5")
        print("  💾 Saved → models/attention_lstm.h5")

    except ImportError:
        print("  ⚠️  TensorFlow not installed.")

    print("\n✅ Day 19 complete!")

if __name__ == "__main__":
    run()
