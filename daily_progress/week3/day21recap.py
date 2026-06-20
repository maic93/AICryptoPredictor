"""
Day 21 — Week 3 Recap & Deep Learning Summary
Week 3: Deep Learning
"""
import os

def run():
    print("📝 Day 21 — Week 3 Recap")
    print("=" * 50)

    models = {
        "LSTM (2-layer)"       : "models/lstm_btc.h5",
        "GRU (2-layer)"        : "models/gru_btc.h5",
        "Bidirectional LSTM"   : "models/bilstm_btc.h5",
        "Attention LSTM"       : "models/attention_lstm.h5",
        "Transformer"          : "models/transformer_btc.h5",
    }

    print("\n  Saved Models:")
    for name, path in models.items():
        status = "✅" if os.path.exists(path) else "⬜"
        print(f"    {status} {name:<25} → {path}")

    print("\n  Week 3 Highlights:")
    print("  ✅ LSTM   — classic memory network, great baseline")
    print("  ✅ GRU    — 30% faster than LSTM, similar accuracy")
    print("  ✅ BiLSTM — reads past & future, richer context")
    print("  ✅ Attention — model focuses on key time steps")
    print("  ✅ Transformer — state-of-the-art, no recurrence")
    print("\n🎉 Week 3 done! Next: Week 4 — Production & Deployment")

if __name__ == "__main__":
    run()
