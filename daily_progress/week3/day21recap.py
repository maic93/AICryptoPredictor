"""
Day 21 — Week 3 Recap & Deep Learning Summary
Week 3: Deep Learning
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("reports", exist_ok=True)

MODELS = {
    "LSTM-window":   {"r2": 0.94, "mae": 820,  "color": "#9B59B6"},
    "GRU-style":     {"r2": 0.93, "mae": 870,  "color": "#E74C3C"},
    "BiLSTM-style":  {"r2": 0.95, "mae": 780,  "color": "#1ABC9C"},
    "Attention":     {"r2": 0.94, "mae": 800,  "color": "#E67E22"},
    "Transformer":   {"r2": 0.93, "mae": 850,  "color": "#8E44AD"},
}

def plot():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    names  = list(MODELS.keys())
    r2s    = [v["r2"]  for v in MODELS.values()]
    maes   = [v["mae"] for v in MODELS.values()]
    colors = [v["color"] for v in MODELS.values()]

    axes[0].barh(names, r2s, color=colors, alpha=0.85)
    axes[0].set_xlim(0.88, 1.0)
    axes[0].set_title("Week 3 Models - R2 Score", fontweight="bold")
    axes[0].set_xlabel("R2 (higher = better)")
    for i, v in enumerate(r2s):
        axes[0].text(v + 0.001, i, f"{v:.2f}", va="center", fontsize=9)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(names, maes, color=colors, alpha=0.85)
    axes[1].set_title("Week 3 Models - MAE (USD)", fontweight="bold")
    axes[1].set_xlabel("MAE in USD (lower = better)")
    for i, v in enumerate(maes):
        axes[1].text(v + 5, i, f"${v:,}", va="center", fontsize=9)
    axes[1].grid(alpha=0.3, axis="x")

    plt.suptitle("Day 21 - Week 3 Deep Learning Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day21chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("Day 21 - Week 3 Recap")
    print("=" * 50)
    print("\n  Models built this week:")
    print("  Day 15: Sequence preparation (60-day lookback)")
    print("  Day 16: LSTM-style sliding window model")
    print("  Day 17: GRU-style with decay gating")
    print("  Day 18: Bidirectional LSTM (forward + backward)")
    print("  Day 19: Attention mechanism (learned timestep weights)")
    print("  Day 20: Transformer self-attention")
    print("\n  All implemented in pure numpy/sklearn")
    print("  — no TensorFlow/PyTorch needed for CI/CD.")
    print("\n  Best performer: BiLSTM-style (R2=0.95)")
    plot()
    print("\n  Next: Week 4 — Backtesting, ensemble, live pipeline!")
    print("\n✅ Week 3 complete!")

if __name__ == "__main__":
    run()
