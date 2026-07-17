"""
Day 28 — Final Summary
Week 4: Production
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from datetime import date

os.makedirs("reports", exist_ok=True)

def plot_summary():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flatten():
        ax.set_facecolor("#161b22")

    # Week progress bars
    weeks  = ["Week 1\nFoundations", "Week 2\nClassical ML",
               "Week 3\nDeep Learning", "Week 4\nProduction"]
    colors = ["#F7931A", "#3498DB", "#9B59B6", "#2ECC71"]
    bars = axes[0,0].barh(weeks, [100,100,100,100], color=colors, alpha=0.9)
    axes[0,0].set_xlim(0, 115)
    axes[0,0].set_title("Project Completion", color="white", fontweight="bold")
    axes[0,0].tick_params(colors="white")
    for bar in bars:
        axes[0,0].text(102, bar.get_y() + bar.get_height()/2,
                       "100%", va="center", color="white", fontsize=10)
    for sp in axes[0,0].spines.values():
        sp.set_color("#30363d")
    axes[0,0].set_facecolor("#161b22")

    # Model R2 scores
    ax = axes[0,1]
    names  = ["LinReg","DecTree","RandForest","XGBoost","LightGBM",
               "LSTM","GRU","BiLSTM","Attention","Transform"]
    scores = [0.94, 0.78, 0.92, 0.95, 0.94, 0.94, 0.93, 0.95, 0.94, 0.93]
    bcolors = ["#3498DB"]*5 + ["#9B59B6"]*5
    ax.bar(names, scores, color=bcolors, alpha=0.9)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Model R2 Scores", color="white", fontweight="bold")
    ax.tick_params(colors="white", axis="both", labelsize=7)
    ax.set_facecolor("#161b22")
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    classic = mpatches.Patch(color="#3498DB", label="Classical ML")
    dl      = mpatches.Patch(color="#9B59B6", label="Deep Learning")
    ax.legend(handles=[classic, dl], fontsize=8,
              facecolor="#0d1117", labelcolor="white")

    # 28-day completion grid
    ax = axes[1,0]
    data = np.ones((4, 7))
    ax.imshow(data, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"D{d}" for d in range(1,8)], fontsize=8, color="white")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Week 1","Week 2","Week 3","Week 4"], color="white")
    for i in range(4):
        for j in range(7):
            ax.text(j, i, "DONE", ha="center", va="center",
                    fontsize=7, color="darkgreen", fontweight="bold")
    ax.set_title("28-Day Completion Grid", color="white", fontweight="bold")
    ax.set_facecolor("#161b22")

    # Final message — ASCII only
    ax = axes[1,1]
    ax.axis("off")
    ax.set_facecolor("#161b22")
    msg = ("PROJECT COMPLETE!\n\n"
           "28 Days  |  4 Weeks\n"
           "10 Models Built\n"
           "28 Reports Generated\n"
           "28 Auto-Commits\n\n"
           f"Finished: {date.today()}\n\n"
           "github.com/maic93/AICryptoPredictor\n\n"
           "Not financial advice.")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", linespacing=2.0,
            transform=ax.transAxes)

    plt.suptitle("AICryptoPredictor - Final Summary (28 Days Complete)",
                  fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()
    out = "reports/day28chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 28 - AICryptoPredictor Complete!")
    print("=" * 60)
    print(f"\n  Finished: {date.today()}")
    print("""
  ============================================
  =                                          =
  =   AICryptoPredictor                     =
  =   28 Days | 4 Weeks | 100% Complete     =
  =                                          =
  ============================================

  Week 1 - Foundations        [DONE]
  Week 2 - Classical ML       [DONE]
  Week 3 - Deep Learning      [DONE]
  Week 4 - Production         [DONE]

  Star the repo if this helped you!
  github.com/maic93/AICryptoPredictor

  Educational only. Not financial advice.
    """)
    plot_summary()
    print("\nProject complete! Congratulations!")

if __name__ == "__main__":
    run()
