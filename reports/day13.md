# Day 13 Report

**Task:** Day 13 — Model comparison leaderboard
**Script:** `daily_progress/week2/day13comparison.py`
**Date:** 2026-07-05 13:40 UTC *(re-run with bug fix)*

## Output

```
🏆 Day 13 — Model Comparison Leaderboard
============================================================

  Rank  Model                         MAE         RMSE       R²
  ------------------------------------------------------------
  1     LinearRegression     $  1,184.68 $  1,499.14   0.9399 👑
  2     LightGBM             $  2,053.91 $  2,516.04   0.8308
  3     RandomForest         $  2,239.75 $  2,733.84   0.8002
  4     XGBoost              $  2,373.13 $  2,985.43   0.7618
  5     DecisionTree         $  2,506.58 $  3,488.39   0.6747
  📈 Chart saved → reports/day13chart.png

✅ Best model: LinearRegression

STDERR:
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
  warnings.warn(
/home/runner/work/AICryptoPredictor/AICryptoPredictor/daily_progress/week2/day13comparison.py:58: UserWarning: Glyph 127942 (\N{TROPHY}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/home/runner/work/AICryptoPredictor/AICryptoPredictor/daily_progress/week2/day13comparison.py:60: UserWarning: Glyph 127942 (\N{TROPHY}) missing from font(s) DejaVu Sans.
  plt.savefig(out, dpi=150); plt.close()
```
