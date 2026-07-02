# Day 13 Report

**Task:** Day 13 — Model comparison leaderboard (MAE, RMSE, R²)
**Script:** `daily_progress/week2/day13comparison.py`
**Date:** 2026-07-02 09:56 UTC

## Output

```
🏆 Day 13 — Model Comparison Leaderboard
============================================================

  Rank  Model                         MAE         RMSE       R²
  ------------------------------------------------------------
  1     LinearRegression     $  1,193.98 $  1,505.99   0.9378 👑
  2     LightGBM             $  2,276.52 $  2,734.69   0.7948
  3     RandomForest         $  2,301.09 $  2,745.04   0.7933
  4     XGBoost              $  2,389.29 $  2,979.94   0.7564
  5     DecisionTree         $  2,484.34 $  3,527.76   0.6585
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
