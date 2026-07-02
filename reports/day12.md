# Day 12 Report

**Task:** Day 12 — LightGBM + Cross-Validation
**Script:** `daily_progress/week2/day12lightgbm.py`
**Date:** 2026-07-02 07:25 UTC (re-run, bug fixed)

## Output

```
💡 Day 12 — LightGBM + Cross-Validation
==================================================
  Fold 1: MAE=$ 8,676.34 | RMSE=$ 9,513.69 | R²=-1.0499
  Fold 2: MAE=$ 4,395.92 | RMSE=$ 6,051.88 | R²=0.7000
  Fold 3: MAE=$ 3,350.62 | RMSE=$ 4,156.50 | R²=0.3190
  Fold 4: MAE=$ 2,808.66 | RMSE=$ 3,841.93 | R²=0.8620
  Fold 5: MAE=$ 2,359.61 | RMSE=$ 2,892.43 | R²=0.7995

  Mean MAE  : $4,318.23 ± $2,282.56
  Mean RMSE : $5,291.29 ± $2,347.34
  Mean R²   : 0.3261

  ℹ️  Note: Fold 1 lower R² is expected — time series CV trains
  on less data in early folds. Performance improves as data grows.
  📈 Chart saved → reports/day12chart.png

✅ LightGBM complete!
```
