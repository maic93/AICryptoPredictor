# Day 12 Report

**Task:** Day 12 — LightGBM + cross-validation
**Script:** `daily_progress/week2/day12lightgbm.py`
**Date:** 2026-07-01 10:38 UTC

## Output

```
💡 Day 12 — LightGBM + Cross-Validation
==================================================
  Fold 1: MAE=$ 8,723.62 | RMSE=$ 9,453.73 | R²=-1.0462
  Fold 2: MAE=$ 4,358.85 | RMSE=$ 5,966.17 | R²=0.7054
  Fold 3: MAE=$ 3,307.92 | RMSE=$ 4,066.86 | R²=0.3199
  Fold 4: MAE=$ 2,456.43 | RMSE=$ 3,414.22 | R²=0.8912
  Fold 5: MAE=$ 2,250.71 | RMSE=$ 2,737.71 | R²=0.8180

  Mean MAE  : $4,219.51 ± $2,371.58
  Mean RMSE : $5,127.74 ± $2,416.24
  Mean R²   : 0.3377
  📈 Chart saved → reports/day12chart.png

✅ LightGBM complete!

STDERR:
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
  warnings.warn(
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
  warnings.warn(
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
  warnings.warn(
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
  warnings.warn(
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/sklearn/utils/validation.py:2827: UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
  warnings.warn(
```
