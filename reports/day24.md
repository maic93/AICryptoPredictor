# Day 24 Report

**Task:** Day 24 — Ensemble model (ML + DL combined)
**Script:** `daily_progress/week4/day24ensemble.py`
**Date:** 2026-07-13 10:18 UTC

## Output

```
Day 24 — Ensemble Model
==================================================
  RandomForest    MAE=$2,166.68
  XGBoost         MAE=$2,115.83
  LightGBM        MAE=$1,809.25

         Ensemble MAE=$1,915.48  R2=0.8562 [BEST]

  Weights: {'RandomForest': '0.310', 'XGBoost': '0.318', 'LightGBM': '0.372'}
  📈 Chart saved → reports/day24chart.png

✅ Ensemble complete!
```
