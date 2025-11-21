# AAIP Fixed Modeling Report

**Fixes Applied**:
1. ✅ cumulative_invitations: shift(1) to avoid target leakage
2. ✅ median_gap: computed within rolling window
3. ✅ Rolling validation: features rebuilt in each slice
4. ✅ Block Bootstrap: preserves temporal structure (block_size=5)
5. ✅ Random seed: np.random.seed(42) for reproducibility
6. ✅ Insufficient data warnings: explicit in results

## Model Performance (Rolling Validation MAE)

| Stream | Model | MAE | Test Size | Train Events |
| --- | --- | --- | --- | --- |
| Alberta Express Entry | Linear | N/A - Insufficient data (3 events, need 12+) | 0 | 3 |
| Alberta Express Entry | RandomForest | N/A - Insufficient data (3 events, need 12+) | 0 | 3 |
| Alberta Express Entry | GradientBoosting | N/A - Insufficient data (3 events, need 12+) | 0 | 3 |
| Alberta Express Entry Stream | Linear | 387.17 | 17 | 28 |
| Alberta Express Entry Stream | RandomForest | 69.46 | 17 | 28 |
| Alberta Express Entry Stream | GradientBoosting | 78.42 | 17 | 28 |
| Alberta Opportunity Stream | Linear | N/A - Insufficient data (12 events, need 12+) | 0 | 12 |
| Alberta Opportunity Stream | RandomForest | N/A - Insufficient data (12 events, need 12+) | 0 | 12 |
| Alberta Opportunity Stream | GradientBoosting | N/A - Insufficient data (12 events, need 12+) | 0 | 12 |
| Dedicated Health Care Pathway | Linear | 126.83 | 7 | 18 |
| Dedicated Health Care Pathway | RandomForest | 32.55 | 7 | 18 |
| Dedicated Health Care Pathway | GradientBoosting | 33.59 | 7 | 18 |
| Tourism and Hospitality Stream | Linear | N/A - Insufficient data (3 events, need 12+) | 0 | 3 |
| Tourism and Hospitality Stream | RandomForest | N/A - Insufficient data (3 events, need 12+) | 0 | 3 |
| Tourism and Hospitality Stream | GradientBoosting | N/A - Insufficient data (3 events, need 12+) | 0 | 3 |

## Best Models

- **Alberta Express Entry Stream**: RandomForest
- **Dedicated Health Care Pathway**: RandomForest

## Forecasts (with Block Bootstrap 95% CI)

| Stream | Date | Predicted | 95% CI Lower | 95% CI Upper | Model | Holiday |
| --- | --- | --- | --- | --- | --- | --- |
| Alberta Express Entry Stream | 2025-11-20 | 91 | 53 | 231 | RandomForest |  |
| Alberta Express Entry Stream | 2025-12-02 | 85 | 46 | 217 | RandomForest |  |
| Alberta Express Entry Stream | 2025-12-20 | 89 | 47 | 240 | RandomForest |  |
| Dedicated Health Care Pathway | 2025-10-19 | 55 | 27 | 92 | RandomForest |  |
| Dedicated Health Care Pathway | 2025-11-06 | 58 | 32 | 93 | RandomForest |  |
| Dedicated Health Care Pathway | 2025-12-03 | 56 | 33 | 106 | RandomForest |  |
