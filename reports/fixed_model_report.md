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
| Alberta Express Entry | Linear | N/A - Insufficient data (3 events, need 8+) | 0 | 3 |
| Alberta Express Entry | RandomForest | N/A - Insufficient data (3 events, need 8+) | 0 | 3 |
| Alberta Express Entry | GradientBoosting | N/A - Insufficient data (3 events, need 8+) | 0 | 3 |
| Alberta Express Entry Stream | Linear | 318.86 | 21 | 28 |
| Alberta Express Entry Stream | RandomForest | 58.51 | 21 | 28 |
| Alberta Express Entry Stream | GradientBoosting | 68.94 | 21 | 28 |
| Alberta Opportunity Stream | Linear | 770.79 | 4 | 12 |
| Alberta Opportunity Stream | RandomForest | 182.52 | 4 | 12 |
| Alberta Opportunity Stream | GradientBoosting | 129.6 | 4 | 12 |
| Dedicated Health Care Pathway | Linear | 84.4 | 11 | 18 |
| Dedicated Health Care Pathway | RandomForest | 21.31 | 11 | 18 |
| Dedicated Health Care Pathway | GradientBoosting | 22.71 | 11 | 18 |
| Tourism and Hospitality Stream | Linear | N/A - Insufficient data (3 events, need 8+) | 0 | 3 |
| Tourism and Hospitality Stream | RandomForest | N/A - Insufficient data (3 events, need 8+) | 0 | 3 |
| Tourism and Hospitality Stream | GradientBoosting | N/A - Insufficient data (3 events, need 8+) | 0 | 3 |

## Best Models

- **Alberta Express Entry Stream**: RandomForest
- **Alberta Opportunity Stream**: GradientBoosting
- **Dedicated Health Care Pathway**: RandomForest

## Forecasts (with Block Bootstrap 95% CI)

| Stream | Date | Predicted | 95% CI Lower | 95% CI Upper | Model | Holiday |
| --- | --- | --- | --- | --- | --- | --- |
| Alberta Express Entry Stream | 2025-11-20 | 91 | 49 | 244 | RandomForest |  |
| Alberta Express Entry Stream | 2025-12-02 | 85 | 44 | 225 | RandomForest |  |
| Alberta Express Entry Stream | 2025-12-20 | 89 | 48 | 252 | RandomForest |  |
| Alberta Opportunity Stream | 2025-12-02 | 902 | 578 | 1002 | GradientBoosting |  |
| Alberta Opportunity Stream | 2026-01-15 | 559 | 408 | 907 | GradientBoosting |  |
| Alberta Opportunity Stream | 2026-03-22 | 559 | 407 | 906 | GradientBoosting |  |
| Dedicated Health Care Pathway | 2025-10-19 | 55 | 24 | 93 | RandomForest |  |
| Dedicated Health Care Pathway | 2025-11-06 | 58 | 31 | 94 | RandomForest |  |
| Dedicated Health Care Pathway | 2025-12-03 | 56 | 33 | 93 | RandomForest |  |
