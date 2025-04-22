# From Forecasts to Fulfillment: Evaluating ARIMA and LGBM in a Simulation-Based Inventory Framework

#### University of Michigan, Industrial and Operation Engineering
#### Contributors: Minxuan Chen | Yiyang Shi | Yushan Yao | Qianxi Zhang

---

## Project Overview

This project benchmarks two forecasting models—**ARIMA** and **LightGBM (LGBM)**—in a simulation-based inventory control environment.

We evaluate forecast performance **not just by accuracy (RMSE/NRMSE)**, but by how well the forecasts support **inventory decisions**, using:
- Efficiency curves (Lost Sales vs. Mean Inventory)
- Financial cost curves (Total Inventory Cost vs. Service Level)
- Realized vs. Target Service Level alignment

---

## Methodology Summary

### 1. Forecasting Models
- **ARIMA**: Classical time-series model for baseline comparison.
- **LightGBM**: Tree-based gradient boosting method using engineered features (lags, rolling means, calendar variables).

### 2. Forecast Evaluation
- **Accuracy**: RMSE & NRMSE on a rolling-window forecast.
- **Inventory Simulation**: Forecasts are fed into a periodic review inventory simulation with lead time.

### 3. Performance Metrics
- Mean Inventory
- Lost Sales
- Realized Service Level (RSL)
- Total Inventory Cost (Holding, Lost Sales, Ordering)