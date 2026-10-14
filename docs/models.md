# TerraPulse AI — Models & Machine Learning Benchmarks

## 1. Seasonal Naive Baseline Forecaster
For MODIS 16-day composite data, one complete tropical/annual year contains approximately 23 observations ($365.25 / 16 \approx 22.825$ periods). The Seasonal Naive baseline repeats the corresponding observation from the previous annual cycle ($t - 23$).

- **Equation**: $\hat{y}_{t+h} = y_{t+h - 23}$
- **Role**: Essential benchmark that any complex machine learning model must convincingly surpass.

## 2. Random Forest Regressor Forecaster
Trained on leak-free lag and temporal features using strict chronological time splits.

### Feature Specification:
- **Temporal Cyclical**: $\sin(2\pi \cdot \text{DOY} / 365.25)$, $\cos(2\pi \cdot \text{DOY} / 365.25)$, $\sin(2\pi \cdot k / 23)$, $\cos(2\pi \cdot k / 23)$
- **Autoregressive Lags**: $y_{t-1}, y_{t-2}, y_{t-3}, y_{t-23}$
- **Rolling Statistics**: Rolling mean over windows 3 and 6, with `shift=1` to guarantee zero contemporaneous or look-ahead leakage.
- **Cross-Variable Predictors**: Coincident LST lag $LST_{t-1}$.

### Hyperparameters:
- `n_estimators`: 150
- `max_depth`: 8
- `random_state`: 42
- `n_jobs`: -1

## 3. Evaluation on Chronological Held-Out Test Set
- **Training Set**: 216 observations (2014-12-19 to 2024-04-22, 80%)
- **Test Set**: 55 observations (2024-05-08 to 2026-09-14, 20%)

| Model | MAE | RMSE | R² Score | Note |
|---|---|---|---|---|
| **Random Forest Regressor** | **0.0048** | **0.0065** | **0.9323** | **+65.8% Error Reduction** |
| **Seasonal Naive Baseline** | 0.0118 | 0.0190 | 0.4177 | Reference benchmark |

## 4. Deep Learning Transformer Decision (Stage 9.4)
The roadmap mandates establishing whether a deep Transformer is justified before launching compute.
- **Decision**: Deep Transformer is **NOT justified** for this task.
- **Scientific Evidence**: With 271 composite steps in the national historical sequence, the Random Forest model achieves near-perfect fidelity ($R^2 = 0.9323$, $\text{RMSE} = 0.0065$). Training an attention-based Transformer with tens of thousands of parameters on a 271-sample series carries high overfitting risk and unnecessary GPU expenditure without tangible generalization benefits.
