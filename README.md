# README — EEEM073 AI and Sustainability Coursework
## Explainable and Efficient AI for Time Series Forecasting of Wind and Solar Energy in the TenneT Region, Germany
### University of Surrey | Student: Anil Emanuel

---

## Project Overview

This project applies machine learning and deep learning techniques to forecast hourly wind and solar energy generation in the TenneT DE control zone in Germany. Three models are trained, evaluated with explainable AI techniques, and compressed to demonstrate both AI for sustainability (accurate renewable energy forecasting) and sustainable AI (lightweight, efficient model design).

The project directly supports:
- **SDG 7** — Affordable and Clean Energy
- **SDG 13** — Climate Action

---

## Notebook Pipeline

The code is split into 6 sequential notebooks that must be run in order. Each notebook saves its outputs for the next notebook to use.

| Notebook | File | Purpose |
|---|---|---|
| 1 | `notebook1_data_loading.ipynb` | Data loading, cleaning, DST handling, feature engineering, MinMaxScaler normalisation, chronological train/val/test split |
| 2 | `notebook2_eda.ipynb` | Exploratory data analysis — annual, monthly, hourly patterns, boxplots, correlation heatmap, scatter plots, summer vs winter comparison |
| 3 | `notebook3_timeseries.ipynb` | ACF, PACF, STL seasonal decomposition (period=168), ADF stationarity tests, rolling statistics |
| 4 | `notebook4_modelling.ipynb` | Train Linear Regression, MLP (128→64→32) and LSTM (128+64 stacked, LOOKBACK=24) models |
| 5 | `notebook5_evaluation.ipynb` | MAE/RMSE/R² evaluation, residual analysis, SHAP (LinearExplainer + KernelExplainer), LIME local explanations |
| 6 | `notebook6_compression.ipynb` | 8 compression experiments: Feature Selection, LASSO, Pruning, Pruning+Quantization, Knowledge Distillation |

---

## Data Sources

All datasets are fully open-source and freely available:

### 1. Wind and Solar Generation Data
- **Source:** SMARD — Bundesnetzagentur (German Federal Network Agency)
- **URL:** https://www.smard.de/home
- **Description:** Hourly wind onshore, wind offshore and solar PV generation data for the TenneT DE control zone, Germany
- **Coverage:** January 2021 to December 2024
- **File:** `Data/generation_data.csv`
- **Citation:** Bundesnetzagentur. (2024). *Electricity generation data — TenneT DE control zone*. SMARD. https://www.smard.de

### 2. Munich Weather Data
- **Source:** Open-Meteo Historical Weather API
- **URL:** https://open-meteo.com
- **Location:** Munich, Germany (48.1351°N, 11.5820°E)
- **Variables:** Temperature (2m), wind speed (10m and 100m), cloud cover, shortwave radiation
- **File:** `Data/munich_weather.csv.csv`
- **Citation:** Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7970649

### 3. Kiel Weather Data
- **Source:** Open-Meteo Historical Weather API
- **URL:** https://open-meteo.com
- **Location:** Kiel, Germany (54.3233°N, 10.1228°E)
- **Variables:** Temperature (2m), wind speed (10m and 100m), cloud cover, shortwave radiation
- **File:** `Data/kiel_weather.csv.csv`
- **Citation:** Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7970649

---

## How To Run The Code

### Step 1 — Set Up Environment

Install all required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap lime statsmodels joblib tensorflow-model-optimization tf_keras
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

**Note:** `tensorflow-model-optimization` and `tf_keras` are required for Notebook 6 (model compression). `lime` is required for Notebook 5 (LIME local explainability).

**Note for Windows users:** TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. The notebooks run on CPU only. For TFLite LSTM quantization (Notebook 6 Step 6.2), float16 quantization is used as an equivalent alternative due to TensorListReserve op constraints on Windows TF 2.11+.

### Step 2 — Prepare Data Files

Place the three data files in a folder called `Data/` in the same directory as the notebooks:

```
project_folder/
├── Data/
│   ├── generation_data.csv
│   ├── munich_weather.csv.csv
│   └── kiel_weather.csv.csv
├── notebook1_data_loading.ipynb
├── notebook2_eda.ipynb
├── notebook3_timeseries.ipynb
├── notebook4_modelling.ipynb
├── notebook5_evaluation.ipynb
├── notebook6_compression.ipynb
├── README.md
└── requirements.txt

```

### Step 3 — Run Notebooks In Order

Run each notebook from top to bottom **in sequence**:

1. Run `notebook1_data_loading.ipynb` — saves all processed data files
2. Run `notebook2_eda.ipynb` — generates EDA figures (Figures 2-8)
3. Run `notebook3_timeseries.ipynb` — generates time series analysis figures (Figures 9-13)
4. Run `notebook4_modelling.ipynb` — trains and saves all three models (Figures 14-16)
5. Run `notebook5_evaluation.ipynb` — evaluates models and generates SHAP/LIME plots (Figures 17-23)
6. Run `notebook6_compression.ipynb` — compresses models and generates compression chart (Figure 24)

**Important:** Each notebook must be fully run before the next one is started, as each notebook saves files that the next notebook loads.

**Approximate runtimes:**
- Notebook 1: ~1 minute
- Notebook 2: ~2 minutes
- Notebook 3: ~3 minutes
- Notebook 4: ~15 minutes (LSTM training takes ~11 minutes)
- Notebook 5: ~15 minutes (SHAP KernelExplainer is slow)
- Notebook 6: ~60 minutes (pruning + distillation training)

---

## Dataset Statistics

| Split | Rows | Date Range | Percentage |
|---|---|---|---|
| Full dataset | 34,896 | 2021-01-08 to 2024-12-31 | 100% |
| Training | 26,112 | 2021-01-08 to 2023-12-31 | 74.8% |
| Validation | 4,368 | 2024-01-01 to 2024-06-30 | 12.5% |
| Test | 4,416 | 2024-07-01 to 2024-12-31 | 12.7% |

**Features:** 34 total
- 9 lag features (t-1, t-24, t-168 × 3 targets)
- 6 rolling mean features (24h, 168h × 3 targets)
- 6 cyclic time features (hour, month, day_of_week sin/cos)
- 10 weather features (Munich + Kiel: temperature, wind speed 10m/100m, cloud cover, shortwave radiation)
- 3 raw time features (hour, month, day_of_week)

**4 DST gaps detected and fixed** (March 2021, 2022, 2023, 2024) using reindex + forward fill.

---

## Files Generated by the Pipeline

After running all notebooks, the following files will be saved in the project folder:

### Processed Data Files
| File | Description |
|---|---|
| `data_merged.csv` | Full merged dataset (unscaled, 34,896 rows, 39 columns) |
| `data_train.csv` | Training set (unscaled, 26,112 rows) |
| `data_val.csv` | Validation set (unscaled, 4,368 rows) |
| `data_test.csv` | Test set (unscaled, 4,416 rows) |
| `data_train_scaled.csv` | Training set (scaled) |
| `data_val_scaled.csv` | Validation set (scaled) |
| `data_test_scaled.csv` | Test set (scaled) |
| `X_train.npy` | Scaled feature array — training (26112, 34) |
| `X_val.npy` | Scaled feature array — validation (4368, 34) |
| `X_test.npy` | Scaled feature array — test (4416, 34) |
| `y_train.npy` | Scaled target array — training (26112, 3) |
| `y_val.npy` | Scaled target array — validation (4368, 3) |
| `y_test.npy` | Scaled target array — test (4416, 3) |
| `feature_scaler.pkl` | Fitted MinMaxScaler for 34 input features |
| `target_scaler.pkl` | Fitted MinMaxScaler for 3 target columns |
| `feature_columns.csv` | List of 34 feature column names |
| `target_columns.csv` | List of 3 target column names |

### Trained Models (Original)
| File | Description |
|---|---|
| `model_linear_regression.pkl` | Linear Regression — 1.64 KB |
| `model_mlp.keras` | MLP (128→64→32, Dropout 0.2) — 208.54 KB |
| `model_lstm.keras` | LSTM (128+64 stacked, LOOKBACK=24) — 1,623.37 KB |

### Compressed Models (Notebook 6)
| File | Description |
|---|---|
| `model_lr_feature_selection.pkl` | LR with SHAP top 10 features — 0.89 KB (45.7% reduction) |
| `model_lr_lasso.pkl` | LR with LASSO regularisation (α=0.001) — 1.51 KB |
| `model_mlp_pruned.keras` | MLP with 80% magnitude pruning — 210.75 KB |
| `model_mlp_pruned_quantized.tflite` | MLP pruned + TFLite int8 quantization — 20.59 KB (90.1% reduction) |
| `model_mlp_distilled.keras` | MLP knowledge distillation student (32→16) — 26.63 KB (87.2% reduction) |
| `model_lstm_pruned.keras` | LSTM with 80% magnitude pruning — 1,623.37 KB |
| `model_lstm_weights_f16.npy` | LSTM pruned + float16 quantization — 264.47 KB (83.7% reduction) |
| `model_lstm_distilled.keras` | LSTM knowledge distillation student LSTM(32) — 53.22 KB (96.7% reduction) |

### Prediction Files
| File | Description |
|---|---|
| `lr_pred.npy` | Linear Regression predictions in MWh (4416, 3) |
| `mlp_pred.npy` | MLP predictions in MWh (4416, 3) |
| `lstm_pred.npy` | LSTM predictions in MWh (4392, 3) |
| `y_test_actual.npy` | Actual test values in MWh (4416, 3) |
| `y_test_actual_lstm.npy` | Actual test values aligned for LSTM (4392, 3) |
| `model_evaluation_results.csv` | Full metrics table (MAE, RMSE, R²) for all models |

### Output Figures
| Figure | File | Description |
|---|---|---|
| Figure 1 | `fig_train_val_test_split.png` | Train/val/test chronological split |
| Figure 2 | `fig_annual_generation.png` | Annual wind and solar generation trends (2021-2024) |
| Figure 3 | `fig_monthly_patterns.png` | Monthly seasonal patterns (3 subplots) |
| Figure 4 | `fig_hourly_profiles.png` | Average hourly generation profiles (3 subplots) |
| Figure 5 | `fig_boxplots.png` | Generation distributions — boxplots (3 subplots) |
| Figure 6 | `fig_correlation_heatmap.png` | Correlation matrix — generation and weather features |
| Figure 7 | `fig_scatter_plots.png` | Weather vs generation scatter plots (3 subplots) |
| Figure 8 | `fig_summer_vs_winter.png` | Summer vs winter hourly comparison (3 subplots) |
| Figure 9 | `fig_acf.png` | ACF analysis — all three targets (lags=200) |
| Figure 10 | `fig_pacf.png` | PACF analysis — all three targets (lags=48) |
| Figure 11 | `fig_stl_wind.png` | STL decomposition — wind onshore (period=168, Jan-Jun 2021) |
| Figure 12 | `fig_stl_solar.png` | STL decomposition — solar PV (period=168, Jan-Jun 2021) |
| Figure 13 | `fig_rolling_stats.png` | Rolling statistics — stationarity visual check |
| Figure 14 | `fig_mlp_training.png` | MLP training and validation loss/MAE curves |
| Figure 15 | `fig_lstm_training.png` | LSTM training and validation loss/MAE curves |
| Figure 16 | `fig_predictions_vs_actual.png` | Predicted vs actual — first week of test set (all 3 models) |
| Figure 17 | `fig_model_comparison.png` | MAE, RMSE, R² bar charts — all models and targets |
| Figure 18 | `fig_residuals.png` | Residual distributions — all models and targets |
| Figure 19 | `fig_shap_lr.png` | SHAP feature importance — Linear Regression (LinearExplainer) |
| Figure 20 | `fig_shap_mlp.png` | SHAP feature importance — MLP (KernelExplainer, seed=42) |
| Figure 21 | `fig_shap_lstm.png` | SHAP feature importance — LSTM (KernelExplainer, flattened, seed=42) |
| Figure 22 | `fig_lime_lr.png` | LIME local explanations — Linear Regression (2 instances) |
| Figure 23 | `fig_lime_mlp.png` | LIME local explanations — MLP (2 instances) |
| Figure 24 | `fig_compression.png` | Compression results — size, R², reduction % (all 8 experiments) |

---

## Key Results

### Model Performance (Test Set: July-December 2024)

| Model | Wind Onshore R² | Wind Offshore R² | Solar R² | Size | Train Time | Inference |
|---|---|---|---|---|---|---|
| Linear Regression | **0.9845** | **0.9477** | 0.9792 | 1.64 KB | 0.14s | ~1ms |
| MLP | 0.9462 | 0.9374 | 0.9674 | 208.54 KB | 36.45s | 718ms |
| LSTM | 0.9614 | 0.8871 | **0.9821** | 1,623.37 KB | 671.10s | 2,571ms |

**Key finding:** Linear Regression achieves the highest R² for wind targets due to excellent lag-based feature engineering. LSTM achieves the best solar R² due to its temporal memory.

### Compression Results (Notebook 6)

| Model | Technique | Size (KB) | Reduction | Avg R² |
|---|---|---|---|---|
| LR | Feature Selection (SHAP top 10) | 0.89 | 45.7% | 0.9675 |
| LR | LASSO (α=0.001, 88.2% sparsity) | 1.51 | 7.6% | 0.9665 |
| MLP | Pruning (80% magnitude) | 210.75 | -1.1%* | 0.9600 |
| MLP | Pruning + TFLite int8 | 20.59 | **90.1%** | 0.9387 |
| MLP | Knowledge Distillation (32→16) | 26.63 | 87.2% | 0.8575 |
| LSTM | Pruning (80% magnitude) | 1,623.37 | 0.0%* | 0.9422 |
| LSTM | Pruning + float16 quantization | 264.47 | 83.7% | 0.9422 |
| LSTM | Knowledge Distillation LSTM(32) | 53.22 | **96.7%** | 0.8394 |

*Zero size reduction is expected — the .keras format stores zeros explicitly. The size benefit is realised through quantization in the next step.

### XAI Summary

| Method | Scope | Models | Key Finding |
|---|---|---|---|
| SHAP (LinearExplainer) | Global | Linear Regression | lag-1 features dominate all targets |
| SHAP (KernelExplainer) | Global | MLP, LSTM | lag-1 + weather features consistent |
| LIME | Local (2 instances) | LR, MLP | Consistent with SHAP global rankings |
| LIME instances | — | — | High Solar: 2024-07-30 12:00 (16,796 MWh), High Wind: 2024-12-18 17:00 (19,195 MWh) |

---

## Model Architectures

### Linear Regression
- **Input:** 34 features
- **Output:** 3 targets (wind onshore, wind offshore, solar)
- **Library:** scikit-learn LinearRegression

### MLP
- **Input:** 34 features
- **Architecture:** Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Dense(3, linear)
- **Total parameters:** 14,915
- **Optimizer:** Adam (lr=0.001)
- **Callbacks:** EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)
- **Trained for:** 16 epochs

### LSTM
- **Input:** (24, 34) — 24-hour lookback window, 34 features
- **Architecture:** LSTM(128, return_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, ReLU) → Dense(3, linear)
- **Total parameters:** 135,043
- **Optimizer:** Adam (lr=0.001)
- **Callbacks:** EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)
- **Trained for:** 22 epochs

---

## Required Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tensorflow>=2.11.0
shap>=0.41.0
lime>=0.2.0
statsmodels>=0.13.0
joblib>=1.2.0
tensorflow-model-optimization>=0.7.0
tf_keras>=2.15.0
```

---

## Hardware Requirements

- **CPU only** — no GPU required (TF GPU not supported on Windows ≥ 2.11)
- **RAM** — minimum 8GB recommended (LSTM training requires ~4GB)
- **Storage** — approximately 500MB for all generated files
- **Python** — version 3.10 recommended

**Estimated total runtime:** 90-100 minutes on a standard CPU

---

## References

- Bundesnetzagentur. (2024). *Electricity generation data — TenneT DE control zone*. SMARD. https://www.smard.de
- Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7970649
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD 2016*. https://doi.org/10.1145/2939672.2939778
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

