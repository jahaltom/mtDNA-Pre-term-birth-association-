# Gradient Boosting GA Prediction Pipeline

This repository contains a **site‑aware Gradient Boosting Regression workflow** for predicting gestational age at birth (GAGEBRTH).

The script:

- Loads metadata
- Handles preprocessing (scaling + one‑hot encoding)
- Performs site‑aware CV / splitting
- Tunes model hyperparameters
- Evaluates performance
- Retrains best model on full dataset
- Generates interpretability reports

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit‑learn
- seaborn
- common_reports.py

## Input File

`Metadata.Final.tsv` must contain:

- `GAGEBRTH` (target)
- feature columns
- `site` column recommended

## Usage

Example:

```bash
python GB.GA.py \
    "HAPLOGROUP,SEX" \
    "MAT_HEIGHT,MAT_WEIGHT" \
    "TOILET,WATER"
```

Arguments:

1. comma‑separated categorical columns  
2. comma‑separated continuous numeric columns  
3. comma‑separated binary columns  

⚠ Do not include `site` in argument 1 — the script detects it automatically.

## What the Script Does

### 1. Preprocesses features  
- standardizes numeric features  
- applies one‑hot encoding to categorical features  

### 2. Site‑Aware Splitting  
- ≥3 sites → unseen‑site generalization test  
- 2 sites → CV respects site but test split mixes  
- otherwise → standard splits  

### 3. Model Training  
Uses scikit‑learn `GradientBoostingRegressor`.

Hyperparameters searched:

```python
n_estimators = [100, 200]
learning_rate = [0.01, 0.1]
max_depth = [3, 5]
```

### 4. Evaluation  
Outputs:

- Mean squared error (MSE)
- R²

### 5. Full Retrain + Interpretation

Runs:

```
run_common_reports(...)
```

producing:

- feature rankings
- interaction effects
- PDP plots
- RFE summary

## Outputs

Files begin with:

```
GA_*
```

Example outputs:

- `GA_top_main_features.tsv`
- `GA_interaction_scores.tsv`
- `GA_pdp*.png`
- `GA_rfe.txt`

## Author

Jeff Haltom

## License

MIT
