# AI-Fingo Documentation

## Project purpose

This folder contains a machine learning pipeline for predicting transaction alert labels from the Fingo dataset. The goal is to classify each transaction into one of:
- `AMAN`
- `PERTIMBANGAN`
- `IMPULSIF`

The model uses transaction metadata and derived features to predict the label and identify alert cases.

## Dataset

The training data is loaded from:
- `../impulsive-detector-main/impulsive-detector-main/data/03_final_data/04_Merged_labeled_transaction.csv`

The dataset uses Indonesian Rupiah (`IDR`) as the canonical currency after source conversions.

## Files created / updated

### `train_model.py`
- Trains a Random Forest classifier on labeled transaction data.
- Reads the dataset from the moved repository location.
- Builds a preprocessing pipeline with standard scaling for numeric features and one-hot encoding for categorical features.
- Saves the trained model to `AI-Fingo/model/fingo_label_classifier.joblib`.
- Fixed `OneHotEncoder` compatibility by using `sparse_output=False`.

### `predict.py`
- Loads the trained model from `AI-Fingo/model/fingo_label_classifier.joblib`.
- Predicts labels for new transactions from a CSV input file.
- Adds `predicted_label` and `alert` columns to the output.
- Flags `PERTIMBANGAN` and `IMPULSIF` as `ALERT`.

### `sample_input.csv`
- Added a sample input file with the required feature columns for prediction.
- Includes example transactions for `Transportasi`, `Makanan`, and `Kesehatan`.

### `compare_impulsive_score.py`
- Compares model performance for:
  - training with `impulsive_score`
  - training without `impulsive_score`
- Helps detect target leakage from score-derived features.

### `time_based_evaluation.py`
- Implements a time-based holdout evaluation using the `timestamp` field.
- Adds rolling time-series cross-validation across consecutive time windows.
- Evaluates two variants:
  - with score-derived features
  - without score-derived features
- Confirms whether the model generalizes on unseen future periods.

### `check_duplicates.py`
- Checks for exact duplicate rows in the dataset.
- Checks whether any exact duplicate row appears in both train and test after a random split.
- Ensures the dataset has no identical record overlap between train and test.

## Feature groups

### Base features
- `amount`
- `amount_log`
- `amount_z`
- `hour`
- `day_of_week`
- `driver_count`
- `transactions_same_day`
- `daily_amount_total`
- `share_of_daily_spend`
- `amount_vs_weekly_avg`
- `budget_limit`
- `spent_so_far`
- `budget_remaining_ratio`

### Base categorical features
- `category`
- `metode_pembayaran`
- `source`
- `time_segment`
- `category_type`
- `is_hedonic_category`
- `is_night`
- `is_weekend`
- `status_transaksi`
- `user_proxy`

### Score-derived features
- `amount_score`
- `amount_percentile`
- `category_score`
- `weekend_score`
- `impulsive_score`
- `signal_band`

## Evaluation results

### Random split performance
- With `impulsive_score`:
  - accuracy: `0.99953`
- Without `impulsive_score`:
  - accuracy: `0.99294`

### Time-based holdout
- With score-derived features:
  - accuracy: `0.99303`
- Without score-derived features:
  - accuracy: `0.95258`

### Rolling time-series CV (5 splits)
- With score-derived features:
  - average accuracy: `0.99237`
- Without score-derived features:
  - average accuracy: `0.94351`

## Key conclusions

- The model is very strong when score-derived features are included.
- Removing score-derived features produces a significant drop in performance.
- Time-based evaluation confirms the model still generalizes well over future periods, but score-derived features remain influential.
- The data contains no exact duplicate rows, and the train/test split does not share identical records.

## How to run

From `AI-Fingo`:

```powershell
python train_model.py
python predict.py --input sample_input.csv --output predictions.csv
python compare_impulsive_score.py
python time_based_evaluation.py
python check_duplicates.py
```

## Notes

- The script `time_based_evaluation.py` is the recommended generalization test for this project.
- If you want a stricter evaluation, you can extend the feature exclusion set or use wider rolling windows.
