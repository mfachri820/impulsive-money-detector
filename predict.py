from pathlib import Path
import argparse
import pandas as pd
from joblib import load

MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'fingo_label_classifier.joblib'

FEATURES = [
    'amount',
    'amount_log',
    'amount_z',
    'amount_score',
    'impulsive_score',
    'hour',
    'day_of_week',
    'driver_count',
    'category',
    'metode_pembayaran',
    'source',
    'time_segment',
    'category_type',
    'is_hedonic_category',
    'is_night',
    'is_weekend',
    'signal_band',
]

ALERT_LABELS = ['PERTIMBANGAN', 'IMPULSIF']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Fingo alert label for new transaction rows')
    parser.add_argument('--input', required=True, help='CSV path containing new transactions')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV path with predictions')
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found. Run train_model.py first. Expected at {MODEL_PATH}')

    df = pd.read_csv(args.input)
    if not all(col in df.columns for col in FEATURES):
        missing = [col for col in FEATURES if col not in df.columns]
        raise ValueError(f'Missing input columns: {missing}')

    model = load(MODEL_PATH)
    X = df[FEATURES].copy()
    X[FEATURES] = X[FEATURES].fillna('missing')

    df['predicted_label'] = model.predict(X)
    df['alert'] = df['predicted_label'].apply(lambda v: 'ALERT' if v in ALERT_LABELS else 'NO_ALERT')
    df.to_csv(args.output, index=False)

    print(f'Predictions saved to {args.output}')
    print(df[['predicted_label', 'alert']].head())
