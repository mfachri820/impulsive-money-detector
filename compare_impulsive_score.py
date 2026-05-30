from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'impulsive-detector-main' / 'impulsive-detector-main' / 'data' / '03_final_data' / '04_Merged_labeled_transaction.csv'

NUMERIC_FEATURES = [
    'amount',
    'amount_log',
    'amount_z',
    'amount_score',
    'hour',
    'day_of_week',
    'driver_count',
]
CATEGORICAL_FEATURES = [
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
TARGET_COLUMN = 'label'


def build_pipeline(include_impulsive_score: bool):
    numeric_features = NUMERIC_FEATURES.copy()
    if include_impulsive_score:
        numeric_features = numeric_features + ['impulsive_score']

    return Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ])),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def load_data():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f'Missing target column: {TARGET_COLUMN}')
    return df


def preprocess(df, include_impulsive_score: bool):
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if include_impulsive_score:
        features = features + ['impulsive_score']

    X = df[features].copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    if include_impulsive_score:
        X['impulsive_score'] = pd.to_numeric(X['impulsive_score'], errors='coerce').fillna(0)
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna('missing').astype(str)
    y = df[TARGET_COLUMN].astype(str)
    return X, y


def evaluate(include_impulsive_score: bool):
    df = load_data()
    X, y = preprocess(df, include_impulsive_score)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = build_pipeline(include_impulsive_score)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    label = 'WITH impulsive_score' if include_impulsive_score else 'WITHOUT impulsive_score'
    print(label)
    print('accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print('-' * 60)


if __name__ == '__main__':
    print('Data:', DATA_PATH)
    evaluate(True)
    evaluate(False)
