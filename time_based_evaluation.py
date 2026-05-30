from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'impulsive-detector-main' / 'impulsive-detector-main' / 'data' / '03_final_data' / '04_Merged_labeled_transaction.csv'
BASE_NUMERIC_FEATURES = [
    'amount',
    'amount_log',
    'amount_z',
    'hour',
    'day_of_week',
    'driver_count',
    'transactions_same_day',
    'daily_amount_total',
    'share_of_daily_spend',
    'amount_vs_weekly_avg',
    'budget_limit',
    'spent_so_far',
    'budget_remaining_ratio',
]
SCORE_NUMERIC_FEATURES = [
    'amount_score',
    'amount_percentile',
    'category_score',
    'weekend_score',
    'impulsive_score',
]
BASE_CATEGORICAL_FEATURES = [
    'category',
    'metode_pembayaran',
    'source',
    'time_segment',
    'category_type',
    'is_hedonic_category',
    'is_night',
    'is_weekend',
    'status_transaksi',
    'user_proxy',
]
SCORE_CATEGORICAL_FEATURES = [
    'signal_band',
]
TARGET_COLUMN = 'label'
TIME_COLUMN = 'timestamp'


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors='coerce')
    return df.sort_values(TIME_COLUMN).reset_index(drop=True)


def feature_set(include_score_features: bool):
    numeric_features = BASE_NUMERIC_FEATURES.copy()
    categorical_features = BASE_CATEGORICAL_FEATURES.copy()
    if include_score_features:
        numeric_features += SCORE_NUMERIC_FEATURES
        categorical_features += SCORE_CATEGORICAL_FEATURES
    return numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    return Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ])),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def preprocess(df, numeric_features, categorical_features):
    features = numeric_features + categorical_features
    X = df[features].copy()
    X[numeric_features] = X[numeric_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    X[categorical_features] = X[categorical_features].fillna('missing').astype(str)
    y = df[TARGET_COLUMN].astype(str)
    return X, y


def time_holdout(df, test_fraction=0.25):
    cutoff = df[TIME_COLUMN].quantile(1.0 - test_fraction)
    train = df[df[TIME_COLUMN] <= cutoff].copy()
    test = df[df[TIME_COLUMN] > cutoff].copy()
    return train, test, cutoff


def evaluate_holdout(df, include_score_features: bool):
    numeric_features, categorical_features = feature_set(include_score_features)
    train, test, cutoff = time_holdout(df, test_fraction=0.25)
    X_train, y_train = preprocess(train, numeric_features, categorical_features)
    X_test, y_test = preprocess(test, numeric_features, categorical_features)

    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    label = 'WITH score-derived features' if include_score_features else 'WITHOUT score-derived features'
    print(label)
    print('train rows:', len(train), 'test rows:', len(test), 'cutoff:', cutoff)
    print('accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print('-' * 80)


def rolling_time_series_cv(df, n_splits=5):
    fold_size = len(df) // (n_splits + 1)
    results = []
    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = train_end + fold_size
        if fold == n_splits - 1:
            test_end = len(df)

        train = df.iloc[:train_end].copy()
        test = df.iloc[test_start:test_end].copy()
        results.append((fold + 1, train, test))
    return results


def evaluate_rolling_cv(df, include_score_features: bool, n_splits=5):
    numeric_features, categorical_features = feature_set(include_score_features)
    results = rolling_time_series_cv(df, n_splits=n_splits)
    fold_metrics = []

    print('Rolling time-series CV (n_splits=%d) %s' % (n_splits, 'WITH score-derived features' if include_score_features else 'WITHOUT score-derived features'))
    for fold_idx, train, test in results:
        X_train, y_train = preprocess(train, numeric_features, categorical_features)
        X_test, y_test = preprocess(test, numeric_features, categorical_features)
        model = build_pipeline(numeric_features, categorical_features)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        fold_metrics.append(accuracy)
        print(f'Fold {fold_idx}: train={len(train)} test={len(test)} accuracy={accuracy:.4f}')

    avg_accuracy = sum(fold_metrics) / len(fold_metrics)
    print('average accuracy:', avg_accuracy)
    print('-' * 80)


if __name__ == '__main__':
    df = load_data()
    print('Dataset loaded:', DATA_PATH)
    print('Time range:', df[TIME_COLUMN].min(), '->', df[TIME_COLUMN].max())
    evaluate_holdout(df, include_score_features=True)
    evaluate_holdout(df, include_score_features=False)
    evaluate_rolling_cv(df, include_score_features=True, n_splits=5)
    evaluate_rolling_cv(df, include_score_features=False, n_splits=5)
