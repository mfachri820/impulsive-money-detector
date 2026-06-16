from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'impulsive-detector-main' / 'impulsive-detector-main' / 'data' / '03_final_data' / '04_Merged_labeled_transaction.csv'
MODEL_DIR = PROJECT_ROOT / 'AI-Fingo' / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'fingo_label_classifier.joblib'

LABEL_COLUMN = 'label'
NUMERIC_FEATURES = [
    'amount',
    'amount_log',
    'amount_z',
    'amount_score',
    'impulsive_score',
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

if __name__ == '__main__':
    print(f'Reading data from {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f'Missing target column: {LABEL_COLUMN}')

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[LABEL_COLUMN].astype(str)

    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(0)
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna('missing')

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES),
    ])

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print('Training model...')
    clf.fit(X_train, y_train)

    print('Evaluating model...')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred, labels=['AMAN', 'PERTIMBANGAN', 'IMPULSIF']))

    dump(clf, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')
