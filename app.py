from pathlib import Path
from joblib import load
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / 'AI-Fingo' / 'model' / 'fingo_label_classifier.joblib'

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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Model not found. Run train_model.py first or place the model at {MODEL_PATH}'
        )
    return load(MODEL_PATH)


MODEL = load_model()


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f'Missing input columns: {missing}')

    X = df[FEATURES].copy()
    X[FEATURES] = X[FEATURES].fillna('missing')

    df = df.copy()
    df['predicted_label'] = MODEL.predict(X)
    df['alert'] = df['predicted_label'].apply(
        lambda v: 'ALERT' if v in ALERT_LABELS else 'NO_ALERT'
    )
    return df


def predict_json(rows):
    if rows is None:
        return {'error': 'No rows provided'}

    if isinstance(rows, dict) and 'data' in rows:
        rows = rows['data']

    try:
        df = pd.DataFrame(rows)
    except Exception as exc:
        return {'error': f'Invalid JSON input: {exc}'}

    try:
        result = predict_dataframe(df)
    except Exception as exc:
        return {'error': str(exc)}

    return result.to_dict(orient='records')


if __name__ == '__main__':
    import gradio as gr

    def predict_file(csv_file):
        if csv_file is None:
            return 'Upload a CSV file with the expected feature columns.'

        try:
            df = pd.read_csv(csv_file.name)
        except Exception as exc:
            return f'Failed to read CSV: {exc}'

        try:
            result = predict_dataframe(df)
        except Exception as exc:
            return str(exc)

        return result[['predicted_label', 'alert'] + FEATURES]

    demo = gr.Blocks()

    with demo:
        gr.Markdown(
            '# Fingo Alert Label Predictor\nUpload a CSV or send JSON rows for inference.'
        )

        with gr.Tab('CSV Upload'):
            csv_input = gr.File(file_types=['.csv'], label='Upload CSV')
            csv_output = gr.Dataframe(type='pandas', label='Predictions')
            csv_button = gr.Button('Submit')
            csv_button.click(predict_file, csv_input, csv_output)

        with gr.Tab('JSON API'):
            json_input = gr.JSON(label='Transaction rows (list of objects)')
            json_output = gr.JSON(label='Predictions')
            json_button = gr.Button('Submit')
            json_button.click(predict_json, json_input, json_output)

    demo.launch()
