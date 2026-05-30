# AI-fingo

This folder contains the machine learning pipeline for predicting transaction alert labels based on `fingo_impulse_signal` and related features.

## Files

- `train_model.py` - trains a multiclass classifier to predict `label` as `AMAN`, `PERTIMBANGAN`, or `IMPULSIF`.
- `predict.py` - loads the trained model and predicts labels for new transaction data.
- `requirements.txt` - Python packages required to run the pipeline.

## Usage

1. Create a Python environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```

3. Make predictions on new rows:
   ```bash
   python predict.py --input sample_input.csv
   ```

4. Compare the effect of `impulsive_score` on model quality:
   ```bash
   python compare_impulsive_score.py
   ```

5. Run a time-based holdout and rolling CV evaluation on unseen periods:
   ```bash
   python time_based_evaluation.py
   ```
   This script evaluates both:
   - time-based holdout on the latest period
   - rolling time-series CV across consecutive windows
   - models with and without score-derived features

7. Use the Gradio app for cloud-style deployment or local inference:
   ```bash
   python app.py
   ```

The training script reads the labeled dataset from the repository data folder:
`../impulsive-detector-main/impulsive-detector-main/data/03_final_data/04_Merged_labeled_transaction.csv`.

## Deploying to Hugging Face

The model can be hosted on Hugging Face by publishing the trained `model/fingo_label_classifier.joblib` file to a Space repository and using `app.py` as the entrypoint.

### Recommended workflow

1. Train the model locally once and confirm `AI-Fingo/model/fingo_label_classifier.joblib` exists.
2. Create a new Hugging Face Space, for example `mfachri820/AIFingo`.
3. Clone the Space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/mfachri820/AIFingo
   ```
4. Copy or move the following files into the cloned Space repo:
   - `app.py`
   - `requirements.txt`
   - `model/fingo_label_classifier.joblib`
5. Commit and push:
   ```bash
   git add app.py requirements.txt model/fingo_label_classifier.joblib
   git commit -m "Add Gradio Space app for Fingo inference"
   git push
   ```

### Notes

- `gradio` is already included in `requirements.txt`.
- The Space app loads the trained model once at startup, so no retraining is needed for each API request.
- The app accepts CSV uploads and returns `predicted_label` and `alert` values.

Once deployed, the Space will run automatically and provide a web endpoint for inference.

## Using the Space API

After deployment, you can call the Space via the Hugging Face API endpoint:

```bash
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" -H "Content-Type: application/json" --data '{"data":[{"amount":100,"amount_log":4.6052,"amount_z":0.3,"amount_score":0.5,"impulsive_score":0.7,"hour":14,"day_of_week":2,"driver_count":1,"category":"food","metode_pembayaran":"card","source":"ecommerce","time_segment":"afternoon","category_type":"essential","is_hedonic_category":false,"is_night":false,"is_weekend":false,"signal_band":"medium"}]}' https://api-inference.huggingface.co/spaces/mfachri820/AI-Fingo
```

The current `app.py` also supports CSV upload through the Space UI.
