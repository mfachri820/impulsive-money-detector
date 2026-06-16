# AI-Fingo Backend Integration Guide

This document provides the integration details required to connect with the AI-Fingo predictive inference service.

The predictive engine is hosted on Hugging Face Spaces and uses Gradio 5 under the hood.

## 🌐 Core Service Specifications

- **Base URL:** `https://mfachri820-ai-fingo.hf.space`
- **Authentication:** Public access by default. Optional `Authorization: Bearer <HF_TOKEN>` header if the Space becomes private.
- **Default Content-Type:** `application/json`

### ⚠️ Gradio 5 Architecture Note

Gradio 5 uses asynchronous event queues instead of standard stateless REST endpoints. Manual HTTP requests must register an execution thread under `/call/`, then read the streamed prediction output.

**Recommendation:** use the official Gradio client libraries for Python or JavaScript. They handle the event-loop handshake automatically.

## ⚡ API Endpoints

### 1. JSON real-time batch inference

Processes raw feature dictionaries via the scikit-learn model pipeline.

- **Relative endpoint:** `/call/predict_json`
- **HTTP method:** `POST`
- **Payload key:** `rows` (array of feature dictionaries)

#### Example request payload

```json
{
  "data": [
    [
      {
        "amount": 100,
        "amount_log": 4.6052,
        "amount_z": 0.3,
        "amount_score": 0.5,
        "impulsive_score": 0.7,
        "hour": 14,
        "day_of_week": 2,
        "driver_count": 1,
        "category": "food",
        "metode_pembayaran": "card",
        "source": "ecommerce",
        "time_segment": "afternoon",
        "category_type": "essential",
        "is_hedonic_category": false,
        "is_night": false,
        "is_weekend": false,
        "signal_band": "medium"
      }
    ]
  ]
}
```

### Event queue flow (raw HTTP)

1. `POST https://mfachri820-ai-fingo.hf.space/call/predict_json`
2. Receive an execution token, e.g.:

```json
{"event_id": "9a7b5c3e21fd409cb..."}
```

3. Poll the event result at:

```text
https://mfachri820-ai-fingo.hf.space/call/predict_json/9a7b5c3e21fd409cb...
```

#### Example response payload

```json
[
  {
    "amount": 100,
    "amount_log": 4.6052,
    "amount_z": 0.3,
    "amount_score": 0.5,
    "impulsive_score": 0.7,
    "hour": 14,
    "day_of_week": 2,
    "driver_count": 1,
    "category": "food",
    "metode_pembayaran": "card",
    "source": "ecommerce",
    "time_segment": "afternoon",
    "category_type": "essential",
    "is_hedonic_category": false,
    "is_night": false,
    "is_weekend": false,
    "signal_band": "medium",
    "predicted_label": "AMAN",
    "alert": "NO_ALERT"
  }
]
```

### 2. Bulk CSV file inference

Accepts a multi-row CSV upload and returns a processed prediction result.

- **Relative endpoint:** `/call/predict_file`
- **HTTP method:** `POST`
- **Payload key:** `csv_file` (Gradio `FileData` object)

## 🛠️ Native client integration

Use the official SDKs to avoid manual event-stream handling.

### Python backends (FastAPI, Django, Flask)

```bash
pip install gradio_client
```

```python
from gradio_client import Client

client = Client("mfachri820/AI-Fingo")

def fetch_transaction_prediction(transaction_features: dict) -> list:
    try:
        response = client.predict(
            rows=[transaction_features],
            api_name="/predict_json"
        )
        return response
    except Exception as e:
        print(f"Prediction service failure: {e}")
        return []

payload = {
    "amount": 100,
    "amount_log": 4.6052,
    "amount_z": 0.3,
    "amount_score": 0.5,
    "impulsive_score": 0.7,
    "hour": 14,
    "day_of_week": 2,
    "driver_count": 1,
    "category": "food",
    "metode_pembayaran": "card",
    "source": "ecommerce",
    "time_segment": "afternoon",
    "category_type": "essential",
    "is_hedonic_category": False,
    "is_night": False,
    "is_weekend": False,
    "signal_band": "medium"
}

prediction_output = fetch_transaction_prediction(payload)
print(prediction_output)
```

### Node.js / TypeScript backends (Express, NestJS)

```bash
npm install @gradio/client
```

```ts
import { Client } from "@gradio/client";

async function runFingoInference(transactionData: Record<string, any>) {
    try {
        const app = await Client.connect("mfachri820/AI-Fingo");
        const response = await app.predict("/predict_json", {
            rows: [transactionData]
        });
        console.log("Prediction output:", response.data);
        return response.data;
    } catch (error) {
        console.error("API call to AI-Fingo failed:", error);
        throw error;
    }
}
```

## ⚙️ Model sandbox specifications

- **Runtime core:** Python (scikit-learn)
- **Dependency constraint:** validated with `scikit-learn==1.7.1`
