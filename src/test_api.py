from gradio_client import Client

# 1. Connect to your active Space container
client = Client("mfachri820/AI-Fingo")

# 2. Match the exact parameter name 'rows' and the target API route
result = client.predict(
    rows=[{
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
    }],
    api_name="/predict_json"
)

print("\n🚀 Prediction Output from Space:")
print(result)