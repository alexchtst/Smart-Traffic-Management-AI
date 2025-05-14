import joblib
import numpy as np

# Load pipeline
pipeline = joblib.load("models/rf_pipeline.joblib")

# Input data (harus sepanjang lookback, misal 4)
input_seq = [113, 225, 69, 258]
input_array = np.array(input_seq).reshape(1, -1)

# Prediksi
prediction = pipeline.predict(input_array)[0]

print(f"📈 Prediksi untuk {input_seq} → {prediction:.2f}")

from ultralytics import YOLO
import os

print(os.path.exists("AI-Detection/videoplayback.mp4"))

model = YOLO("AI-Detection/ambulance5mar/ambulance5mar.pt")

model.predict(source="AI-Detection/videoplayback.mp4", show=True)