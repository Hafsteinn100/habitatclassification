import joblib
import numpy as np
from pathlib import Path
from utils import extract_features 

# 1. Load the model ONCE when the server starts
# We do NOT train here. We just load the file you made earlier.
MODEL_PATH = Path(__file__).parent / "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("Did you run 'python train_model.py' first?")
    model = None

def predict(patch):
    """
    Takes a single patch (15, 35, 35), extracts features, and returns the predicted class.
    """
    if model is None:
        raise RuntimeError("Model is not loaded! Run train_model.py first.")

    # 1. Feature Extraction (Must match what you used in training!)
    # This now includes the NDVI math we added to utils.py
    features = extract_features(patch)
    
    # 2. Reshape for the model (1 sample, many features)
    features_batch = features.reshape(1, -1)
    
    # 3. Predict
    prediction = model.predict(features_batch)
    
    # Return the integer class
    return int(prediction[0])