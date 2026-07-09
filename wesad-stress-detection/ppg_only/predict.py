import os
import json
import pickle
import numpy as np
import pandas as pd

def predict_offline(data_path, models_dir):
    """
    Load the trained model and scaler, and run predictions on the dataset offline.
    Useful for sanity checking the deployment pipeline.
    """
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'model.pkl')
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Run train.py first.")
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    feature_names = metadata['feature_names']
    
    df = pd.read_csv(data_path)
    
    # Ensure columns match training order
    X = df[feature_names]
    
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    df['predicted_label'] = predictions
    df['stress_probability'] = probabilities
    
    print(f"Ran offline prediction on {len(df)} samples.")
    print("Sample output:")
    print(df[['subject_id', 'label', 'predicted_label', 'stress_probability']].head())
    
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    models_dir = os.path.join(base_dir, 'saved_models')
    
    predict_offline(data_path, models_dir)
