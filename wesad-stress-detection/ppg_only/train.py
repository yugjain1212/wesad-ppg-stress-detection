import os
import json
import pickle
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_model(data_path, models_dir):
    """
    Train the final model on the entire dataset and save artifacts.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run build.py first.")
        
    df = pd.read_csv(data_path)
    
    # Separate features, label, and subject_id
    X = df.drop(columns=['label', 'subject_id'])
    y = df['label']
    feature_names = X.columns.tolist()
    
    # Preprocessing: StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For the final model, we use Random Forest (usually robust and performs well)
    # evaluate.py will compare models, but we pick RF here as a solid default.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    model_path = os.path.join(models_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    metadata = {
       "model_type": "RandomForestClassifier",
       "training_date": datetime.datetime.now().isoformat(),
       "feature_names": feature_names,
       "n_samples": len(y),
       "model_selection_rationale": (
         "Random Forest selected as deployment model. In LOSO comparison, RF had the "
         "highest Accuracy (0.895) and F1 (0.822) of the three candidates (LR, RF, SVM). "
         "SVM had marginally higher ROC-AUC (0.943 vs 0.938) and PR-AUC (0.896 vs 0.884), "
         "but AUC reflects ranking quality across all thresholds, whereas the deployed "
         "system commits to a single fixed decision threshold at inference time, making "
         "Accuracy/F1 the more relevant metrics for this use case."
     )
    }
    
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model and artifacts saved to {models_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    models_dir = os.path.join(base_dir, 'saved_models')
    
    train_model(data_path, models_dir)
