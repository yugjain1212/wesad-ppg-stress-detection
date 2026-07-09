import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

def evaluate_models(data_path, results_dir):
    """
    Perform Leave-One-Subject-Out (LOSO) cross-validation and generate evaluation artifacts.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run build.py first.")
        
    df = pd.read_csv(data_path)
    subjects = df['subject_id'].unique()
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # We will track results for Random Forest as the primary model to plot,
    # but we will evaluate all models and save a summary.
    
    all_results = []
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        y_true_all = []
        y_prob_all = []
        y_pred_all = []
        
        fold_metrics = []
        
        for subj in subjects:
            train_df = df[df['subject_id'] != subj]
            test_df = df[df['subject_id'] == subj]
            
            X_train = train_df.drop(columns=['label', 'subject_id'])
            y_train = train_df['label']
            
            X_test = test_df.drop(columns=['label', 'subject_id'])
            y_test = test_df['label']
            
            # Scale per fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            y_true_all.extend(y_test)
            y_prob_all.extend(y_prob)
            y_pred_all.extend(y_pred)
            
            # Fold metrics
            fold_metrics.append({
                'subject': subj,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
                'pr_auc': average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan
            })
            
        # Aggregate metrics
        y_true_all = np.array(y_true_all)
        y_prob_all = np.array(y_prob_all)
        y_pred_all = np.array(y_pred_all)
        
        agg_accuracy = accuracy_score(y_true_all, y_pred_all)
        agg_precision = precision_score(y_true_all, y_pred_all)
        agg_recall = recall_score(y_true_all, y_pred_all)
        agg_f1 = f1_score(y_true_all, y_pred_all)
        agg_roc_auc = roc_auc_score(y_true_all, y_prob_all)
        agg_pr_auc = average_precision_score(y_true_all, y_prob_all)
        
        all_results.append({
            'Model': model_name,
            'Accuracy': agg_accuracy,
            'Precision': agg_precision,
            'Recall': agg_recall,
            'F1': agg_f1,
            'ROC-AUC': agg_roc_auc,
            'PR-AUC': agg_pr_auc
        })
        
        # Save plots for Random Forest
        if model_name == "Random Forest":
            os.makedirs(results_dir, exist_ok=True)
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_all, y_pred_all)
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            fig.colorbar(cax)
            for (i, j), z in np.ndenumerate(cm):
                ax.text(j, i, '{:d}'.format(z), ha='center', va='center')
            plt.title('Confusion Matrix (Aggregated LOSO)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
            plt.close()
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {agg_roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
            plt.close()
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
            plt.figure()
            plt.plot(recall, precision, label=f'PR curve (area = {agg_pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(results_dir, 'pr_curve.png'))
            plt.close()
            
            # Save fold metrics
            pd.DataFrame(fold_metrics).to_csv(os.path.join(results_dir, 'rf_fold_metrics.csv'), index=False)

    # Save aggregated results
    results_df = pd.DataFrame(all_results)
    print("\n--- Aggregated LOSO Results ---")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    results_dir = os.path.join(base_dir, 'results')
    
    evaluate_models(data_path, results_dir)
