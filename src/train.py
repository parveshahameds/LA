import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import get_preprocessing_pipeline, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def train_and_evaluate():
    # Make sure output directories exist
    os.makedirs("models", exist_ok=True)
    
    # 1. Load historical projects dataset
    if not os.path.exists("data/historical_projects.csv"):
        raise FileNotFoundError("Historical dataset 'data/historical_projects.csv' not found. Run data_generator.py first.")
        
    print("Loading historical data...")
    df = pd.read_csv("data/historical_projects.csv")
    
    # 2. Split features and targets
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y_class = df["delayed"]
    y_reg = df["actual_delay_days"]
    
    # 3. Train-test split
    # Split using same random state to keep classifications and regressions aligned
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    # 4. Preprocessing
    print("Fitting preprocessor...")
    preprocessor = get_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 5. Model Training - Classifier
    print("Training RandomForestClassifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(X_train_processed, y_train_class)
    
    # 6. Model Training - Regressor
    print("Training RandomForestRegressor...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    regressor.fit(X_train_processed, y_train_reg)
    
    # 7. Model Evaluation
    print("Evaluating models...")
    # Classifier evaluation
    y_pred_class = classifier.predict(X_test_processed)
    y_prob_class = classifier.predict_proba(X_test_processed)[:, 1]
    
    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class)
    recall = recall_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)
    roc_auc = roc_auc_score(y_test_class, y_prob_class)
    
    # Regressor evaluation
    y_pred_reg = regressor.predict(X_test_processed)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    # Create metadata JSON
    metadata = {
        "model_version": "1.0.0",
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_dataset_size": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "classification": {
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(roc_auc), 4)
            },
            "regression": {
                "mae_days": round(float(mae), 2),
                "rmse_days": round(float(rmse), 2),
                "r2_score": round(float(r2), 4)
            }
        }
    }
    
    # Print metrics
    print("\n--- Classification Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\n--- Regression Metrics ---")
    print(f"MAE:       {mae:.2f} days")
    print(f"RMSE:      {rmse:.2f} days")
    print(f"R2 Score:  {r2:.4f}")
    
    # 8. Persist artifacts
    print("\nSaving model files to models/...")
    joblib.dump(classifier, "models/delay_classifier.pkl")
    joblib.dump(regressor, "models/delay_regressor.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Training process complete!")

if __name__ == "__main__":
    train_and_evaluate()
