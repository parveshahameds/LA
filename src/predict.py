import os
import joblib
import pandas as pd
import numpy as np

# Load models safely
def load_ml_artifacts():
    classifier_path = "models/delay_classifier.pkl"
    regressor_path = "models/delay_regressor.pkl"
    preprocessor_path = "models/preprocessor.pkl"
    
    if not (os.path.exists(classifier_path) and os.path.exists(regressor_path) and os.path.exists(preprocessor_path)):
        raise FileNotFoundError("Models not trained yet. Run train.py first.")
        
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return classifier, regressor, preprocessor

def get_risk_category(risk_score):
    if risk_score <= 24:
        return "LOW", "🟢"
    elif risk_score <= 49:
        return "MODERATE", "🟡"
    elif risk_score <= 74:
        return "HIGH", "🟠"
    else:
        return "CRITICAL", "🔴"

def predict_project_risk(project_df, classifier, regressor, preprocessor):
    """
    Predicts the delay probability and expected delay for a dataframe of projects.
    Returns the dataframe with predicted fields added.
    """
    # Transform features
    X_processed = preprocessor.transform(project_df)
    
    # Predict delay probability
    prob = classifier.predict_proba(X_processed)[:, 1]
    
    # Predict expected delay days
    delay_days = regressor.predict(X_processed)
    # expected delay should be zero if probability of delay is very small, or scaled
    # Let's clean up regression output to be positive and logically consistent
    delay_days = np.maximum(0, delay_days)
    
    # Add predictions
    result_df = project_df.copy()
    result_df["delay_probability"] = np.round(prob * 100, 1)
    
    # Risk score is equivalent to probability in this context, range 0-100
    result_df["risk_score"] = np.round(prob * 100).astype(int)
    
    # Add Risk Category and Emoji
    categories = []
    emojis = []
    for score in result_df["risk_score"]:
        cat, emo = get_risk_category(score)
        categories.append(cat)
        emojis.append(emo)
        
    result_df["risk_category"] = categories
    result_df["risk_emoji"] = emojis
    result_df["expected_delay_days"] = np.round(delay_days).astype(int)
    
    # Format expected delay as months for presentation
    result_df["expected_delay_months"] = np.round(delay_days / 30.4, 1)
    
    return result_df

def calculate_stage_wise_risk(project_row):
    """
    Calculates stage-wise risks (0 to 100%) for a single project based on its features.
    These are logical risk metrics capturing the probability of failure/bottlenecks in each stage.
    """
    # Extract features
    land_area = project_row.get("land_area_acres", 100.0)
    affected_families = project_row.get("affected_families", 50)
    district_delay = project_row.get("historical_district_delay_rate", 0.3)
    type_delay = project_row.get("historical_project_type_delay_rate", 0.3)
    
    approval_delay = project_row.get("approval_delay_days", 0.0)
    pending_approvals = project_row.get("pending_approvals", 0.0)
    
    doc_pct = project_row.get("documentation_completion_pct", 100.0)
    ownership_conflicts = project_row.get("ownership_conflicts", 0.0)
    
    legal_disputes = project_row.get("legal_disputes", 0.0)
    dispute_age = project_row.get("average_dispute_age_days", 0.0)
    
    comp_approved_pct = project_row.get("compensation_approved_pct", 100.0)
    comp_paid_pct = project_row.get("compensation_paid_pct", 100.0)
    comp_pending_families = project_row.get("compensation_pending_families", 0)
    
    rr_pct = project_row.get("rr_progress_pct", 100.0)
    rr_pending_families = project_row.get("rehabilitation_pending_families", 0)
    
    possession_pct = project_row.get("possession_pct", 100.0)
    
    backlog = project_row.get("department_backlog", 10.0)
    
    # 1. Preliminary Identification Risk
    r1 = 30 * min(affected_families/1000.0, 1.0) + 30 * min(land_area/1500.0, 1.0) + 40 * district_delay
    
    # 2. Notification Risk
    r2 = 50 * min(approval_delay/150.0, 1.0) + 30 * (pending_approvals/4.0) + 20 * type_delay
    
    # 3. Survey & Documentation Risk
    r3 = 60 * (1.0 - doc_pct/100.0) + 40 * min(ownership_conflicts/15.0, 1.0)
    
    # 4. Objections / Legal Review Risk
    r4 = 50 * min(legal_disputes/10.0, 1.0) + 50 * min(dispute_age/365.0, 1.0)
    
    # 5. Compensation Assessment Risk
    r5 = 70 * (1.0 - comp_approved_pct/100.0) + 30 * min(affected_families/1000.0, 1.0)
    
    # 6. Compensation Disbursement Risk
    r6 = 70 * (1.0 - comp_paid_pct/100.0) + 30 * min(comp_pending_families/300.0, 1.0)
    
    # 7. Rehabilitation & Resettlement Risk
    r7 = 70 * (1.0 - rr_pct/100.0) + 30 * min(rr_pending_families/300.0, 1.0)
    
    # 8. Possession Risk
    r8 = 80 * (1.0 - possession_pct/100.0) + 20 * (1.0 - doc_pct/100.0)
    
    # 9. Closure Risk
    r9 = 50 * (backlog/100.0) + 50 * (1.0 - possession_pct/100.0)
    
    stage_risks = {
        "Preliminary Identification": np.clip(r1, 5, 98),
        "Notification": np.clip(r2, 5, 98),
        "Survey & Documentation": np.clip(r3, 5, 98),
        "Objections / Legal Review": np.clip(r4, 5, 98),
        "Compensation Assessment": np.clip(r5, 5, 98),
        "Compensation Disbursement": np.clip(r6, 5, 98),
        "Rehabilitation & Resettlement": np.clip(r7, 5, 98),
        "Possession": np.clip(r8, 5, 98),
        "Closure": np.clip(r9, 5, 98)
    }
    
    # Round stages
    return {k: round(float(v)) for k, v in stage_risks.items()}
