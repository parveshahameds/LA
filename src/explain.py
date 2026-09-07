import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, get_feature_names

# Map raw feature names to clean public-facing names
FEATURE_LABEL_MAP = {
    "compensation_pct": "Compensation Disbursed %",
    "compensation_delay_days": "Compensation Delay (Days)",
    "legal_cases": "Active Legal Cases",
    "court_stay": "Active Court Stay Injunction",
    "approval_delay_days": "Administrative Approval Delay (Days)",
    "pending_approvals": "Pending Approvals Count",
    "documentation_pct": "Documentation Completion %",
    "rr_progress_pct": "Rehabilitation & Resettlement Progress %",
    "possession_pct": "Land Possession Handover %",
    "stakeholder_delay_days": "Stakeholder Delay (Days)",
    "historical_delay_rate": "Historical District Delay Rate",
    "affected_families": "Affected Families Count",
    "land_area_ha": "Land Acquisition Area (Hectares)",
    "planned_days": "Planned Project Duration (Days)",
    "compensation_paid_pct": "Compensation Paid %",
    "legal_disputes": "Legal Disputes Count",
    "documentation_completion_pct": "Documentation Completion %",
    "land_area_acres": "Land Area (Acres)",
    "planned_duration_days": "Planned Project Duration"
}

def clean_feature_name_and_value(feat_name, val):
    """
    Cleans up a feature name (including one-hot categories) and values for display.
    """
    # Check if this is a one-hot encoded category (e.g. project_type_Highway)
    for cat in CATEGORICAL_FEATURES:
        prefix = f"cat__{cat}_"
        if feat_name.startswith(prefix):
            category_val = feat_name.replace(prefix, "")
            clean_cat_name = cat.replace("_", " ").title()
            return f"{clean_cat_name}: {category_val}", "Yes" if val == 1 else "No"
            
    # Check if numerical feature has prefix from ColumnTransformer
    if feat_name.startswith("num__"):
        feat_name = feat_name.replace("num__", "")
        
    label = FEATURE_LABEL_MAP.get(feat_name, feat_name.replace("_", " ").title())
    
    # Format values nicely
    if "pct" in feat_name:
        return label, f"{val:.1f}%"
    elif "days" in feat_name or "age" in feat_name:
        return label, f"{int(val)} days"
    elif "acres" in feat_name:
        return label, f"{val:.1f} acres"
    elif "families" in feat_name:
        return label, f"{int(val)} families"
    else:
        # Check if it's float or int
        if isinstance(val, float):
            return label, f"{val:.2f}"
        return label, str(val)

def explain_prediction(project_row, classifier, preprocessor, historical_df):
    """
    Explains the delay prediction for a single project.
    Tries to use SHAP, and falls back to feature contribution method if SHAP is slow/fails.
    """
    # Prepare the single row DataFrame
    row_df = pd.DataFrame([project_row])
    # Extract raw features matching input features
    features_df = row_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    
    # Process
    X_processed = preprocessor.transform(features_df)
    processed_feature_names = get_feature_names(preprocessor)
    
    # Try using SHAP first
    try:
        import shap
        # RandomForestClassifier has a fast TreeExplainer
        explainer = shap.TreeExplainer(classifier)
        
        # shap_values returns a list for multiclass, or 3D array in new shap versions.
        # For RF binary classifier, index 1 is the positive class (delayed).
        shap_vals = explainer.shap_values(X_processed)
        
        # Handle different output shapes of shap_values
        if isinstance(shap_vals, list):
            # Old SHAP API returns a list of arrays [class_0_vals, class_1_vals]
            instance_shap = shap_vals[1][0]
        elif len(shap_vals.shape) == 3:
            # Shape is (samples, features, classes)
            instance_shap = shap_vals[0, :, 1]
        else:
            # shape is (samples, features) if it's regression or single-output
            instance_shap = shap_vals[0]
            
        contributions = []
        for name, s_val, raw_val in zip(processed_feature_names, instance_shap, X_processed[0]):
            clean_name, display_val = clean_feature_name_and_value(name, raw_val)
            contributions.append({
                "feature": clean_name,
                "display_value": display_val,
                "raw_val": raw_val,
                "shap_val": s_val
            })
            
    except Exception as e:
        # Fallback to feature importance approximation
        print(f"SHAP explanation failed, executing fallback method. Details: {e}")
        
        # Compute baseline mean values of processed features from historical dataset
        # In a real environment, we'd cache these baseline stats
        X_hist = historical_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        X_hist_processed = preprocessor.transform(X_hist)
        baseline_means = np.mean(X_hist_processed, axis=0)
        
        importances = classifier.feature_importances_
        
        # Local contribution = (value - mean) * importance
        # This highlights features that deviate from typical values, weighted by their predictive power
        deviations = X_processed[0] - baseline_means
        local_contributions = deviations * importances
        
        contributions = []
        for name, s_val, raw_val in zip(processed_feature_names, local_contributions, X_processed[0]):
            clean_name, display_val = clean_feature_name_and_value(name, raw_val)
            contributions.append({
                "feature": clean_name,
                "display_value": display_val,
                "raw_val": raw_val,
                "shap_val": s_val
            })
            
    # Sort contributions by absolute impact
    contributions = sorted(contributions, key=lambda x: abs(x["shap_val"]), reverse=True)
    
    # Filter to show only meaningful contributors (non-zero)
    contributions = [c for c in contributions if abs(c["shap_val"]) > 0.0001]
    
    # Get top 5 drivers
    top_drivers = contributions[:6]
    
    # Standardize/scale shap_vals to look nice as percentages (relative to the sum of absolute values)
    total_abs_shap = sum(abs(c["shap_val"]) for c in top_drivers)
    if total_abs_shap > 0:
        for c in top_drivers:
            # relative importance in top 5
            c["impact_pct"] = round((abs(c["shap_val"]) / total_abs_shap) * 100)
            c["direction"] = "Increase Risk" if c["shap_val"] > 0 else "Reduce Risk"
    else:
        for c in top_drivers:
            c["impact_pct"] = 0
            c["direction"] = "Neutral"
            
    return top_drivers
