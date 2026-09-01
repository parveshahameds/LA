import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from predict import predict_project_risk

def simulate_intervention(project_row, modifications, classifier, regressor, preprocessor):
    """
    Takes a single project row (dict or pandas Series), applies the modifications,
    runs model inference, and returns (new_probability, new_expected_delay_months).
    """
    # Convert series or dict to dict
    modified_dict = dict(project_row).copy()
    
    # Apply modifications
    for k, v in modifications.items():
        if k in modified_dict:
            modified_dict[k] = v
            
    # Adjust dependent variables to maintain correlation consistency
    if "compensation_paid_pct" in modifications and "affected_families" in modified_dict:
        paid_pct = modifications["compensation_paid_pct"]
        affected = modified_dict["affected_families"]
        modified_dict["compensation_pending_families"] = int(affected * (1.0 - paid_pct / 100.0))
        
    if "rr_progress_pct" in modifications and "affected_families" in modified_dict:
        rr_pct = modifications["rr_progress_pct"]
        affected = modified_dict["affected_families"]
        modified_dict["rehabilitation_pending_families"] = int(affected * (1.0 - rr_pct / 100.0))
        
    if "legal_disputes" in modifications:
        disputes = modifications["legal_disputes"]
        if disputes == 0:
            modified_dict["average_dispute_age_days"] = 0.0
            
    # Convert to DataFrame for inference
    df_instance = pd.DataFrame([modified_dict])
    
    # Run prediction
    pred_df = predict_project_risk(df_instance, classifier, regressor, preprocessor)
    
    new_prob = pred_df.loc[0, "delay_probability"]
    new_delay_months = pred_df.loc[0, "expected_delay_months"]
    
    return new_prob, new_delay_months
