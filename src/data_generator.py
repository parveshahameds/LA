import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Definitions matching the PDF dataset
PROJECT_TYPES = ["Highway", "Railway", "Metro", "Expressway", "Industrial Corridor", "Urban Development", "Power/Grid"]
STATES = ["Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Karnataka", "Odisha", "Kerala", "Bihar", "Madhya Pradesh", "Telangana", "Andhra Pradesh"]

DISTRICT_COORDS = {
    "Nayagarh": (20.1250, 85.1050),
    "Khordha": (20.1815, 85.6212),
    "Thrissur": (10.5276, 76.2144),
    "Ernakulam": (9.9816, 76.2999),
    "Araria": (26.1492, 87.4988),
    "Supaul": (26.1260, 86.6050),
    "Nashik": (19.9975, 73.7898),
    "Pune": (18.5204, 73.8567),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Salem": (11.6643, 78.1460),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Chennai": (13.0827, 80.2707),
    "Krishna": (16.1809, 81.1303),
    "Guntur": (16.3067, 80.4365),
    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9689, 79.5941),
    "Belagavi": (15.8497, 74.4977),
    "Bengaluru Rural": (13.2274, 77.5878),
    "Bengaluru": (12.9716, 77.5946),
    "Mysuru": (12.2958, 76.6394),
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Mumbai": (19.0760, 72.8777),
    "Nagpur": (21.1458, 79.0882),
    "Thane": (19.2183, 72.9781),
    "Noida": (28.5355, 77.3910),
    "Varanasi": (25.3176, 82.9739),
    "Agra": (27.1767, 78.0081)
}

STAGES = [
    "Preliminary Identification",
    "Notification",
    "Survey & Documentation",
    "Objections / Legal Review",
    "Compensation Assessment",
    "Compensation Disbursement",
    "Rehabilitation & Resettlement",
    "Possession",
    "Closure"
]

def load_or_create_pdf_dataset():
    pdf_path = "data/pdf_dataset_200.csv"
    if os.path.exists(pdf_path):
        return pd.read_csv(pdf_path)
    else:
        raise FileNotFoundError(f"{pdf_path} not found.")

def generate_and_save_data():
    os.makedirs("data", exist_ok=True)
    
    # Load exact 200 records from the PDF
    pdf_df = load_or_create_pdf_dataset()
    
    # Generate coordinates and project metadata
    latitudes = []
    longitudes = []
    project_names = []
    current_stages = []
    notification_statuses = []
    
    for idx, row in pdf_df.iterrows():
        dist = row["district"]
        state = row["state"]
        ptype = row["project_type"]
        pid = row["project_id"]
        
        coords = DISTRICT_COORDS.get(dist, (20.0, 78.0))
        # Deterministic coordinate offset so multiple points in same district don't overlap
        h = abs(hash(pid))
        lat_offset = ((h % 100) - 50) / 1000.0
        lon_offset = (((h // 100) % 100) - 50) / 1000.0
        latitudes.append(coords[0] + lat_offset)
        longitudes.append(coords[1] + lon_offset)
        
        project_names.append(f"{dist} {ptype} Corridor ({pid})")
        
        possession = row["possession_pct"]
        rr = row["rr_progress_pct"]
        comp = row["compensation_pct"]
        doc = row["documentation_pct"]
        
        if possession >= 90:
            stage = "Possession"
        elif rr >= 70:
            stage = "Rehabilitation & Resettlement"
        elif comp >= 75:
            stage = "Compensation Disbursement"
        elif doc >= 80:
            stage = "Survey & Documentation"
        elif row.get("pending_approvals", 0) > 0:
            stage = "Notification"
        else:
            stage = "Preliminary Identification"
        current_stages.append(stage)
        
        if doc >= 75:
            notification_statuses.append("Section 19 (Declaration) Issued")
        elif doc >= 40:
            notification_statuses.append("Section 11 (Preliminary) Issued")
        else:
            notification_statuses.append("Under Survey / Preliminary Review")
            
    pdf_df["latitude"] = latitudes
    pdf_df["longitude"] = longitudes
    pdf_df["project_name"] = project_names
    pdf_df["current_stage"] = current_stages
    pdf_df["notification_status"] = notification_statuses
    pdf_df["current_elapsed_days"] = np.clip(np.round(pdf_df["planned_days"] * 0.65).astype(int), 30, None)
    
    # Save both historical (training) and current (active monitoring) datasets
    pdf_df.to_csv("data/historical_projects.csv", index=False)
    pdf_df.to_csv("data/current_projects.csv", index=False)
    print("Historical and current project datasets successfully updated from the 200-row PDF dataset.")

if __name__ == "__main__":
    generate_and_save_data()
