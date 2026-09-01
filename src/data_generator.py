import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Definitions
PROJECT_TYPES = ["Highway", "Railway", "Metro", "Airport", "Power/Grid", "Irrigation", "Industrial Area", "Urban Development"]
STATES = ["Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Karnataka", "Gujarat"]
DISTRICTS = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Varanasi", "Agra"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Mangaluru", "Belagavi"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"]
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

DISTRICT_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Thane": (19.2183, 72.9781),
    "Nashik": (19.9975, 73.7898),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Noida": (28.5355, 77.3910),
    "Varanasi": (25.3176, 82.9739),
    "Agra": (27.1767, 78.0081),
    "Bengaluru": (12.9716, 77.5946),
    "Mysuru": (12.2958, 76.6394),
    "Hubballi": (15.3647, 75.1240),
    "Mangaluru": (12.9141, 74.8560),
    "Belagavi": (15.8497, 74.4977),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Rajkot": (22.3039, 70.8022),
    "Gandhinagar": (23.2156, 72.6369)
}

# Historical district delay rates
DISTRICT_DELAY_RATES = {}
for state, dists in DISTRICTS.items():
    for dist in dists:
        DISTRICT_DELAY_RATES[dist] = np.random.uniform(0.15, 0.55)

# Historical project type delay rates
PROJECT_TYPE_DELAY_RATES = {
    "Highway": 0.35,
    "Railway": 0.45,
    "Metro": 0.50,
    "Airport": 0.55,
    "Power/Grid": 0.20,
    "Irrigation": 0.40,
    "Industrial Area": 0.30,
    "Urban Development": 0.25
}

def generate_project_names(n, project_types, states, districts):
    names = []
    prefixes = ["NH", "SH", "Greenfield", "Dedicated", "Urban", "Expressway", "High-Speed", "Rural"]
    irrigation_prefixes = ["Canal Network", "Reservoir Link", "Lift Irrigation", "Dam Construction"]
    industrial_prefixes = ["SEZ Hub", "Industrial Corridor", "Smart City Node", "Tech Park Site"]
    
    for i in range(n):
        p_type = project_types[i]
        state = states[i]
        dist = districts[i]
        
        if p_type == "Highway":
            names.append(f"{np.random.choice(prefixes)} Route-{np.random.randint(10, 999)} ({dist} Segment)")
        elif p_type == "Railway":
            names.append(f"{dist} Rail Link & Yard Expansion")
        elif p_type == "Metro":
            names.append(f"{dist} Metro Phase {np.random.randint(1, 3)} - Line {np.random.choice(['A', 'B', 'C'])}")
        elif p_type == "Airport":
            names.append(f"International Airport Runway Phase {np.random.randint(1, 3)} - {dist}")
        elif p_type == "Irrigation":
            names.append(f"{np.random.choice(irrigation_prefixes)} - {dist}")
        elif p_type in ["Industrial Area", "Urban Development"]:
            names.append(f"{np.random.choice(industrial_prefixes)} ({dist})")
        else:
            names.append(f"{p_type} Development Project - {dist}")
    return names

def generate_base_data(num_records, is_historical=True):
    project_types = np.random.choice(PROJECT_TYPES, size=num_records)
    states = np.random.choice(STATES, size=num_records)
    districts = [np.random.choice(DISTRICTS[state]) for state in states]
    project_names = generate_project_names(num_records, project_types, states, districts)
    
    # Generate coordinates based on district
    latitudes = []
    longitudes = []
    for dist in districts:
        coords = DISTRICT_COORDS.get(dist, (20.0, 78.0))
        # Add random offset so multiple projects in same district don't overlap completely
        latitudes.append(coords[0] + np.random.uniform(-0.06, 0.06))
        longitudes.append(coords[1] + np.random.uniform(-0.06, 0.06))
        
    # Numeric variables
    land_area_acres = np.round(np.random.exponential(scale=200, size=num_records) + 20, 1)
    affected_families = np.round(land_area_acres * np.random.uniform(0.2, 1.8, size=num_records) + np.random.randint(2, 20, size=num_records)).astype(int)
    
    # Durations
    planned_duration_days = np.random.randint(180, 1500, size=num_records)
    
    # Generate delay rates
    hist_district_rate = np.array([DISTRICT_DELAY_RATES[d] for d in districts])
    hist_type_rate = np.array([PROJECT_TYPE_DELAY_RATES[t] for t in project_types])
    
    data = pd.DataFrame({
        "project_id": [f"PRJ-{i+1:04d}" if is_historical else f"PRJ-ACT-{i+1:04d}" for i in range(num_records)],
        "project_name": project_names,
        "project_type": project_types,
        "state": states,
        "district": districts,
        "latitude": latitudes,
        "longitude": longitudes,
        "land_area_acres": land_area_acres,
        "affected_families": affected_families,
        "planned_duration_days": planned_duration_days,
        "historical_district_delay_rate": hist_district_rate,
        "historical_project_type_delay_rate": hist_type_rate
    })
    
    return data

def generate_features_and_delay(df, is_historical=True):
    n = len(df)
    
    # For both historical and current projects, we generate them at a random stage of execution
    # to capture snapshots.
    stage_probs = [0.08, 0.12, 0.15, 0.15, 0.12, 0.15, 0.15, 0.08] # active stages (1-8)
    current_stages = np.random.choice(STAGES[:8], size=n, p=stage_probs)
    
    possession_pct = np.zeros(n)
    documentation_completion_pct = np.zeros(n)
    compensation_approved_pct = np.zeros(n)
    compensation_paid_pct = np.zeros(n)
    rr_progress_pct = np.zeros(n)
    notification_status = []
    current_elapsed_days = []
    approval_delay_days = []
    pending_approvals = []
    legal_disputes = []
    average_dispute_age_days = []
    ownership_conflicts = []
    stakeholder_response_days = []
    department_backlog = []
    compensation_pending_families = []
    rehabilitation_pending_families = []
    
    for i, stage in enumerate(current_stages):
        stage_idx = STAGES.index(stage)
        
        # Elapsed days proportional to stage index
        planned_days = df.loc[i, "planned_duration_days"]
        progress_ratio = (stage_idx + 1) / 9.0
        elapsed = int(planned_days * progress_ratio * np.random.uniform(0.8, 1.2))
        current_elapsed_days.append(elapsed)
        
        # Notification status
        if stage_idx == 0:
            notification_status.append("Pending")
        elif stage_idx == 1:
            notification_status.append("In Progress")
        else:
            notification_status.append("Completed")
            
        # Documentation
        if stage_idx <= 1:
            doc_pct = np.random.uniform(0, 30)
        elif stage_idx == 2:
            doc_pct = np.random.uniform(30, 85)
        else:
            doc_pct = np.random.uniform(85, 100)
        documentation_completion_pct[i] = np.round(doc_pct, 1)
        
        # Approvals and delays
        app_del = np.round(np.random.exponential(scale=35) * (stage_idx + 1) * 0.5)
        approval_delay_days.append(app_del)
        pending_apps = np.random.randint(0, 4) if stage_idx < 5 else 0
        pending_approvals.append(pending_apps)
        
        # Compensation approved
        if stage_idx < 4:
            comp_app = 0.0
        elif stage_idx == 4:
            comp_app = np.random.uniform(10, 90)
        else:
            comp_app = np.random.uniform(90, 100)
        compensation_approved_pct[i] = np.round(comp_app, 1)
        
        # Compensation paid
        if stage_idx < 5:
            comp_paid = 0.0
        elif stage_idx == 5:
            comp_paid = np.random.uniform(5, 80)
        else:
            comp_paid = np.random.uniform(80, 100)
        compensation_paid_pct[i] = np.round(comp_paid, 1)
        
        # R&R progress
        if stage_idx < 6:
            rr_prog = 0.0
        elif stage_idx == 6:
            rr_prog = np.random.uniform(10, 85)
        else:
            rr_prog = np.random.uniform(85, 100)
        rr_progress_pct[i] = np.round(rr_prog, 1)
        
        # Possession pct
        if stage_idx < 7:
            poss = 0.0
        elif stage_idx == 7:
            poss = np.random.uniform(10, 90)
        else:
            poss = 100.0
        possession_pct[i] = np.round(poss, 1)
        
        # Disputes
        disputes = np.random.poisson(lam=1.2) + (1 if stage_idx in [3, 4] else 0)
        legal_disputes.append(disputes)
        age = np.random.uniform(15, 365) if disputes > 0 else 0.0
        average_dispute_age_days.append(np.round(age, 1))
        
        # Conflicts
        ownership_conflicts.append(np.random.poisson(lam=1.8))
        
        # Department backlog & stakeholder response
        stakeholder_response_days.append(np.round(np.random.normal(loc=30, scale=12) + (app_del * 0.1)))
        department_backlog.append(np.random.randint(2, 80))
        
        # Pending families
        aff_families = df.loc[i, "affected_families"]
        comp_pending = int(aff_families * (1.0 - comp_paid / 100.0))
        rr_pending = int(aff_families * (1.0 - rr_prog / 100.0))
        
        compensation_pending_families.append(comp_pending)
        rehabilitation_pending_families.append(rr_pending)
        
    df["current_stage"] = current_stages
    df["notification_status"] = notification_status
    df["documentation_completion_pct"] = documentation_completion_pct
    df["approval_delay_days"] = approval_delay_days
    df["pending_approvals"] = pending_approvals
    df["compensation_approved_pct"] = compensation_approved_pct
    df["compensation_paid_pct"] = compensation_paid_pct
    df["compensation_pending_families"] = compensation_pending_families
    df["legal_disputes"] = legal_disputes
    df["average_dispute_age_days"] = average_dispute_age_days
    df["ownership_conflicts"] = ownership_conflicts
    df["rr_progress_pct"] = rr_progress_pct
    df["rehabilitation_pending_families"] = rehabilitation_pending_families
    df["possession_pct"] = possession_pct
    df["stakeholder_response_days"] = np.clip(stakeholder_response_days, 5, 150)
    df["department_backlog"] = department_backlog
    df["current_elapsed_days"] = current_elapsed_days
    
    # Calculate true delay probability index
    # Note: higher values of these features should correlate with delay
    risk_score = (
        0.18 * np.clip(df["approval_delay_days"].values / 120.0, 0, 1) +
        0.18 * np.clip(df["legal_disputes"].values / 10.0, 0, 1) +
        0.18 * (1.0 - df["compensation_paid_pct"].values / 100.0) +
        0.12 * (1.0 - df["documentation_completion_pct"].values / 100.0) +
        0.10 * (1.0 - df["rr_progress_pct"].values / 100.0) +
        0.08 * np.clip(df["department_backlog"].values / 80.0, 0, 1) +
        0.06 * np.clip(df["stakeholder_response_days"].values / 60.0, 0, 1) +
        0.05 * df["historical_district_delay_rate"].values +
        0.05 * df["historical_project_type_delay_rate"].values
    )
    
    if is_historical:
        # Generate target variables for historical training set
        # Add random normal noise to make it realistic
        noise = np.random.normal(0, 0.08, size=n)
        prob_delay = risk_score + noise
        
        # 1 if prob_delay > threshold, else 0
        # Set threshold to get ~40% delay rate
        delayed = (prob_delay > 0.40).astype(int)
        df["delayed"] = delayed
        
        # Actual delay days (0 if not delayed)
        actual_delay_days = np.zeros(n)
        delayed_indices = delayed == 1
        num_delayed = np.sum(delayed_indices)
        
        # Delay days proportional to risk factors + exponential noise
        base_delay = df.loc[delayed_indices, "planned_duration_days"].values * 0.25
        extra_delay = risk_score[delayed_indices] * 350 + np.random.exponential(scale=60, size=num_delayed)
        actual_delay_days[delayed_indices] = np.round(base_delay + extra_delay).astype(int)
        
        df["actual_delay_days"] = actual_delay_days
        
        # Compute dates
        start_dates = [datetime(2021, 1, 1) + timedelta(days=np.random.randint(0, 730)) for _ in range(n)]
        planned_completion_dates = [start_dates[i] + timedelta(days=int(df.loc[i, "planned_duration_days"])) for i in range(n)]
        actual_completion_dates = [planned_completion_dates[i] + timedelta(days=int(df.loc[i, "actual_delay_days"])) for i in range(n)]
        
        df["planned_completion_date"] = [d.strftime("%Y-%m-%d") for d in planned_completion_dates]
        df["actual_completion_date"] = [d.strftime("%Y-%m-%d") for d in actual_completion_dates]
    else:
        # For current/active projects, actual completion is unknown, delayed status is unknown
        df["delayed"] = np.nan
        df["actual_delay_days"] = np.nan
        
        # Approximate start date
        start_dates = [datetime(2025, 1, 1) - timedelta(days=int(df.loc[i, "current_elapsed_days"])) for i in range(n)]
        planned_completion_dates = [start_dates[i] + timedelta(days=int(df.loc[i, "planned_duration_days"])) for i in range(n)]
        
        df["planned_completion_date"] = [d.strftime("%Y-%m-%d") for d in planned_completion_dates]
        df["actual_completion_date"] = np.nan
        
    return df

def generate_and_save_data():
    os.makedirs("data", exist_ok=True)
    
    print("Generating historical projects (5,000 records)...")
    hist_base = generate_base_data(5000, is_historical=True)
    hist_full = generate_features_and_delay(hist_base, is_historical=True)
    hist_full.to_csv("data/historical_projects.csv", index=False)
    
    print("Generating current active projects (500 records)...")
    curr_base = generate_base_data(500, is_historical=False)
    curr_full = generate_features_and_delay(curr_base, is_historical=False)
    curr_full.to_csv("data/current_projects.csv", index=False)
    
    print("Datasets successfully written to data/ folder.")

if __name__ == "__main__":
    generate_and_save_data()
