def generate_recommendations(top_drivers, project_row):
    """
    Generates actionable administrative recommendations based on the top risk-increasing drivers.
    """
    # Filter drivers that are actively increasing risk
    risk_drivers = [d for d in top_drivers if d.get("shap_val", 0) > 0 or d.get("direction") == "Increase Risk"]
    
    if not risk_drivers:
        return {
            "top_action": "No high-risk drivers detected. Continue standard monitoring and progress tracking.",
            "additional_actions": ["Maintain active communication between state and district departments."]
        }
        
    recommendations_pool = {
        "Compensation Paid %": "Prioritize disbursement of outstanding compensation. Establish local camp offices in the district to expedite pending family payments.",
        "Compensation Pending Families": "Establish local camp offices in the district to expedite pending family payments.",
        "Legal Disputes Count": "Engage district mediation officers to fast-track dispute resolutions and settle ownership conflicts outside of courts.",
        "Average Dispute Age (Days)": "Establish dedicated fast-track tribunals or legal committees for resolving long-standing land litigation cases.",
        "Administrative Approval Delay": "Escalate pending clearances to the State Coordination Committee. Conduct a joint department review to resolve documentation bottlenecks.",
        "Pending Approvals Count": "Deploy an inter-departmental task force to expedite clearance of outstanding administrative approvals.",
        "Documentation Completion %": "Deploy additional surveying teams to digitize local land records and complete pending land survey filings.",
        "Rehabilitation & Resettlement Progress": "Accelerate site allocations for resettlement and execute critical rehabilitation schemes before initiating possession.",
        "Rehabilitation Pending Families": "Prioritize rehabilitation site allocation and support packages for the pending affected families.",
        "Department Case Backlog": "Temporarily reallocate administrative staff to the district office to clear the backlog of pending land files.",
        "Stakeholder Response Time": "Setup a direct coordination cell with project stakeholders and local panchayats to reduce communication turnaround times.",
        "Land Ownership Conflicts": "Initiate joint boundary demarcation audits and verify revenue records to settle conflicting ownership claims.",
        "Possession Progress %": "Coordinate with local administrative officers to conduct the formal possession proceedings after completing compensation disbursements."
    }
    
    actions = []
    for driver in risk_drivers:
        feature_name = driver["feature"]
        
        # Check for keyword matches in cleaned feature name
        matched_action = None
        for key, rec_text in recommendations_pool.items():
            if key in feature_name:
                # Customize recommendation text if specific features have values we want to mention
                if key == "Compensation Paid %" and "compensation_paid_pct" in project_row:
                    val = project_row["compensation_paid_pct"]
                    matched_action = f"Compensation payout is lagging at {val}%. Prioritize disbursement of outstanding compensation and establish local camp offices to expedite pending payments."
                elif key == "Legal Disputes Count" and "legal_disputes" in project_row:
                    val = int(project_row["legal_disputes"])
                    matched_action = f"There are {val} active legal disputes. Engage district mediation officers to fast-track resolutions and settle ownership conflicts outside of courts."
                elif key == "Administrative Approval Delay" and "approval_delay_days" in project_row:
                    val = int(project_row["approval_delay_days"])
                    matched_action = f"Administrative approval delay has reached {val} days. Escalate pending clearances to the State Coordination Committee."
                elif key == "Documentation Completion %" and "documentation_completion_pct" in project_row:
                    val = project_row["documentation_completion_pct"]
                    matched_action = f"Documentation completion is low at {val}%. Deploy additional surveying teams to digitize local land records and complete land filings."
                elif key == "Rehabilitation & Resettlement Progress" and "rr_progress_pct" in project_row:
                    val = project_row["rr_progress_pct"]
                    matched_action = f"R&R progress is lagging at {val}%. Accelerate site allocations and execute critical rehabilitation schemes before initiating possession."
                else:
                    matched_action = rec_text
                break
                
        if matched_action:
            actions.append(matched_action)
            
    # Default if no specific match
    if not actions:
        actions = ["Prioritize resolving administrative disputes and coordinate between departments to speed up approvals."]
        
    return {
        "top_action": actions[0],
        "additional_actions": actions[1:] if len(actions) > 1 else ["Monitor progress weekly to prevent administrative bottlenecks."]
    }
