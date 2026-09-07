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
    "Compensation Disbursed %": "Prioritize disbursement of outstanding compensation. Establish local camp offices in the district to expedite pending family payments.",
    "Compensation Paid %": "Prioritize disbursement of outstanding compensation. Establish local camp offices in the district to expedite pending family payments.",
    "Compensation Delay (Days)": "Expedite award inquiry and fund transfer to escrow accounts to eliminate compensation disbursement delays.",
    "Active Court Stay Injunction": "URGENT: File an expedited petition in the High Court for vacating stay orders and engage Special Government Pleaders.",
    "Active Legal Cases": "Engage district mediation officers to fast-track dispute resolutions and settle ownership conflicts outside of courts.",
    "Legal Disputes Count": "Engage district mediation officers to fast-track dispute resolutions and settle ownership conflicts outside of courts.",
    "Administrative Approval Delay": "Escalate pending clearances to the State Coordination Committee. Conduct a joint department review to resolve documentation bottlenecks.",
    "Pending Approvals Count": "Deploy an inter-departmental task force to expedite clearance of outstanding administrative approvals.",
    "Documentation Completion %": "Deploy additional surveying teams to digitize local land records and complete pending land survey filings.",
    "Rehabilitation & Resettlement Progress": "Accelerate site allocations for resettlement and execute critical rehabilitation schemes before initiating possession.",
    "Land Possession Handover %": "Coordinate with district revenue authorities and law enforcement to complete formal possession proceedings.",
    "Possession Progress %": "Coordinate with local administrative officers to conduct the formal possession proceedings after completing compensation disbursements.",
    "Stakeholder Delay (Days)": "Setup a direct coordination cell with project implementing agencies to streamline stakeholder reviews."
}

    actions = []
    for driver in risk_drivers:
        feature_name = driver["feature"]
        
        # Check for keyword matches in cleaned feature name
        matched_action = None
        for key, rec_text in recommendations_pool.items():
            if key in feature_name:
                if "Court Stay" in key:
                    matched_action = "URGENT: Court stay injunction is active. Mobilize legal team to file for vacation of stay in High Court."
                elif "Legal" in key and ("legal_cases" in project_row or "legal_disputes" in project_row):
                    val = int(project_row.get("legal_cases", project_row.get("legal_disputes", 1)))
                    matched_action = f"There are {val} active legal cases. Convene district fast-track mediation to resolve disputes outside litigation."
                elif "Compensation Delay" in key and "compensation_delay_days" in project_row:
                    val = int(project_row["compensation_delay_days"])
                    matched_action = f"Compensation disbursement is delayed by {val} days. Issue treasury clearance directives immediately."
                elif "Compensation" in key and ("compensation_pct" in project_row or "compensation_paid_pct" in project_row):
                    val = project_row.get("compensation_pct", project_row.get("compensation_paid_pct", 50))
                    matched_action = f"Compensation payout is lagging at {val}%. Prioritize fund disbursements to eliminate landowner opposition."
                elif "Approval" in key and "approval_delay_days" in project_row:
                    val = int(project_row["approval_delay_days"])
                    matched_action = f"Administrative approvals delayed by {val} days. Schedule urgent State Clearance Committee hearing."
                elif "Documentation" in key and ("documentation_pct" in project_row or "documentation_completion_pct" in project_row):
                    val = project_row.get("documentation_pct", project_row.get("documentation_completion_pct", 50))
                    matched_action = f"Documentation completion is at {val}%. Deploy revenue survey teams to expedite land titling."
                elif "Rehabilitation" in key and "rr_progress_pct" in project_row:
                    val = project_row["rr_progress_pct"]
                    matched_action = f"R&R progress is at {val}%. Accelerate resettlement plot allocations before demanding land possession."
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
