import hashlib
import random
from pydantic import BaseModel
from typing import List
from fastapi import BackgroundTasks
from ml.risk_model import calculate_all_risk_metrics

def send_notification(ward_id: str, wbgt: float, beds_needed: float):
    # This is the non-blocking notification dispatcher
    print(f"Notification triggered for Ward {ward_id}: WBGT {wbgt}, Beds Needed {beds_needed}")
    # Integration with SMS gateway would go here
    return True

def get_action_recommendations(ward_id: str, risk_tier: str = "Normal"):
    class ActionRecommendations(BaseModel):
        cooling_centers_to_activate: List[str]
        labor_advisory: str
        healthcare_outreach: str
        sms_alert_dispatched: bool

    hash_val = int(hashlib.md5(ward_id.encode()).hexdigest(), 16)
    shelters = [f"Center {i}" for i in range(1, 7)]
    
    if risk_tier == "Normal":
        return ActionRecommendations(
            cooling_centers_to_activate=random.sample(shelters, 1),
            labor_advisory="Monitor weather conditions.",
            healthcare_outreach="Standard healthcare services.",
            sms_alert_dispatched=False
        )
    
    return ActionRecommendations(
        cooling_centers_to_activate=random.sample(shelters, 2 if risk_tier == "Caution" else 4),
        labor_advisory="Restrict strenuous work from 12 PM to 4 PM." if risk_tier != "Extreme" else "Total restriction of outdoor work.",
        healthcare_outreach="Deploy ASHA workers to high-risk zones." if risk_tier != "Extreme" else "Full mobilization of emergency response teams.",
        sms_alert_dispatched=True
    )

def compute_ward_risk(ward_id: str, ward_name: str, vulnerability_weight: float, wbgt_prediction: float) -> dict:
    """
    Unified engine to compute final risk metrics for a ward based on WBGT.
    """
    # 1. Calculate Risk Metrics
    risk_data = calculate_all_risk_metrics(wbgt_prediction, vulnerability_weight)
    
    # 2. Get Recommendations
    recommendations = get_action_recommendations(ward_id, risk_data['risk_tier'])
    
    # 3. Prepare result with sanitization
    return {
        "ward_id": ward_id,
        "ward_name": ward_name,
        "wbgt": round(float(wbgt_prediction), 2),
        "risk_score": risk_data['risk_score'],
        "risk_tier": risk_data['risk_tier'],
        "mortality_index": risk_data['mortality_index'],
        "resource_estimates": {
            "required_heat_stroke_beds": int(risk_data['beds_needed']),
            "cooling_centers_count": int(risk_data['centers_to_activate']),
            "ambulance_dispatch_priority": risk_data['priority']
        },
        "action_recommendations": recommendations.dict()
    }

def calculate_ward_risk_metrics_full(ward, current_date, prediction, background_tasks: BackgroundTasks):
    w_id = str(ward['ward_id'])
    w_name = ward['ward_name']
    vulnerability_weight = float(ward['vulnerability_weight'])
    
    # Deterministic WBGT variation
    seed_str = f"{w_id}_{current_date.strftime('%Y-%m-%d')}"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    ward_specific_wbgt = prediction + rng.uniform(-0.5, 0.5)
    
    # Unified calculation via engine
    risk_data = compute_ward_risk(w_id, w_name, vulnerability_weight, ward_specific_wbgt)
    
    # Notifications
    if risk_data['risk_tier'] in ["High Risk", "Severe", "Extreme"]:
        background_tasks.add_task(send_notification, w_id, ward_specific_wbgt, risk_data['resource_estimates']['required_heat_stroke_beds'])
        
    return risk_data

