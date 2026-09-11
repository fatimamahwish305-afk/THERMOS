import math

def calculate_all_risk_metrics(ward_specific_wbgt, vulnerability_weight):
    # 1. Hazard Index & Risk Score
    hazard_index = min(50, max(0, (ward_specific_wbgt - 15) * 2))
    scaling_factor = 1.2
    risk_score = min(100, hazard_index * vulnerability_weight * scaling_factor)
    
    # 2. Mortality Index (Scale 0-100)
    mortality_index = min(100, max(0, (ward_specific_wbgt - 20) * 4 * vulnerability_weight))
    
    # 3. Resource Estimates (Risk Tiers - 4 Tiers)
    if risk_score < 25:
        risk_tier = "Normal"
        beds_needed = 0
        centers_to_activate = 1
        priority = "Low"
    elif risk_score < 50:
        risk_tier = "Caution"
        beds_needed = 10
        centers_to_activate = 2
        priority = "Medium"
    elif risk_score < 75:
        risk_tier = "High Risk"
        beds_needed = 50
        centers_to_activate = 3
        priority = "High"
    else:
        risk_tier = "Extreme"
        beds_needed = 150
        centers_to_activate = 6
        priority = "Emergency"
        
    return {
        "risk_score": round(float(risk_score), 2),
        "risk_tier": risk_tier,
        "mortality_index": round(float(mortality_index), 2),
        "beds_needed": int(beds_needed),
        "centers_to_activate": int(centers_to_activate),
        "priority": priority
    }
