# speed_features.py

def get_road_class(vei_string):
    """Extracts the road category from strings like 'EV6 S1D1'"""
    if not vei_string or len(vei_string) < 2:
        return "Unknown"
    
    prefix = vei_string[:2].upper()
    mapping = {
        "EV": "Europavei",
        "RV": "Riksvei",
        "FV": "Fylkesvei",
        "KV": "Kommunal vei",
        "PV": "Privat vei"
    }
    return mapping.get(prefix, "Other")

def one_hot_encode_road(road_class):
    """Converts road class to a list/vector for ML"""
    classes = ["Europavei", "Riksvei", "Fylkesvei", "Kommunal vei", "Other"]
    return [1 if road_class == c else 0 for c in classes]

def calculate_urbanization(road_class, speed_limit):
    """
    Estimates urbanization (0.0 to 1.0)
    High value = Urban (many pedestrians), Low value = Rural/Highway
    """
    # Logic: Low speed limits on Municipal/County roads = Very Urban
    if road_class == "Kommunal vei" and speed_limit <= 40:
        return 1.0
    if road_class == "Fylkesvei" and speed_limit <= 50:
        return 0.7
    if road_class in ["Europavei", "Riksvei"] and speed_limit >= 80:
        return 0.1
    return 0.4 # Default middle ground

def engineer_all_features(api_data, previous_speed_limit=None):
    """
    The main function your controller will call.
    Transforms raw API dict into ML-ready features.
    """
    if not api_data or api_data.get("status") != "ok":
        return None

    fart = api_data["fartsgrense"]
    vei = api_data["vei"]
    
    road_class = get_road_class(vei)
    
    features = {
        # 1. Normalization (0.0 - 1.0)
        "norm_speed_limit": round(fart / 110, 3),
        
        # 2. Categorization (One-Hot)
        "road_type_vec": one_hot_encode_road(road_class),
        
        # 3. Urbanization Index
        "urbanization_idx": calculate_urbanization(road_class, fart),
        
        # 4. Speed Delta (If we have history)
        "speed_delta": (fart - previous_speed_limit) if previous_speed_limit is not None else 0
    }
    
    return features