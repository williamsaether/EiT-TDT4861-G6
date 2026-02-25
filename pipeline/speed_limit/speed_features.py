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
    Estimates urbanization (0.0 to 1.0).
    Higher value = higher density of pedestrians, driveways, and complexity.
    """
    # 1. Municipal roads (Highest urbanization)
    if road_class == "Kommunal vei":
        if speed_limit <= 40:
            return 1.0  # Typical residential area, many pedestrians / cyclists
        if speed_limit <= 60:
            return 0.7  # Collector road in city/town
        return 0.5      # Municipal rural road

    # 2. County roads (Medium/Varied)
    if road_class == "Fylkesvei":
        if speed_limit <= 50:
            return 0.8  # Through traffic in town/village
        if speed_limit <= 70:
            return 0.4  # Typical country road with scattered buildings
        return 0.2      # Higher standard county road

    # 3. National and European roads (Lowest urbanization/Most isolated)
    if road_class in ["Europavei", "Riksvei"]:
        if speed_limit <= 60:
            return 0.3  # Through traffic in urban-adjacent areas
        if speed_limit <= 80:
            return 0.1  # Standard country road/motor traffic road
        return 0.0      # Motorway (completely isolated from surroundings)

    # Default if road class is unknown
    return 0.4

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
        "speed_delta": round((fart - previous_speed_limit)/110, 3) if previous_speed_limit is not None else 0
    }
    
    return features