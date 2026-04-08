import time
from nvdb_speed import get_speed_limit_data
import speed_features # Our new file

class SpeedController:
    def __init__(self):
        self.last_speed_limit = None
        self.last_road_id = None

    def get_ml_input_vector(self, lat, lon):
        # 1. Fetch raw data from API
        raw_data = get_speed_limit_data(lat, lon)
        
        if raw_data and raw_data["status"] == "ok":
            # 2. Engineer features
            engineered = speed_features.engineer_all_features(
                raw_data, 
                previous_speed_limit=self.last_speed_limit
            )
            
            # 3. Update state for next delta calculation
            self.last_speed_limit = raw_data["fartsgrense"]
            
            # This 'engineered' dict can now be converted to a list/tensor for your ML model
            return engineered
        
        return None

# Example usage
controller = SpeedController()
features2 = controller.get_ml_input_vector(63.326244, 10.334259)
features1 = controller.get_ml_input_vector(59.833322, 10.410803)
print(f"E6 features: {features2}")
print(f"Gamle Drammensvei features: {features1}")