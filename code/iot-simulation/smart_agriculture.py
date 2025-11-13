AI-Driven IoT Smart Agriculture System
Simulating crop yield prediction using sensor data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json

class SmartAgricultureSystem:
    def __init__(self):
        self.sensors = {
            'soil_moisture': {'unit': '%', 'range': (0, 100)},
            'temperature': {'unit': '°C', 'range': (-10, 45)},
            'humidity': {'unit': '%', 'range': (0, 100)},
            'ph_level': {'unit': 'pH', 'range': (3, 9)},
            'nitrogen': {'unit': 'ppm', 'range': (0, 100)},
            'phosphorus': {'unit': 'ppm', 'range': (0, 50)},
            'potassium': {'unit': 'ppm', 'range': (0, 50)},
            'light_intensity': {'unit': 'lux', 'range': (0, 100000)}
        }
        self.model = None
        self.sensor_data = None
        
    def generate_sensor_data(self, num_samples=1000):
        """Generate synthetic sensor data for simulation"""
        np.random.seed(42)
        
        data = {}
        for sensor, specs in self.sensors.items():
            min_val, max_val = specs['range']
            data[sensor] = np.random.uniform(min_val, max_val, num_samples)
        
        # Create realistic correlations
        data['soil_moisture'] = np.clip(data['soil_moisture'] + 
                                       data['humidity'] * 0.1 - 
                                       data['temperature'] * 0.5, 0, 100)
        
        # Generate crop yield based on sensor data (target variable)
        data['crop_yield'] = (
            data['soil_moisture'] * 0.3 +
            np.clip(data['temperature'], 15, 35) * 0.2 +
            data['humidity'] * 0.1 +
            np.clip(data['ph_level'], 5.5, 7.5) * 0.15 +
            data['nitrogen'] * 0.1 +
            data['phosphorus'] * 0.08 +
            data['potassium'] * 0.07 +
            np.random.normal(0, 5, num_samples)  # Noise
        )
        
        self.sensor_data = pd.DataFrame(data)
        return self.sensor_data
    
    def train_yield_prediction_model(self):
        """Train AI model to predict crop yield"""
        if self.sensor_data is None:
            self.generate_sensor_data()
        
        X = self.sensor_data.drop('crop_yield', axis=1)
        y = self.sensor_data['crop_yield']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        return self.model, mae, r2
    
    def predict_optimal_conditions(self):
        """Find optimal sensor readings for maximum yield"""
        feature_importance = pd.DataFrame({
            'feature': list(self.sensors.keys()),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance for Crop Yield:")
        print(feature_importance)
        
        # Simulate optimal conditions
        optimal_conditions = {}
        for sensor in self.sensors:
            if sensor in ['temperature', 'ph_level']:
                # These typically have optimal ranges
                if sensor == 'temperature':
                    optimal_conditions[sensor] = 25.0  # Optimal temperature
                elif sensor == 'ph_level':
                    optimal_conditions[sensor] = 6.5   # Optimal pH
            else:
                # Higher values generally better (within reason)
                optimal_conditions[sensor] = self.sensors[sensor]['range'][1] * 0.8
        
        return optimal_conditions, feature_importance
    
    def create_data_flow_diagram(self):
        """Generate description of data flow diagram"""
        diagram_description = """
        SMART AGRICULTURE DATA FLOW DIAGRAM:
        
        [SENSOR LAYER]
        │
        ├─ Soil Moisture Sensor → Data
        ├─ Temperature Sensor → Data  
        ├─ Humidity Sensor → Data
        ├─ pH Sensor → Data
        ├─ NPK Sensors → Data
        └─ Light Sensor → Data
        │
        [EDGE PROCESSING LAYER]
        │
        ├─ Data Collection & Preprocessing
        ├─ Local AI Model (Yield Prediction)
        ├─ Real-time Decision Making
        └─ Actuator Control (Irrigation, etc.)
        │
        [CLOUD LAYER] (Optional)
        │
        ├─ Historical Data Storage
        ├─ Model Retraining
        ├─ Analytics Dashboard
        └─ Remote Monitoring
        
        [ACTUATOR LAYER]
        │
        ├─ Smart Irrigation System
        ├─ Fertilizer Dispensers
        ├─ Climate Control
        └─ Alert System
        """
        return diagram_description

def main():
    # Initialize smart agriculture system
    ag_system = SmartAgricultureSystem()
    
    print("=== AI-Driven IoT Smart Agriculture System ===\n")
    
    # Display sensor information
    print("SENSORS DEPLOYED:")
    for sensor, specs in ag_system.sensors.items():
        print(f"- {sensor.replace('_', ' ').title()}: {specs['range'][0]} to {specs['range'][1]} {specs['unit']}")
    
    # Generate and display sample data
    print("\nGENERATING SENSOR DATA...")
    sensor_data = ag_system.generate_sensor_data()
    print(f"Generated {len(sensor_data)} samples of sensor data")
    print("\nSample Data:")
    print(sensor_data.head())
    
    # Train AI model
    print("\nTRAINING CROP YIELD PREDICTION MODEL...")
    model, mae, r2 = ag_system.train_yield_prediction_model()
    
    # Find optimal conditions
    print("\nANALYZING OPTIMAL GROWING CONDITIONS...")
    optimal_conditions, feature_importance = ag_system.predict_optimal_conditions()
    
    print("\nOPTIMAL SENSOR READINGS FOR MAXIMUM YIELD:")
    for sensor, value in optimal_conditions.items():
        unit = ag_system.sensors[sensor]['unit']
        print(f"- {sensor.replace('_', ' ').title()}: {value:.1f} {unit}")
    
    # Display data flow
    print("\n" + "="*50)
    print(ag_system.create_data_flow_diagram())
    
    # Generate insights
    print("\nKEY INSIGHTS:")
    print("1. Real-time monitoring enables precision agriculture")
    print("2. AI predictions help optimize resource usage")
    print("3. Edge processing reduces cloud dependency")
    print("4. Predictive maintenance for agricultural equipment")
    print("5. Data-driven decision making for farmers")

if __name__ == "__main__":
    main()
