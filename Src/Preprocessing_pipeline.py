import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Union, List
import warnings
warnings.filterwarnings('ignore')


class PreprocessingPipeline:
    """
    Production preprocessing pipeline.
    
    Transforms raw sensor data into 18 features for model prediction.
    NO manual feature engineering needed in backend!
    """
    
    def __init__(self):
        """Initialize pipeline with feature list"""
        self.feature_columns = [
            # Sensor features (5)
            'Air_temperature',
            'Process_temperature',
            'Rotational_speed',
            'Torque',
            'Tool_wear',
            
            # Engineered features (5)
            'Temp_Difference',
            'Power',
            'Torque_Speed_Ratio',
            'Temp_Rate_of_Change',
            'RPM_Variance',
            
            # Datetime features (3)
            'month',
            'hour',
            'dayofweek',
            
            # Temporal features (2)
            'machine_age_hours',
            'hours_since_last',
            
            # Machine type (3)
            'Type_H',
            'Type_L',
            'Type_M'
        ]
        
        self.is_fitted = False
        
    def transform_single(self, data: Dict) -> np.ndarray:
        """
        Transform single sensor reading into 18 features.
        
        Args:
            data: Dictionary with sensor readings
                {
                    'datetime': '2025-01-20 14:30:00',
                    'Type': 'M',  # or 'H', 'L'
                    'Air_temperature': 300.0,
                    'Process_temperature': 310.0,
                    'Rotational_speed': 1480,
                    'Torque': 42.0,
                    'Tool_wear': 150,
                    'machine_age_hours': 15000,  # optional, default 10000
                    'hours_since_last': 8,       # optional, default 8
                    'Temp_Rate_of_Change': 0.15, # optional, default 0.0
                    'RPM_Variance': 35.0         # optional, default 20.0
                }
        
        Returns:
            numpy array (1, 18) ready for model prediction
        """
        
        # Extract raw sensor values
        air_temp = float(data.get('Air_temperature', 300.0))
        process_temp = float(data.get('Process_temperature', 310.0))
        rpm = float(data.get('Rotational_speed', 1500))
        torque = float(data.get('Torque', 40.0))
        tool_wear = float(data.get('Tool_wear', 100))
        
        # Calculate engineered features (EXACT dari FE.ipynb!)
        temp_difference = process_temp - air_temp
        power = torque * rpm / 9.5488
        torque_speed_ratio = torque / (rpm + 1)
        
        # Optional features (with defaults)
        temp_rate_of_change = float(data.get('Temp_Rate_of_Change', 0.0))
        rpm_variance = float(data.get('RPM_Variance', 20.0))
        
        # Extract datetime features
        dt_str = data.get('datetime', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        dt = pd.to_datetime(dt_str)
        month = dt.month
        hour = dt.hour
        dayofweek = dt.dayofweek
        
        # Temporal features
        machine_age_hours = float(data.get('machine_age_hours', 10000))
        hours_since_last = float(data.get('hours_since_last', 8))
        
        # Machine type encoding (one-hot)
        machine_type = data.get('Type', 'M')
        type_h = 1 if machine_type == 'H' else 0
        type_l = 1 if machine_type == 'L' else 0
        type_m = 1 if machine_type == 'M' else 0
        
        # Build feature vector (ORDER MATTERS!)
        features = [
            # Sensor (5)
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear,
            
            # Engineered (5)
            temp_difference,
            power,
            torque_speed_ratio,
            temp_rate_of_change,
            rpm_variance,
            
            # Datetime (3)
            month,
            hour,
            dayofweek,
            
            # Temporal (2)
            machine_age_hours,
            hours_since_last,
            
            # Machine type (3)
            type_h,
            type_l,
            type_m
        ]
        
        return np.array(features).reshape(1, -1)
    
    def transform_batch(self, data_list: List[Dict]) -> np.ndarray:
        """
        Transform multiple sensor readings.
        
        Args:
            data_list: List of dictionaries with sensor readings
            
        Returns:
            numpy array (n_samples, 18)
        """
        results = []
        for data in data_list:
            features = self.transform_single(data)
            results.append(features[0])
        
        return np.array(results)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform dataframe (for training).
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            numpy array of processed features
        """
        self.is_fitted = True
        
        # Convert dataframe to list of dicts
        data_list = df.to_dict('records')
        
        return self.transform_batch(data_list)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names in order.
        
        Returns:
            List of 18 feature names
        """
        return self.feature_columns.copy()
    
    def save(self, filepath: str):
        """
        Save pipeline to file.
        
        Args:
            filepath: Path to save pipeline, e.g. 'models/preprocessing_pipeline.pkl'
        """
        joblib.dump(self, filepath)
        print(f"✓ Preprocessing pipeline saved to: {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """
        Load pipeline from file.
        
        Args:
            filepath: Path to saved pipeline
            
        Returns:
            PreprocessingPipeline object
        """
        pipeline = joblib.load(filepath)
        print(f"✓ Preprocessing pipeline loaded from: {filepath}")
        return pipeline
    
    def validate_input(self, data: Dict) -> tuple:
        """
        Validate input data.
        
        Args:
            data: Input dictionary
            
        Returns:
            (is_valid, error_message)
        """
        required_fields = [
            'Air_temperature',
            'Process_temperature',
            'Rotational_speed',
            'Torque',
            'Tool_wear',
            'Type'
        ]
        
        # Check required fields
        missing = [f for f in required_fields if f not in data]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
        
        # Validate ranges
        try:
            air_temp = float(data['Air_temperature'])
            if not 200 <= air_temp <= 400:
                return False, f"Air_temperature out of range: {air_temp} (expected 200-400K)"
            
            process_temp = float(data['Process_temperature'])
            if not 200 <= process_temp <= 400:
                return False, f"Process_temperature out of range: {process_temp} (expected 200-400K)"
            
            rpm = float(data['Rotational_speed'])
            if not 1000 <= rpm <= 3000:
                return False, f"Rotational_speed out of range: {rpm} (expected 1000-3000 RPM)"
            
            torque = float(data['Torque'])
            if not 0 <= torque <= 100:
                return False, f"Torque out of range: {torque} (expected 0-100 Nm)"
            
            tool_wear = float(data['Tool_wear'])
            if not 0 <= tool_wear <= 300:
                return False, f"Tool_wear out of range: {tool_wear} (expected 0-300 min)"
            
            machine_type = data['Type']
            if machine_type not in ['H', 'M', 'L']:
                return False, f"Invalid Type: {machine_type} (expected H, M, or L)"
            
        except (ValueError, TypeError) as e:
            return False, f"Invalid data type: {str(e)}"
        
        return True, ""

if __name__ == "__main__":
    print("="*80)
    print("PREPROCESSING PIPELINE - DEMO & TESTING")
    print("="*80)
    
    # Create pipeline
    print("\n[1/5] Creating pipeline...")
    pipeline = PreprocessingPipeline()
    print("✓ Pipeline created")
    print(f"✓ Features: {len(pipeline.feature_columns)}")
    
    # Test data
    print("\n[2/5] Testing with sample data...")
    
    test_data = {
        'datetime': '2025-01-20 14:30:00',
        'Type': 'M',
        'Air_temperature': 300.0,
        'Process_temperature': 310.0,
        'Rotational_speed': 1480,
        'Torque': 42.0,
        'Tool_wear': 150,
        'machine_age_hours': 15000,
        'hours_since_last': 8,
        'Temp_Rate_of_Change': 0.15,
        'RPM_Variance': 35.0
    }
    
    print("\nINPUT DATA:")
    for key, value in test_data.items():
        print(f"   {key:25s} = {value}")
    
    # Validate
    print("\n[3/5] Validating input...")
    is_valid, error_msg = pipeline.validate_input(test_data)
    if is_valid:
        print("Input validation passed")
    else:
        print(f"Validation failed: {error_msg}")
    
    # Transform
    print("\n[4/5] Transforming...")
    features = pipeline.transform_single(test_data)
    
    print("\nOUTPUT FEATURES (18):")
    feature_names = pipeline.get_feature_names()
    for i, (name, value) in enumerate(zip(feature_names, features[0])):
        if i < 5:
            category = "SENSOR"
        elif i < 10:
            category = "ENGINEERED"
        elif i < 13:
            category = "DATETIME"
        elif i < 15:
            category = "TEMPORAL"
        else:
            category = "TYPE"
        print(f"   {i+1:2d}. {name:25s} = {value:>10.4f}  [{category}]")
    
    print(f"\n✓ Shape: {features.shape}")
    print(f"✓ Ready for model prediction!")
    
    # Save
    print("\n[5/5] Saving pipeline...")
    import os
    os.makedirs('Pipeline', exist_ok=True)
    pipeline.save('Pipeline/preprocessing_pipeline.pkl')
    
    # Test load
    print("\nTesting load...")
    loaded_pipeline = PreprocessingPipeline.load('Pipeline/preprocessing_pipeline.pkl')
    
    # Verify
    features_loaded = loaded_pipeline.transform_single(test_data)
    if np.allclose(features, features_loaded):
        print("Pipeline works correctly after load!")
    else:
        print("Error: Features don't match after load!")
 