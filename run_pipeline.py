import os
import sys
import yaml
import pandas as pd
sys.path.append('src')

from data_preprocessing import StrokeDataPreprocessor
from imbalanced_handler import ImbalancedDataHandler
from model_training import StrokeModelTrainer
# (Tạm thời chưa dùng model_evaluation để chạy cho mượt đã)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("🏥 STARTING STROKE PREDICTION PIPELINE...")
    
    # 1. Setup
    config = load_config()
    
    # 2. Preprocessing
    print("\n--- STEP 1: PREPROCESSING ---")
    preprocessor = StrokeDataPreprocessor(config)
    df = preprocessor.load_data(config['data']['raw_data_path'])
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.encode_categorical_features(df)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Lưu data đã xử lý
    os.makedirs('data/processed', exist_ok=True)
    X_test_scaled.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    # 3. Imbalanced Handling
    print("\n--- STEP 2: IMBALANCED HANDLING ---")
    handler = ImbalancedDataHandler(random_state=config['data']['random_state'])
    handler.analyze_imbalance(y_train)
    
    # Mặc định dùng SMOTE
    X_resampled, y_resampled = handler.apply_smote(X_train_scaled, y_train)
    
    # 4. Training
    print("\n--- STEP 3: TRAINING ---")
    trainer = StrokeModelTrainer(random_state=config['data']['random_state'])
    
    trainer.train_logistic_regression(X_resampled, y_resampled)
    trainer.train_random_forest(X_resampled, y_resampled)
    trainer.train_xgboost(X_resampled, y_resampled)
    trainer.train_lightgbm(X_resampled, y_resampled)
    
    trainer.save_models()
    
    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()