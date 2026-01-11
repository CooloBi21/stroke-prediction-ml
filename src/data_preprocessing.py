"""
Module Preprocessing dữ liệu y tế - Stroke Prediction
Xử lý missing values, encoding, scaling
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import joblib
import os

class StrokeDataPreprocessor:
    """Class xử lý dữ liệu bệnh nhân đột quỵ"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.knn_imputer = KNNImputer(n_neighbors=config['preprocessing']['knn_neighbors'])
        
    def load_data(self, filepath):
        """Load dữ liệu từ CSV"""
        print("="*60)
        print("📂 ĐANG LOAD DỮ LIỆU...")
        print("="*60)
        
        df = pd.read_csv(filepath)
        print(f"\n✓ Số lượng bệnh nhân: {len(df)}")
        print(f"✓ Số lượng features: {df.shape[1]}")
        
        # Phân tích class imbalance
        stroke_counts = df['stroke'].value_counts()
        print(f"\n⚠️ PHÂN TÍCH MẤT CÂN BẰNG DỮ LIỆU:")
        print(f"  - Không đột quỵ (0): {stroke_counts[0]} ({stroke_counts[0]/len(df)*100:.2f}%)")
        print(f"  - Có đột quỵ (1): {stroke_counts[1]} ({stroke_counts[1]/len(df)*100:.2f}%)")
        print(f"  - Tỷ lệ imbalance: 1:{stroke_counts[0]/stroke_counts[1]:.1f}")
        
        return df
    
    def handle_missing_values(self, df):
        """Xử lý missing values bằng KNN Imputer"""
        print("\n" + "="*60)
        print("🔧 XỬ LÝ MISSING VALUES...")
        print("="*60)
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n📋 Missing values trước khi xử lý:")
            for col in missing[missing > 0].index:
                print(f"  - {col}: {missing[col]} ({missing[col]/len(df)*100:.2f}%)")
        
        # Xử lý BMI bằng KNN Imputer
        if df['bmi'].isnull().sum() > 0:
            print(f"\n🔄 Đang impute BMI bằng KNN (k={self.config['preprocessing']['knn_neighbors']})...")
            numeric_cols = ['age', 'avg_glucose_level', 'bmi']
            df[numeric_cols] = self.knn_imputer.fit_transform(df[numeric_cols])
            print(f"✓ Đã impute {missing['bmi']} giá trị BMI")
        
        # Xử lý smoking_status nếu có missing
        if 'smoking_status' in df.columns and df['smoking_status'].isnull().sum() > 0:
            df['smoking_status'].fillna('Unknown', inplace=True)
        
        print(f"\n✓ Hoàn thành! Tổng missing values còn lại: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical_features(self, df):
        """Encoding các features categorical"""
        print("\n" + "="*60)
        print("🏷️ ENCODING CATEGORICAL FEATURES...")
        print("="*60)
        
        df_encoded = df.copy()
        categorical_cols = ['gender', 'ever_married', 'work_type', 
                          'Residence_type', 'smoking_status']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                print(f"\n  Encoding {col}: {df_encoded[col].unique()}")
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
        
        print(f"\n✓ Đã encode {len(categorical_cols)} features")
        return df_encoded
    
    def prepare_data(self, df):
        """Chuẩn bị dữ liệu: train-test split và scaling"""
        print("\n" + "="*60)
        print("✂️ CHIA DỮ LIỆU TRAIN-TEST...")
        print("="*60)
        
        # Loại bỏ cột id
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Tách features và target
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        
        # Stratified split (quan trọng cho imbalanced data!)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        print(f"\n📊 Kết quả chia dữ liệu:")
        print(f"  - Training set: {len(X_train)} samples")
        print(f"  - Testing set: {len(X_test)} samples")
        print(f"\n  Phân bố class trong training set:")
        print(f"  - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
        print(f"  - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
        
        # Scaling features
        print(f"\n📏 CHUẨN HÓA FEATURES...")
        scale_cols = self.config['preprocessing']['scale_features']
        print(f"  Scaling columns: {scale_cols}")
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[scale_cols] = self.scaler.fit_transform(X_train[scale_cols])
        X_test_scaled[scale_cols] = self.scaler.transform(X_test[scale_cols])
        
        print(f"✓ Đã scale training và testing set")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Lưu preprocessor để sử dụng sau"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'knn_imputer': self.knn_imputer
        }
        joblib.dump(preprocessor_data, filepath)
        print(f"\n💾 Đã lưu preprocessor tại: {filepath}")

def main():
    """Demo preprocessing pipeline"""
    import yaml
    
    print("\n" + "🏥 "*20)
    print("STROKE PREDICTION - DATA PREPROCESSING")
    print("🏥 "*20)
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Khởi tạo preprocessor
    preprocessor = StrokeDataPreprocessor(config)
    
    # Load dữ liệu
    df = preprocessor.load_data(config['data']['raw_path'])
    
    # Xử lý missing values
    df = preprocessor.handle_missing_values(df)
    
    # Encoding categorical
    df = preprocessor.encode_categorical_features(df)
    
    # Chia train-test và scale
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Lưu dữ liệu đã xử lý
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("\n💾 Đã lưu dữ liệu processed:")
    print("  - data/processed/X_train.csv")
    print("  - data/processed/X_test.csv")
    print("  - data/processed/y_train.csv")
    print("  - data/processed/y_test.csv")
    
    # Lưu preprocessor
    preprocessor.save_preprocessor()
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH PREPROCESSING!")
    print("="*60)

if __name__ == "__main__":
    main()