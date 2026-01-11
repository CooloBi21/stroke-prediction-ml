"""
Script chạy toàn bộ pipeline ML
Từ preprocessing → imbalanced handling → training → evaluation
"""
import os
import sys
import yaml
import time
from datetime import datetime

# Thêm src vào path
sys.path.append('src')

from data_preprocessing import StrokeDataPreprocessor
from imbalanced_handler import ImbalancedDataHandler
from model_training import StrokeModelTrainer
from model_evaluation import ModelEvaluator

def create_directories():
    """Tạo các thư mục cần thiết"""
    directories = [
        'data/raw',
        'data/processed',
        'data/processed/resampled',
        'models',
        'results'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def print_header(text):
    """In header đẹp"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_step(step_num, step_name):
    """In bước thực hiện"""
    print(f"\n{'🔹'*30}")
    print(f"  BƯỚC {step_num}: {step_name}")
    print(f"{'🔹'*30}\n")

def main():
    """Main pipeline execution"""
    start_time = time.time()
    
    print("\n" + "🏥 "*40)
    print(" "*20 + "STROKE PREDICTION ML PIPELINE")
    print(" "*20 + f"Bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🏥 "*40)
    
    # ========================================
    # SETUP
    # ========================================
    print_header("SETUP & INITIALIZATION")
    
    print("📁 Tạo thư mục cần thiết...")
    create_directories()
    
    print("⚙️ Load cấu hình...")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Kiểm tra dataset
    data_path = config['data']['raw_path']
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Dataset không tìm thấy tại {data_path}")
        print("📥 Vui lòng download dataset từ Kaggle:")
        print("   https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
        sys.exit(1)
    
    print("✓ Setup hoàn tất!")
    
    # ========================================
    # STEP 1: DATA PREPROCESSING
    # ========================================
    print_step(1, "DATA PREPROCESSING")
    
    preprocessor = StrokeDataPreprocessor(config)
    
    # Load data
    df = preprocessor.load_data(data_path)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Encode categorical
    df = preprocessor.encode_categorical_features(df)
    
    # Train-test split & scaling
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\n✅ Preprocessing hoàn tất!")
    
    # ========================================
    # STEP 2: IMBALANCED DATA HANDLING
    # ========================================
    print_step(2, "IMBALANCED DATA HANDLING")
    
    handler = ImbalancedDataHandler(config)
    
    # Phân tích imbalance
    analysis = handler.analyze_imbalance(y_train)
    
    # Áp dụng các kỹ thuật resampling
    print("\n🔄 Áp dụng các kỹ thuật resampling...")
    
    X_smote, y_smote = handler.apply_smote(X_train, y_train)
    X_adasyn, y_adasyn = handler.apply_adasyn(X_train, y_train)
    X_ros, y_ros = handler.apply_random_oversampling(X_train, y_train)
    X_smote_tomek, y_smote_tomek = handler.apply_smote_tomek(X_train, y_train)
    
    # Visualizations
    handler.visualize_comparison(y_train)
    handler.print_summary()
    
    # Save resampled data (SMOTE)
    X_smote.to_csv('data/processed/resampled/X_train_smote.csv', index=False)
    y_smote.to_csv('data/processed/resampled/y_train_smote.csv', index=False)
    
    print("\n✅ Imbalanced handling hoàn tất!")
    
    # ========================================
    # STEP 3: MODEL TRAINING
    # ========================================
    print_step(3, "MODEL TRAINING")
    
    trainer = StrokeModelTrainer(config)
    
    print(f"\n⚙️ Hyperparameter tuning: {'ENABLED' if config['training']['hyperparameter_tuning'] else 'DISABLED'}")
    print("  (Set 'hyperparameter_tuning: true' trong config.yaml để enable)\n")
    
    # Train models
    if config['training']['models']['logistic_regression']:
        trainer.train_logistic_regression(X_smote, y_smote)
    
    if config['training']['models']['random_forest']:
        trainer.train_random_forest(X_smote, y_smote)
    
    if config['training']['models']['xgboost']:
        trainer.train_xgboost(X_smote, y_smote)
    
    if config['training']['models']['lightgbm']:
        trainer.train_lightgbm(X_smote, y_smote)
    
    # Print summary
    trainer.print_summary()
    
    # Save models
    trainer.save_models()
    
    print("\n✅ Training hoàn tất!")
    
    # ========================================
    # STEP 4: MODEL EVALUATION
    # ========================================
    print_step(4, "MODEL EVALUATION")
    
    evaluator = ModelEvaluator(config, models_dir='models')
    
    # Load models
    evaluator.load_models()
    
    # Evaluate each model
    for model_name, model in evaluator.models.items():
        evaluator.evaluate_single_model(model, X_test, y_test, model_name)
    
    # Generate visualizations
    if config['evaluation']['visualizations']:
        evaluator.plot_confusion_matrices(y_test)
        evaluator.plot_roc_curves(y_test)
        evaluator.plot_precision_recall_curves(y_test)
        evaluator.plot_feature_importance(X_test)
    
    # Classification reports
    evaluator.generate_classification_reports(y_test)
    
    # Comparison table
    df_comparison = evaluator.create_comparison_table()
    
    print("\n✅ Evaluation hoàn tất!")
    
    # ========================================
    # SUMMARY
    # ========================================
    end_time = time.time()
    total_time = end_time - start_time
    
    print_header("PIPELINE EXECUTION SUMMARY")
    
    print(f"\n⏱️ Thời gian thực thi: {total_time:.2f}s ({total_time/60:.2f} phút)")
    
    print(f"\n📊 Kết quả:")
    print(f"  - Dataset: {len(df)} bệnh nhân")
    print(f"  - Train/Test: {len(X_train)}/{len(X_test)}")
    print(f"  - Models trained: {len(evaluator.models)}")
    print(f"  - Imbalance ratio: 1:{analysis['imbalance_ratio']:.1f}")
    
    print(f"\n📁 Output files:")
    print(f"  - Processed data: data/processed/")
    print(f"  - Resampled data: data/processed/resampled/")
    print(f"  - Models: models/")
    print(f"  - Results: results/")
    
    print(f"\n🏆 Best Model:")
    best_model_name = df_comparison.loc[
        df_comparison['Recall'].astype(float).idxmax(), 'Model'
    ]
    best_recall = df_comparison.loc[
        df_comparison['Recall'].astype(float).idxmax(), 'Recall'
    ]
    print(f"  - {best_model_name} (Recall: {best_recall})")
    
    print("\n" + "="*80)
    print("✅ PIPELINE HOÀN TẤT THÀNH CÔNG!")
    print(f"  Kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    print("💡 Next steps:")
    print("  1. Xem kết quả trong thư mục 'results/'")
    print("  2. Đọc file 'results/model_comparison.csv'")
    print("  3. Sử dụng models trong 'models/' cho dự đoán\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline bị ngắt bởi người dùng!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)