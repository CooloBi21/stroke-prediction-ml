"""
Module Evaluation Models - Stroke Prediction
Confusion Matrix, ROC Curve, Precision-Recall Curve, Feature Importance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import os
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelEvaluator:
    """Class đánh giá hiệu suất models"""
    
    def __init__(self, config, models_dir='models'):
        self.config = config
        self.models_dir = models_dir
        self.models = {}
        self.evaluation_results = {}
    
    def load_models(self):
        """Load tất cả models đã train"""
        print("\n" + "="*60)
        print("📥 ĐANG LOAD MODELS...")
        print("="*60)
        
        model_files = [f for f in os.listdir(self.models_dir)
                      if f.endswith('.pkl') and f != 'cv_results.pkl' 
                      and f != 'preprocessor.pkl']
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            filepath = os.path.join(self.models_dir, model_file)
            self.models[model_name] = joblib.load(filepath)
            print(f"✓ Đã load {model_name}")
        
        print(f"\n✓ Tổng cộng: {len(self.models)} models")
    
    def evaluate_single_model(self, model, X_test, y_test, model_name):
        """Đánh giá một model trên test set"""
        print(f"\n{'='*60}")
        print(f"🔍 ĐÁNH GIÁ {model_name.upper()}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Tính các metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        
        # Print metrics
        print(f"\n📊 Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} ← Tỷ lệ dự đoán đúng trong các ca ĐOÁN là đột quỵ")
        print(f"  Recall: {metrics['recall']:.4f} ← Tỷ lệ phát hiện được ca đột quỵ THỰC SỰ (quan trọng nhất!)")
        print(f"  F1-Score: {metrics['f1_score']:.4f} ← Cân bằng Precision và Recall")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Phân tích y tế
        print(f"\n🏥 Phân tích y tế:")
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"  True Positives (TP): {tp} - Phát hiện ĐÚNG ca đột quỵ")
        print(f"  False Negatives (FN): {fn} - BỎ SÓT ca đột quỵ (nguy hiểm!)")
        print(f"  False Positives (FP): {fp} - Cảnh báo SAI (dương tính giả)")
        print(f"  True Negatives (TN): {tn} - Phát hiện đúng ca KHÔNG đột quỵ")
        
        if fn > 0:
            print(f"\n  ⚠️ CẢNH BÁO: Có {fn} ca đột quỵ BỊ BỎ SÓT!")
            print(f"    → Trong y tế, False Negative rất nguy hiểm!")
        
        # Lưu kết quả
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def plot_confusion_matrices(self, y_test, figsize=(16, 4)):
        """Vẽ confusion matrices cho tất cả models"""
        print(f"\n{'='*60}")
        print("📊 TẠO CONFUSION MATRICES")
        print(f"{'='*60}")
        
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'],
                ax=axes[idx],
                cbar=False
            )
            
            axes[idx].set_title(f'{model_name}\nConfusion Matrix',
                              fontweight='bold', fontsize=12)
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=10)
            
            # Thêm annotations cho y tế
            tn, fp, fn, tp = cm.ravel()
            axes[idx].text(0.5, -0.2,
                         f'FN={fn} (Bỏ sót)',
                         transform=axes[idx].transAxes,
                         ha='center', fontsize=9, color='red')
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: results/confusion_matrices.png")
        plt.show()
    
    def plot_roc_curves(self, y_test, figsize=(10, 8)):
        """Vẽ ROC curves cho tất cả models"""
        print(f"\n{'='*60}")
        print("📈 TẠO ROC CURVES")
        print(f"{'='*60}")
        
        plt.figure(figsize=figsize)
        
        for model_name, results in self.evaluation_results.items():
            y_pred_proba = results['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Đường baseline
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: results/roc_curves.png")
        plt.show()
    
    def plot_precision_recall_curves(self, y_test, figsize=(10, 8)):
        """Vẽ Precision-Recall curves (quan trọng cho imbalanced data)"""
        print(f"\n{'='*60}")
        print("📈 TẠO PRECISION-RECALL CURVES")
        print(f"{'='*60}")
        
        plt.figure(figsize=figsize)
        
        for model_name, results in self.evaluation_results.items():
            y_pred_proba = results['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{model_name} (AUC = {pr_auc:.4f})')
        
        # Baseline
        baseline = (y_test == 1).sum() / len(y_test)
        plt.plot([0, 1], [baseline, baseline], 'k--', linewidth=1,
                label=f'Baseline ({baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison\n(Quan trọng cho Imbalanced Data)',
                 fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: results/precision_recall_curves.png")
        plt.show()
    
    def plot_feature_importance(self, X_test, top_n=15, figsize=(12, 8)):
        """Vẽ feature importance cho tree-based models"""
        print(f"\n{'='*60}")
        print("📊 PHÂN TÍCH FEATURE IMPORTANCE")
        print(f"{'='*60}")
        
        tree_based_models = ['RandomForest', 'XGBoost', 'LightGBM']
        available_models = [m for m in tree_based_models if m in self.models]
        
        if not available_models:
            print("⚠️ Không có tree-based model nào!")
            return
        
        n_models = len(available_models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(available_models):
            model = self.models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                continue
            
            # Tạo DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': X_test.columns,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            axes[idx].barh(range(len(feature_importance_df)),
                         feature_importance_df['importance'],
                         color='steelblue')
            axes[idx].set_yticks(range(len(feature_importance_df)))
            axes[idx].set_yticklabels(feature_importance_df['feature'])
            axes[idx].set_xlabel('Importance', fontsize=10)
            axes[idx].set_title(f'{model_name}\nTop {top_n} Features',
                              fontweight='bold', fontsize=12)
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
            
            # In ra console
            print(f"\n🔝 Top 5 features của {model_name}:")
            for i, row in feature_importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Đã lưu: results/feature_importance.png")
        plt.show()
    
    def generate_classification_reports(self, y_test):
        """Tạo classification reports chi tiết"""
        print(f"\n{'='*60}")
        print("📋 CLASSIFICATION REPORTS")
        print(f"{'='*60}")
        
        for model_name, results in self.evaluation_results.items():
            y_pred = results['y_pred']
            
            print(f"\n{'─'*60}")
            print(f"{model_name}")
            print(f"{'─'*60}")
            print(classification_report(y_test, y_pred,
                                       target_names=['No Stroke', 'Stroke'],
                                       digits=4))
    
    def create_comparison_table(self):
        """Tạo bảng so sánh các models"""
        print(f"\n{'='*60}")
        print("📊 BẢNG SO SÁNH MODELS")
        print(f"{'='*60}")
        
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(f"\n{df_comparison.to_string(index=False)}")
        
        # Lưu ra CSV
        df_comparison.to_csv('results/model_comparison.csv', index=False)
        print("\n✓ Đã lưu: results/model_comparison.csv")
        
        # Tìm best model
        best_recall_idx = df_comparison['Recall'].astype(float).idxmax()
        
        print(f"\n🏆 BEST MODEL (theo Recall - quan trọng nhất):")
        print(f"  {df_comparison.loc[best_recall_idx, 'Model']}")
        print(f"  Recall: {df_comparison.loc[best_recall_idx, 'Recall']}")
        
        return df_comparison

def main():
    """Demo evaluation pipeline"""
    import yaml
    
    print("\n" + "🎯 "*20)
    print("MODEL EVALUATION MODULE")
    print("🎯 "*20)
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Tạo thư mục results
    os.makedirs('results', exist_ok=True)
    
    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"\n📊 Test data shape: {X_test.shape}")
    print(f"📊 Class distribution: {np.bincount(y_test)}")
    
    # Khởi tạo evaluator
    evaluator = ModelEvaluator(config, models_dir='models')
    
    # Load models
    evaluator.load_models()
    
    # Evaluate từng model
    print("\n" + "🔍 "*20)
    print("BẮT ĐẦU EVALUATION")
    print("🔍 "*20)
    
    for model_name, model in evaluator.models.items():
        evaluator.evaluate_single_model(model, X_test, y_test, model_name)
    
    # Visualizations
    if config['evaluation']['visualizations']:
        print("\n" + "📊 "*20)
        print("TẠO VISUALIZATIONS")
        print("📊 "*20)
        
        evaluator.plot_confusion_matrices(y_test)
        evaluator.plot_roc_curves(y_test)
        evaluator.plot_precision_recall_curves(y_test)
        evaluator.plot_feature_importance(X_test)
    
    # Reports
    evaluator.generate_classification_reports(y_test)
    df_comparison = evaluator.create_comparison_table()
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH EVALUATION!")
    print("="*60)
    print("\n📁 Kết quả đã lưu trong thư mục 'results/'")

if __name__ == "__main__":
    main()