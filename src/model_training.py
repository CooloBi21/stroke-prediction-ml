"""
Module Training Models - Stroke Prediction
Train 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM
với Cross-Validation và Hyperparameter Tuning
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import time

warnings.filterwarnings('ignore')

class StrokeModelTrainer:
    """Class training nhiều models với tối ưu cho imbalanced data"""
    
    def __init__(self, config):
        self.config = config
        self.random_state = config['data']['random_state']
        self.cv_folds = config['training']['cv_folds']
        self.models = {}
        self.cv_results = {}
        self.best_models = {}
        
        # Stratified K-Fold cho imbalanced data
        self.skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
    
    def train_logistic_regression(self, X_train, y_train, tune=False):
        """Training Logistic Regression"""
        print("\n" + "="*60)
        print("🔵 TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        if tune and self.config['training']['hyperparameter_tuning']:
            print("\n⚙️ Đang thực hiện Hyperparameter Tuning...")
            
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
            
            scorer = make_scorer(f1_score)
            lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            grid_search = GridSearchCV(
                lr, param_grid,
                cv=self.skf,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            best_model = grid_search.best_estimator_
            print(f"\n✓ Best parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"  - {param}: {value}")
            print(f"\n✓ Best F1-Score (CV): {grid_search.best_score_:.4f}")
            print(f"✓ Training time: {training_time:.2f}s")
        else:
            print("\n⚙️ Training với default parameters...")
            best_model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            start_time = time.time()
            best_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"✓ Training time: {training_time:.2f}s")
        
        # Cross-validation
        self._evaluate_cv(best_model, X_train, y_train, "Logistic Regression")
        
        self.models['LogisticRegression'] = best_model
        self.best_models['LogisticRegression'] = {
            'model': best_model,
            'training_time': training_time
        }
        
        return best_model
    
    def train_random_forest(self, X_train, y_train, tune=False):
        """Training Random Forest"""
        print("\n" + "="*60)
        print("🌲 TRAINING RANDOM FOREST")
        print("="*60)
        
        if tune and self.config['training']['hyperparameter_tuning']:
            print("\n⚙️ Đang thực hiện Hyperparameter Tuning...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            scorer = make_scorer(f1_score)
            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            
            grid_search = GridSearchCV(
                rf, param_grid,
                cv=self.skf,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            best_model = grid_search.best_estimator_
            print(f"\n✓ Best parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"  - {param}: {value}")
            print(f"\n✓ Best F1-Score (CV): {grid_search.best_score_:.4f}")
            print(f"✓ Training time: {training_time:.2f}s")
        else:
            print("\n⚙️ Training với default parameters...")
            best_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            start_time = time.time()
            best_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"✓ Training time: {training_time:.2f}s")
        
        self._evaluate_cv(best_model, X_train, y_train, "Random Forest")
        
        self.models['RandomForest'] = best_model
        self.best_models['RandomForest'] = {
            'model': best_model,
            'training_time': training_time
        }
        
        return best_model
    
    def train_xgboost(self, X_train, y_train, tune=False):
        """Training XGBoost"""
        print("\n" + "="*60)
        print("🚀 TRAINING XGBOOST")
        print("="*60)
        
        # Tính scale_pos_weight cho imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"\n📊 Scale pos weight: {scale_pos_weight:.2f}")
        
        if tune and self.config['training']['hyperparameter_tuning']:
            print("\n⚙️ Đang thực hiện Hyperparameter Tuning...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'scale_pos_weight': [scale_pos_weight]
            }
            
            scorer = make_scorer(f1_score)
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            grid_search = GridSearchCV(
                xgb_model, param_grid,
                cv=self.skf,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            best_model = grid_search.best_estimator_
            print(f"\n✓ Best parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"  - {param}: {value}")
            print(f"\n✓ Best F1-Score (CV): {grid_search.best_score_:.4f}")
            print(f"✓ Training time: {training_time:.2f}s")
        else:
            print("\n⚙️ Training với default parameters...")
            best_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            start_time = time.time()
            best_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"✓ Training time: {training_time:.2f}s")
        
        self._evaluate_cv(best_model, X_train, y_train, "XGBoost")
        
        self.models['XGBoost'] = best_model
        self.best_models['XGBoost'] = {
            'model': best_model,
            'training_time': training_time
        }
        
        return best_model
    
    def train_lightgbm(self, X_train, y_train, tune=False):
        """Training LightGBM"""
        print("\n" + "="*60)
        print("⚡ TRAINING LIGHTGBM")
        print("="*60)
        
        if tune and self.config['training']['hyperparameter_tuning']:
            print("\n⚙️ Đang thực hiện Hyperparameter Tuning...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50]
            }
            
            scorer = make_scorer(f1_score)
            lgbm_model = lgb.LGBMClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
            
            grid_search = GridSearchCV(
                lgbm_model, param_grid,
                cv=self.skf,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            best_model = grid_search.best_estimator_
            print(f"\n✓ Best parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"  - {param}: {value}")
            print(f"\n✓ Best F1-Score (CV): {grid_search.best_score_:.4f}")
            print(f"✓ Training time: {training_time:.2f}s")
        else:
            print("\n⚙️ Training với default parameters...")
            best_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
            start_time = time.time()
            best_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"✓ Training time: {training_time:.2f}s")
        
        self._evaluate_cv(best_model, X_train, y_train, "LightGBM")
        
        self.models['LightGBM'] = best_model
        self.best_models['LightGBM'] = {
            'model': best_model,
            'training_time': training_time
        }
        
        return best_model
    
    def _evaluate_cv(self, model, X, y, model_name):
        """Đánh giá model bằng cross-validation"""
        print(f"\n🔍 Cross-Validation ({self.cv_folds}-fold):")
        
        # Metrics quan trọng cho imbalanced data
        metrics = {
            'Recall': make_scorer(recall_score),
            'F1-Score': make_scorer(f1_score),
            'Precision': make_scorer(precision_score)
        }
        
        cv_scores = {}
        for metric_name, scorer in metrics.items():
            scores = cross_val_score(
                model, X, y,
                cv=self.skf,
                scoring=scorer,
                n_jobs=-1
            )
            cv_scores[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"  {metric_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        self.cv_results[model_name] = cv_scores
    
    def save_models(self, output_dir='models'):
        """Lưu tất cả models"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("💾 ĐANG LƯU MODELS...")
        print("="*60)
        
        for model_name, model_info in self.best_models.items():
            filepath = os.path.join(output_dir, f'{model_name}.pkl')
            joblib.dump(model_info['model'], filepath)
            print(f"✓ Đã lưu {model_name} tại: {filepath}")
        
        # Lưu CV results
        cv_filepath = os.path.join(output_dir, 'cv_results.pkl')
        joblib.dump(self.cv_results, cv_filepath)
        print(f"✓ Đã lưu CV results tại: {cv_filepath}")
    
    def print_summary(self):
        """In tổng kết training results"""
        print("\n" + "="*60)
        print("📊 TỔNG KẾT TRAINING RESULTS")
        print("="*60)
        
        summary_data = []
        for model_name, cv_scores in self.cv_results.items():
            summary_data.append({
                'Model': model_name,
                'Recall': f"{cv_scores['Recall']['mean']:.4f} ± {cv_scores['Recall']['std']:.4f}",
                'F1-Score': f"{cv_scores['F1-Score']['mean']:.4f} ± {cv_scores['F1-Score']['std']:.4f}",
                'Precision': f"{cv_scores['Precision']['mean']:.4f} ± {cv_scores['Precision']['std']:.4f}",
                'Training Time': f"{self.best_models[model_name]['training_time']:.2f}s"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(f"\n{df_summary.to_string(index=False)}")
        
        # Tìm best model theo Recall (quan trọng nhất cho y tế)
        best_model_name = max(
            self.cv_results.keys(),
            key=lambda x: self.cv_results[x]['Recall']['mean']
        )
        
        print(f"\n🏆 BEST MODEL (theo Recall): {best_model_name}")
        print(f"  Recall: {self.cv_results[best_model_name]['Recall']['mean']:.4f}")

def main():
    """Demo training pipeline"""
    import yaml
    
    print("\n" + "🤖 "*20)
    print("MODEL TRAINING MODULE")
    print("🤖 "*20)
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load dữ liệu đã resampled (SMOTE)
    X_train = pd.read_csv('data/processed/resampled/X_train_smote.csv')
    y_train = pd.read_csv('data/processed/resampled/y_train_smote.csv').values.ravel()
    
    print(f"\n📊 Training data shape: {X_train.shape}")
    print(f"📊 Class distribution: {np.bincount(y_train)}")
    
    # Khởi tạo trainer
    trainer = StrokeModelTrainer(config)
    
    # Training các models
    print("\n" + "🚀 "*20)
    print("BẮT ĐẦU TRAINING MODELS")
    print("🚀 "*20)
    
    if config['training']['models']['logistic_regression']:
        trainer.train_logistic_regression(X_train, y_train)
    
    if config['training']['models']['random_forest']:
        trainer.train_random_forest(X_train, y_train)
    
    if config['training']['models']['xgboost']:
        trainer.train_xgboost(X_train, y_train)
    
    if config['training']['models']['lightgbm']:
        trainer.train_lightgbm(X_train, y_train)
    
    # Print summary
    trainer.print_summary()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH TRAINING!")
    print("="*60)

if __name__ == "__main__":
    main()