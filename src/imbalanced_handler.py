"""
Module Xử lý Imbalanced Data - TRỌNG TÂM ĐỒ ÁN
So sánh 4 kỹ thuật: SMOTE, ADASYN, Random Oversampling, SMOTE+Tomek
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek
import os

class ImbalancedDataHandler:
    """Class xử lý imbalanced data với nhiều kỹ thuật"""
    
    def __init__(self, config):
        self.config = config
        self.random_state = config['data']['random_state']
        self.resampling_results = {}
    
    def analyze_imbalance(self, y):
        """Phân tích mức độ mất cân bằng"""
        print("\n" + "="*60)
        print("📊 PHÂN TÍCH MỨC ĐỘ MẤT CÂN BẰNG DỮ LIỆU")
        print("="*60)
        
        counter = Counter(y)
        majority_class = max(counter, key=counter.get)
        minority_class = min(counter, key=counter.get)
        
        total = len(y)
        majority_count = counter[majority_class]
        minority_count = counter[minority_class]
        imbalance_ratio = majority_count / minority_count
        
        print(f"\n📈 Thống kê:")
        print(f"  - Tổng số samples: {total}")
        print(f"  - Class {majority_class} (Majority): {majority_count} ({majority_count/total*100:.2f}%)")
        print(f"  - Class {minority_class} (Minority): {minority_count} ({minority_count/total*100:.2f}%)")
        print(f"\n⚠️ Mức độ mất cân bằng:")
        print(f"  - Tỷ lệ: 1:{imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            severity = "🔴 RẤT NGHIÊM TRỌNG"
        elif imbalance_ratio > 5:
            severity = "🟠 NGHIÊM TRỌNG"
        elif imbalance_ratio > 2:
            severity = "🟡 VỪA PHẢI"
        else:
            severity = "🟢 NHẸ"
        print(f"  - Mức độ: {severity}")
        
        return {
            'total': total,
            'majority_class': majority_class,
            'minority_class': minority_class,
            'majority_count': majority_count,
            'minority_count': minority_count,
            'imbalance_ratio': imbalance_ratio
        }
    
    def apply_smote(self, X, y):
        """Áp dụng SMOTE - Synthetic Minority Over-sampling"""
        print("\n" + "="*60)
        print("🔄 ÁP DỤNG SMOTE")
        print("="*60)
        
        smote = SMOTE(
            sampling_strategy=self.config['imbalanced']['sampling_strategy'],
            k_neighbors=self.config['imbalanced']['k_neighbors'],
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"\n📊 Kết quả:")
        print(f"  - Trước SMOTE: {len(X)} samples")
        print(f"  - Sau SMOTE: {len(X_resampled)} samples")
        print(f"  - Đã tạo thêm: {len(X_resampled) - len(X)} synthetic samples")
        
        counter = Counter(y_resampled)
        print(f"\n  Phân bố mới:")
        for class_label, count in counter.items():
            print(f"  - Class {class_label}: {count} ({count/len(y_resampled)*100:.2f}%)")
        
        self.resampling_results['SMOTE'] = {
            'X': X_resampled,
            'y': y_resampled,
            'distribution': dict(counter)
        }
        
        return X_resampled, y_resampled
    
    def apply_adasyn(self, X, y):
        """Áp dụng ADASYN - Adaptive Synthetic Sampling"""
        print("\n" + "="*60)
        print("🔄 ÁP DỤNG ADASYN")
        print("="*60)
        
        adasyn = ADASYN(
            sampling_strategy=self.config['imbalanced']['sampling_strategy'],
            n_neighbors=self.config['imbalanced']['k_neighbors'],
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        print(f"\n📊 Kết quả:")
        print(f"  - Trước ADASYN: {len(X)} samples")
        print(f"  - Sau ADASYN: {len(X_resampled)} samples")
        
        counter = Counter(y_resampled)
        self.resampling_results['ADASYN'] = {
            'X': X_resampled,
            'y': y_resampled,
            'distribution': dict(counter)
        }
        
        return X_resampled, y_resampled
    
    def apply_random_oversampling(self, X, y):
        """Áp dụng Random Oversampling"""
        print("\n" + "="*60)
        print("🔄 ÁP DỤNG RANDOM OVERSAMPLING")
        print("="*60)
        
        ros = RandomOverSampler(
            sampling_strategy=self.config['imbalanced']['sampling_strategy'],
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        counter = Counter(y_resampled)
        self.resampling_results['RandomOS'] = {
            'X': X_resampled,
            'y': y_resampled,
            'distribution': dict(counter)
        }
        
        return X_resampled, y_resampled
    
    def apply_smote_tomek(self, X, y):
        """Áp dụng SMOTE + Tomek Links"""
        print("\n" + "="*60)
        print("🔄 ÁP DỤNG SMOTE + TOMEK LINKS")
        print("="*60)
        
        smote_tomek = SMOTETomek(
            sampling_strategy=self.config['imbalanced']['sampling_strategy'],
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        counter = Counter(y_resampled)
        self.resampling_results['SMOTE+Tomek'] = {
            'X': X_resampled,
            'y': y_resampled,
            'distribution': dict(counter)
        }
        
        return X_resampled, y_resampled
    
    def visualize_comparison(self, original_y, figsize=(15, 8)):
        """Visualize so sánh các kỹ thuật"""
        print("\n" + "="*60)
        print("📊 TẠO BIỂU ĐỒ SO SÁNH")
        print("="*60)
        
        n_methods = len(self.resampling_results) + 1
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Plot original
        counter_original = Counter(original_y)
        axes[0].bar(counter_original.keys(), counter_original.values(),
                   color=['#3498db', '#e74c3c'])
        axes[0].set_title('Original Data\n(Imbalanced)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, (k, v) in enumerate(counter_original.items()):
            axes[0].text(k, v + max(counter_original.values())*0.02,
                        f'{v}\n({v/sum(counter_original.values())*100:.1f}%)',
                        ha='center', fontsize=10)
        
        # Plot resampled
        for idx, (method_name, result) in enumerate(self.resampling_results.items(), 1):
            counter = result['distribution']
            axes[idx].bar(counter.keys(), counter.values(),
                         color=['#2ecc71', '#f39c12'])
            axes[idx].set_title(f'{method_name}\n(Balanced)', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Class')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(axis='y', alpha=0.3)
            
            for i, (k, v) in enumerate(counter.items()):
                axes[idx].text(k, v + max(counter.values())*0.02,
                              f'{v}\n({v/sum(counter.values())*100:.1f}%)',
                              ha='center', fontsize=10)
        
        # Ẩn subplot thừa
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/resampling_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu biểu đồ tại: results/resampling_comparison.png")
        plt.show()
    
    def print_summary(self):
        """In tổng kết các kỹ thuật"""
        print("\n" + "="*60)
        print("📝 TỔNG KẾT CÁC KỸ THUẬT RESAMPLING")
        print("="*60)
        
        summary_data = []
        for method, result in self.resampling_results.items():
            dist = result['distribution']
            summary_data.append({
                'Kỹ thuật': method,
                'Tổng samples': sum(dist.values()),
                'Class 0': dist.get(0, 0),
                'Class 1': dist.get(1, 0),
                'Tỷ lệ': f"{dist.get(0, 0)/dist.get(1, 1):.2f}:1"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(f"\n{df_summary.to_string(index=False)}")
        
        print("\n💡 KHUYẾN NGHỊ:")
        print("  - SMOTE: Tốt cho hầu hết trường hợp")
        print("  - ADASYN: Tốt khi có outliers")
        print("  - Random OS: Đơn giản nhưng dễ overfit")
        print("  - SMOTE+Tomek: Cân bằng và làm sạch")

def main():
    """Demo pipeline"""
    import yaml
    
    print("\n" + "⚖️ "*20)
    print("IMBALANCED DATA HANDLING MODULE")
    print("⚖️ "*20)
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load dữ liệu đã preprocessing
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    # Khởi tạo handler
    handler = ImbalancedDataHandler(config)
    
    # Phân tích mức độ imbalance
    analysis = handler.analyze_imbalance(y_train)
    
    # Áp dụng các kỹ thuật
    print("\n" + "🔄 "*20)
    print("BẮT ĐẦU ÁP DỤNG CÁC KỸ THUẬT RESAMPLING")
    print("🔄 "*20)
    
    X_smote, y_smote = handler.apply_smote(X_train, y_train)
    X_adasyn, y_adasyn = handler.apply_adasyn(X_train, y_train)
    X_ros, y_ros = handler.apply_random_oversampling(X_train, y_train)
    X_smote_tomek, y_smote_tomek = handler.apply_smote_tomek(X_train, y_train)
    
    # Visualize
    os.makedirs('results', exist_ok=True)
    handler.visualize_comparison(y_train)
    
    # Print summary
    handler.print_summary()
    
    # Lưu dữ liệu SMOTE (phổ biến nhất)
    os.makedirs('data/processed/resampled', exist_ok=True)
    pd.DataFrame(X_smote).to_csv('data/processed/resampled/X_train_smote.csv', index=False)
    pd.DataFrame(y_smote).to_csv('data/processed/resampled/y_train_smote.csv', index=False)
    
    print("\n💾 Đã lưu dữ liệu SMOTE:")
    print("  - data/processed/resampled/X_train_smote.csv")
    print("  - data/processed/resampled/y_train_smote.csv")
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH XỬ LÝ IMBALANCED DATA!")
    print("="*60)

if __name__ == "__main__":
    main()