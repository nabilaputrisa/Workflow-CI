import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import pickle
import yaml
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed untuk reproducibility
np.random.seed(42)

class CoffeeMLflowModel:
    def __init__(self, experiment_name="Coffee_Classification_Experiment"):
        print("="*70)
        print("☕ COFFEE CLASSIFICATION WITH MLFLOW")
        print("="*70)
        
        # Setup tracking URI yang lebih reliable
        tracking_path = "file:./mlruns"
        mlflow.set_tracking_uri(tracking_path)
        
        # Pastikan folder mlruns ada
        os.makedirs("./mlruns", exist_ok=True)
        
        # Nonaktifkan autolog untuk kontrol manual
        mlflow.sklearn.autolog(disable=True)
        
        # Set experiment
        experiment = mlflow.set_experiment(experiment_name)
        
        print(f"📊 Experiment: {experiment_name}")
        print(f"📁 Tracking URI: {tracking_path}")
        print(f"📂 MLRuns directory: {os.path.abspath('./mlruns')}")
        print("="*70)
    
    def load_preprocessed_data(self, data_path="kopi_preprocessed.csv"):
        """
        Memuat data kopi yang sudah dipreprocessing
        """
        try:
            df = pd.read_csv(data_path)
            print(f"✅ Data loaded successfully. Shape: {df.shape}")
            print(f"📊 Columns: {list(df.columns)}")
            print(f"\n🔍 First 3 rows:")
            print(df.head(3))
            
            # Cek missing values
            missing = df.isnull().sum().sum()
            if missing > 0:
                print(f"⚠️  Found {missing} missing values")
            else:
                print("✅ No missing values")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ File {data_path} not found!")
            print("📝 Creating sample data for demonstration...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Membuat data sample jika file tidak ditemukan"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'drink': [f'Drink_{i}' for i in range(n_samples)],
            'Volume (ml)': np.random.uniform(100, 500, n_samples),
            'Calories': np.random.uniform(50, 300, n_samples),
            'Caffeine (mg)': np.random.uniform(50, 200, n_samples),
            'type_Energy Drinks': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'type_Energy Shots': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
            'type_Soft Drinks': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
            'type_Tea': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'type_Water': np.random.choice([True, False], n_samples, p=[0.1, 0.9])
        }
        
        df = pd.DataFrame(data)
        print("✅ Created sample data with shape:", df.shape)
        return df
    
    def prepare_features_target(self, df):
        """
        Menyiapkan fitur dan target
        """
        print("\n" + "="*60)
        print("PREPARING FEATURES AND TARGET")
        print("="*60)
        
        # Features - ambil kolom numerik
        numeric_cols = ['Volume (ml)', 'Calories', 'Caffeine (mg)']
        
        # Cek kolom yang tersedia
        available_cols = [col for col in numeric_cols if col in df.columns]
        if not available_cols:
            # Gunakan semua kolom numerik selain yang bukan fitur
            available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[available_cols].copy()
        print(f"✅ Features: {list(X.columns)}")
        print(f"✅ X shape: {X.shape}")
        
        # Membuat target dari kolom type
        type_cols = [col for col in df.columns if col.startswith('type_')]
        
        if len(type_cols) > 0:
            print(f"✅ Found type columns: {type_cols}")
            
            # Buat target berdasarkan type yang True
            y_list = []
            for idx, row in df.iterrows():
                true_types = []
                for col in type_cols:
                    if row[col] == True or row[col] == 1:
                        true_types.append(col.replace('type_', ''))
                
                if true_types:
                    y_list.append(true_types[0])
                else:
                    y_list.append('Coffee')
            
            le = LabelEncoder()
            y = le.fit_transform(y_list)
            
            self.label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
            self.label_encoder = le
            
            print(f"✅ Target classes: {list(le.classes_)}")
            
            # Distribusi kelas
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n📊 Class distribution:")
            for cls, count in zip(unique, counts):
                class_name = le.inverse_transform([cls])[0]
                percentage = (count / len(y)) * 100
                print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        else:
            print("⚠️  No type columns found. Creating synthetic target...")
            le = LabelEncoder()
            y = le.fit_transform(np.random.choice(['Class_A', 'Class_B', 'Class_C'], len(X)))
            self.label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
            self.label_encoder = le
        
        return X, y, le
    
    def _create_conda_yaml(self, artifact_dir):
        """Membuat file conda.yaml untuk environment"""
        conda_env = {
            'name': 'coffee-classification-env',
            'channels': ['conda-forge', 'defaults'],
            'dependencies': [
                'python=3.9',
                'pip',
                {
                    'pip': [
                        'mlflow==2.11.3',
                        'scikit-learn==1.3.2',
                        'pandas==2.1.3',
                        'numpy==1.24.4',
                        'matplotlib==3.8.2',
                        'seaborn==0.13.0',
                        'pyyaml==6.0.1'
                    ]
                }
            ]
        }
        
        conda_file = os.path.join(artifact_dir, "conda.yaml")
        with open(conda_file, 'w') as f:
            yaml.dump(conda_env, f, default_flow_style=False)
        
        print(f"  ✓ conda.yaml created")
        return conda_file
    
    def _create_mlmodel_file(self, artifact_dir, model_signature=None):
        """Membuat file MLmodel untuk metadata model"""
        mlmodel_content = """\
artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: null
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.9.0
  sklearn:
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.3.2
run_id: placeholder_run_id
utc_time_created: '2025-12-23 21:16:32.123456'
"""

        # Ganti placeholder dengan timestamp aktual
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        mlmodel_content = mlmodel_content.replace(
            "'2025-12-23 21:16:32.123456'", 
            f"'{timestamp}'"
        )
        
        mlmodel_file = os.path.join(artifact_dir, "MLmodel")
        with open(mlmodel_file, 'w') as f:
            f.write(mlmodel_content)
        
        print(f"  ✓ MLmodel file created")
        return mlmodel_file
    
    def _create_model_artifact_folder(self, artifact_dir, model, X_sample):
        """Membuat folder model dengan semua file yang diperlukan"""
        model_artifact_dir = os.path.join(artifact_dir, "model")
        os.makedirs(model_artifact_dir, exist_ok=True)
        
        # 1. Simpan model sebagai pickle
        model_file = os.path.join(model_artifact_dir, "model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # 2. Buat conda.yaml
        self._create_conda_yaml(model_artifact_dir)
        
        # 3. Buat MLmodel file
        self._create_mlmodel_file(model_artifact_dir)
        
        # 4. Buat requirements.txt
        requirements_content = """mlflow==2.11.3
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.4
matplotlib==3.8.2
seaborn==0.13.0
pyyaml==6.0.1
"""
        requirements_file = os.path.join(model_artifact_dir, "requirements.txt")
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        # 5. Simpan input example
        input_example = {
            "input_data": X_sample.iloc[0:1].to_dict('records')[0]
        }
        input_example_file = os.path.join(model_artifact_dir, "input_example.json")
        with open(input_example_file, 'w') as f:
            json.dump(input_example, f, indent=2)
        
        # 6. Buat metadata tambahan
        metadata = {
            "model_type": "RandomForestClassifier",
            "training_date": datetime.now().isoformat(),
            "features": list(X_sample.columns.tolist()),
            "n_classes": len(self.label_encoder.classes_) if hasattr(self, 'label_encoder') else "unknown"
        }
        metadata_file = os.path.join(model_artifact_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Model artifact folder created with all files")
        return model_artifact_dir
    
    def train_model(self, X, y, label_encoder, model_name="Coffee_RandomForest"):
        """
        Melatih model dan menyimpan semua artifact
        """
        print("\n" + "="*60)
        print("TRAINING MODEL WITH MLFLOW")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Train set: {X_train.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        # Buat nama run yang unik
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"
        
        print(f"\n🚀 Starting MLflow run: {run_name}")
        
        # MULAI RUN MLFLOW
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"📝 Run ID: {run_id}")
            
            # 1. LOG PARAMETERS
            print("\n📝 Logging parameters...")
            params = {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "test_size": 0.2,
                "features": ", ".join(X.columns.tolist()),
                "n_classes": len(np.unique(y))
            }
            
            for key, value in params.items():
                mlflow.log_param(key, value)
                print(f"  {key}: {value}")
            
            # 2. TRAIN MODEL
            print("\n🔧 Training RandomForest model...")
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=params["random_state"],
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # 3. EVALUATE MODEL
            print("\n📊 Evaluating model...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # LOG METRICS
            metrics = {
                "accuracy": accuracy,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_score_weighted": f1,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
            
            # 4. BUAT FOLDER ARTIFACT LOKAL
            artifact_dir = f"./artifacts_{run_id}"
            os.makedirs(artifact_dir, exist_ok=True)
            
            print(f"\n🎨 Creating artifacts in: {artifact_dir}")
            
            # 5. BUAT FOLDER MODEL DENGAN SEMUA FILE
            model_artifact_dir = self._create_model_artifact_folder(artifact_dir, model, X)
            
            # 6. BUAT ARTIFACT LAINNYA
            # a. Feature importance
            self._save_feature_importance(model, X.columns, artifact_dir)
            
            # b. Label mapping
            self._save_label_mapping(artifact_dir)
            
            # c. Evaluation plots
            self._save_evaluation_plots(model, X_test, y_test, y_pred, label_encoder, artifact_dir)
            
            # d. Sample predictions
            self._save_sample_predictions(model, X_test, y_test, label_encoder, artifact_dir)
            
            # e. Training summary
            self._save_training_summary(params, metrics, artifact_dir)
            
            # f. Log model ke MLflow dengan signature
            print("\n💾 Logging model to MLflow...")
            # Pertama, log model menggunakan MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="mlflow_model",
                registered_model_name=model_name,
                conda_env=self._create_conda_yaml(artifact_dir),
                signature=None  # Optional: bisa tambahkan signature
            )
            
            # 7. LOG SEMUA ARTIFACT KE MLFLOW
            print("\n📦 Logging all artifacts to MLflow...")
            for root, dirs, files in os.walk(artifact_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    # Hitung path relatif untuk artifact
                    rel_path = os.path.relpath(filepath, artifact_dir)
                    mlflow.log_artifact(filepath, artifact_path=os.path.dirname(rel_path) if os.path.dirname(rel_path) else None)
                    print(f"  ✓ {rel_path}")
            
            # 8. LOG DATA SAMPLE
            X_sample = X.head().to_dict()
            sample_file = os.path.join(artifact_dir, "data_sample.json")
            with open(sample_file, 'w') as f:
                json.dump(X_sample, f, indent=2)
            mlflow.log_artifact(sample_file)
            
            # 9. SAVE MODEL SEPARATELY (backup)
            model_file = os.path.join(artifact_dir, "model_backup.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_file)
            
            # 10. TAMPILKAN INFORMASI FINAL
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Tampilkan informasi run
            print(f"\n📊 Model Performance:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            print(f"\n📁 Artifact Locations:")
            print(f"  Local: {artifact_dir}")
            print(f"  MLflow: {mlflow.get_artifact_uri()}")
            
            print(f"\n📦 Files in model artifact folder:")
            model_files = os.listdir(model_artifact_dir)
            for file in sorted(model_files):
                print(f"  • {file}")
            
            print(f"\n🔗 Run Information:")
            print(f"  Run ID: {run_id}")
            print(f"  Run Name: {run_name}")
            print(f"  Experiment: {mlflow.get_experiment(run.info.experiment_id).name}")
            
            # Classification report
            from sklearn.metrics import classification_report
            print(f"\n📋 Classification Report:")
            report = classification_report(y_test, y_pred, 
                                         target_names=label_encoder.classes_,
                                         zero_division=0)
            print(report)
            
            # Simpan classification report
            report_file = os.path.join(artifact_dir, "classification_report.txt")
            with open(report_file, 'w') as f:
                f.write(report)
            mlflow.log_artifact(report_file)
            
            # 11. VERIFIKASI FILE YANG DIBUAT
            print(f"\n🔍 VERIFICATION - Essential files created:")
            essential_files = [
                "model/MLmodel",
                "model/conda.yaml", 
                "model/model.pkl",
                "model/requirements.txt",
                "confusion_matrix.png",
                "feature_importance.png",
                "accuracy_per_class.png",
                "feature_importance.json",
                "label_mapping.json"
            ]
            
            for file in essential_files:
                file_path = os.path.join(artifact_dir, file)
                if os.path.exists(file_path):
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} - MISSING!")
            
            return model, run_id, artifact_dir
    
    def _save_feature_importance(self, model, feature_names, artifact_dir):
        """Simpan feature importance"""
        importance = model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))
        
        # Simpan sebagai JSON
        json_file = os.path.join(artifact_dir, "feature_importance.json")
        with open(json_file, 'w') as f:
            json.dump(importance_dict, f, indent=2)
        
        # Buat plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        
        plt.bar(range(len(feature_names)), importance[indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance - Coffee Classification')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
        
        plot_file = os.path.join(artifact_dir, "feature_importance.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
        print(f"  ✓ Feature importance saved")
    
    def _save_label_mapping(self, artifact_dir):
        """Simpan label mapping"""
        if hasattr(self, 'label_mapping'):
            json_file = os.path.join(artifact_dir, "label_mapping.json")
            with open(json_file, 'w') as f:
                json.dump(self.label_mapping, f, indent=2)
            print(f"  ✓ Label mapping saved")
    
    def _save_evaluation_plots(self, model, X_test, y_test, y_pred, label_encoder, artifact_dir):
        """Buat dan simpan plot evaluasi"""
        from sklearn.metrics import confusion_matrix
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix - Coffee Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_file = os.path.join(artifact_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_file, dpi=300)
        plt.close()
        
        # Accuracy per Class
        classes = label_encoder.classes_
        accuracies = []
        
        for i in range(len(classes)):
            mask = y_test == i
            if mask.any():
                class_acc = accuracy_score(y_test[mask], y_pred[mask])
                accuracies.append(class_acc)
            else:
                accuracies.append(0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(classes)), accuracies, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Class - Coffee Classification')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.2f}', ha='center', va='bottom')
        
        acc_file = os.path.join(artifact_dir, "accuracy_per_class.png")
        plt.tight_layout()
        plt.savefig(acc_file, dpi=300)
        plt.close()
        
        print(f"  ✓ Evaluation plots saved")
    
    def _save_sample_predictions(self, model, X_test, y_test, label_encoder, artifact_dir):
        """Simpan sample predictions"""
        # Ambil 5 sample pertama
        X_sample = X_test.iloc[:5]
        y_true_sample = y_test[:5]
        y_pred_sample = model.predict(X_sample)
        y_proba_sample = model.predict_proba(X_sample)
        
        predictions = []
        for i in range(len(X_sample)):
            pred_dict = {
                "features": X_sample.iloc[i].to_dict(),
                "true_class": label_encoder.inverse_transform([y_true_sample[i]])[0],
                "predicted_class": label_encoder.inverse_transform([y_pred_sample[i]])[0],
                "probabilities": dict(zip(label_encoder.classes_, y_proba_sample[i].tolist()))
            }
            predictions.append(pred_dict)
        
        json_file = os.path.join(artifact_dir, "sample_predictions.json")
        with open(json_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"  ✓ Sample predictions saved")
    
    def _save_training_summary(self, params, metrics, artifact_dir):
        """Simpan summary training"""
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params,
            "metrics": metrics,
            "model_info": "RandomForest Classifier for Coffee Classification",
            "essential_files": [
                "model/MLmodel",
                "model/conda.yaml",
                "model/model.pkl",
                "model/requirements.txt",
                "confusion_matrix.png",
                "feature_importance.png",
                "accuracy_per_class.png"
            ]
        }
        
        json_file = os.path.join(artifact_dir, "training_summary.json")
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Training summary saved")
    
    def run_pipeline(self, data_path="kopi_preprocessed.csv"):
        """
        Jalankan seluruh pipeline
        """
        try:
            print("\n" + "="*70)
            print("🚀 STARTING PIPELINE EXECUTION")
            print("="*70)
            
            # 1. Load data
            print("\n[1/4] 📂 Loading data...")
            df = self.load_preprocessed_data(data_path)
            
            # 2. Prepare features
            print("\n[2/4] 🔧 Preparing features...")
            X, y, label_encoder = self.prepare_features_target(df)
            
            # 3. Train model
            print("\n[3/4] 🚀 Training model...")
            model, run_id, artifact_dir = self.train_model(X, y, label_encoder)
            
            # 4. Display results
            print("\n[4/4] 📊 Displaying results...")
            self._display_results(run_id, artifact_dir)
            
            return model
            
        except Exception as e:
            print(f"\n❌ Error in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _display_results(self, run_id, artifact_dir):
        """Tampilkan hasil dan instruksi"""
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Tampilkan artifact yang dibuat
        if os.path.exists(artifact_dir):
            print(f"\n📁 Artifacts created in: {os.path.abspath(artifact_dir)}")
            
            # Tampilkan struktur folder
            print("\n📂 Folder Structure:")
            for root, dirs, files in os.walk(artifact_dir):
                level = root.replace(artifact_dir, '').count(os.sep)
                indent = '  ' * level
                print(f'{indent}📁 {os.path.basename(root) if root != artifact_dir else "artifacts"}')
                subindent = '  ' * (level + 1)
                for file in files:
                    if file not in ['.DS_Store', 'Thumbs.db']:  # Exclude system files
                        size = os.path.getsize(os.path.join(root, file)) / 1024
                        print(f'{subindent}📄 {file} ({size:.1f} KB)')
        
        # INSTRUKSI SCREENSHOT
        screenshot_instructions = f"""
        {'='*70}
        📸 INSTRUKSI SCREENSHOT UNTUK ARTIFAK.JPG
        {'='*70}
        
        LANGKAH 1: BUKA MLFLOW UI
        1. Terminal baru: mlflow ui --port 5000 --host 0.0.0.0
        2. Browser: http://localhost:5000
        
        LANGKAH 2: AMBIL SCREENSHOT
        Navigasi ke: Experiments → Coffee_Classification_Experiment → Run {run_id}
        
        YANG HARUS TERLIHAT DI SCREENSHOT (artifak.jpg):
        ┌─────────────────────────────────────────────────────────┐
        │ 📦 ARTIFACTS SECTION                                    │
        ├─────────────────────────────────────────────────────────┤
        │ 📁 mlflow_model/     (folder model dari MLflow)        │
        │   ├── MLmodel        ← WAJIB TERLIHAT!                 │
        │   ├── conda.yaml     ← WAJIB TERLIHAT!                 │
        │   ├── model.pkl                                        │
        │   └── requirements.txt                                 │
        │                                                         │
        │ 📁 model/           (folder model custom kita)         │
        │   ├── MLmodel        ← WAJIB TERLIHAT!                 │
        │   ├── conda.yaml     ← WAJIB TERLIHAT!                 │
        │   ├── model.pkl                                        │
        │   ├── requirements.txt                                 │
        │   ├── metadata.json                                    │
        │   └── input_example.json                               │
        │                                                         │
        │ 📄 confusion_matrix.png    (plot)                      │
        │ 📄 accuracy_per_class.png  (plot)                      │
        │ 📄 feature_importance.png  (plot)                      │
        │ 📄 feature_importance.json (data)                      │
        │ 📄 label_mapping.json      (data)                      │
        └─────────────────────────────────────────────────────────┘
        
        LANGKAH 3: JIKA MLFLOW UI ERROR
        Buka folder: {os.path.abspath(artifact_dir)}
        Screenshot isi folder yang menampilkan:
        - Folder 'model/' dengan file MLmodel dan conda.yaml
        - Semua file .png dan .json
        
        PASTIKAN FILE INI TERLIHAT DI SCREENSHOT:
        ✅ model/MLmodel
        ✅ model/conda.yaml
        ✅ confusion_matrix.png
        ✅ feature_importance.png
        {'='*70}
        """
        
        print(screenshot_instructions)
        
        # Simpan instruksi
        with open("screenshot_instructions.txt", "w") as f:
            f.write(screenshot_instructions)
        
        # Cek file penting
        print("\n🔍 CHECKING ESSENTIAL FILES:")
        essential_files_to_check = [
            os.path.join(artifact_dir, "model", "MLmodel"),
            os.path.join(artifact_dir, "model", "conda.yaml"),
            os.path.join(artifact_dir, "model", "model.pkl"),
            os.path.join(artifact_dir, "confusion_matrix.png"),
            os.path.join(artifact_dir, "feature_importance.png"),
        ]
        
        for file_path in essential_files_to_check:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / 1024
                print(f"  ✅ {os.path.basename(file_path)} ({size:.1f} KB)")
            else:
                print(f"  ❌ {os.path.basename(file_path)} - MISSING!")
        
        print(f"\n📸 Untuk screenshot, pastikan minimal 2 file ini terlihat:")
        print(f"   1. model/MLmodel")
        print(f"   2. model/conda.yaml")
        print(f"   3. Salah satu plot (.png file)")

def main():
    """Fungsi utama"""
    print("🔧 Initializing MLflow Coffee Classification...")
    
    try:
        # Pastikan folder mlruns ada
        os.makedirs("./mlruns", exist_ok=True)
        
        # Inisialisasi pipeline
        pipeline = CoffeeMLflowModel()
        
        # Jalankan pipeline
        model = pipeline.run_pipeline("kopi_preprocessed.csv")
        
        if model:
            print("\n✅ Script completed successfully!")
            print("\n📋 Next steps for screenshots:")
            print("   1. Open terminal and run: mlflow ui --port 5000")
            print("   2. Open browser to: http://localhost:5000")
            print("   3. Navigate to your run and take screenshot")
            print("   4. Save as: artifak.jpg")
            print("\n⚠️  IMPORTANT: Make sure 'MLmodel' and 'conda.yaml' are visible in screenshot!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Install PyYAML jika belum ada
    try:
        import yaml
    except ImportError:
        print("Installing PyYAML...")
        import subprocess
        subprocess.check_call(["pip", "install", "pyyaml"])
        import yaml
    
    main()