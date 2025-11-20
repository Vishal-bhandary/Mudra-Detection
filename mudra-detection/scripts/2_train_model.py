"""
Enhanced training with separate models - FIXED VERSION
Compatible with enhanced landmark extraction
"""
import pickle
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataAugmenter:
    """Data augmentation for hand landmarks"""
    
    @staticmethod
    def add_noise(features, noise_level=0.02):
        """Add Gaussian noise to features"""
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    @staticmethod
    def scale(features, scale_range=(0.9, 1.1)):
        """Scale features randomly"""
        scale_factor = np.random.uniform(*scale_range)
        return features * scale_factor
    
    @staticmethod
    def rotate_2d(landmarks, angle_range=(-15, 15)):
        """Rotate hand landmarks in 2D plane"""
        # Landmarks should be (21*3=63,) shaped
        if len(landmarks) != 63:
            return landmarks
        
        landmarks_reshaped = landmarks.reshape(-1, 3)
        angle = np.radians(np.random.uniform(*angle_range))
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        rotated = landmarks_reshaped @ rotation_matrix.T
        return rotated.flatten()
    
    @staticmethod
    def augment(features, landmarks_size=63):
        """Apply random augmentation"""
        # Extract landmarks part (first 63 values per hand)
        # For single hand: first 63, for double: first 126
        features = features.copy()
        
        # Augment first hand landmarks
        if len(features) >= 63:
            landmarks1 = features[:63].copy()
            if np.random.random() > 0.5:
                landmarks1 = DataAugmenter.rotate_2d(landmarks1)
            if np.random.random() > 0.5:
                landmarks1 = DataAugmenter.add_noise(landmarks1)
            if np.random.random() > 0.5:
                landmarks1 = DataAugmenter.scale(landmarks1)
            features[:63] = landmarks1
        
        # Augment second hand landmarks if present
        if len(features) >= 126:
            landmarks2 = features[63:126].copy()
            if np.random.random() > 0.5:
                landmarks2 = DataAugmenter.rotate_2d(landmarks2)
            if np.random.random() > 0.5:
                landmarks2 = DataAugmenter.add_noise(landmarks2)
            if np.random.random() > 0.5:
                landmarks2 = DataAugmenter.scale(landmarks2)
            features[63:126] = landmarks2
        
        return features

class EnhancedMudraClassifier:
    def __init__(self, model_type='single'):
        """
        model_type: 'single' or 'double'
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        self.class_weights = None
        
    def prepare_data(self, dataset, augment=True, augment_factor=3):
        """Prepare data with enhanced features and augmentation"""
        
        data_list = dataset[f'{self.model_type}_hand']
        
        if len(data_list) == 0:
            raise ValueError(f"No {self.model_type} hand data found!")
        
        X = []
        y = []
        
        print(f"\nPreparing {self.model_type} hand data...")
        print(f"Total samples: {len(data_list)}")
        
        # Inspect first entry to understand structure
        if len(data_list) > 0:
            print("\nInspecting data structure...")
            sample_entry = data_list[0]
            print(f"Entry keys: {sample_entry.keys()}")
            print(f"Features structure: {type(sample_entry['features'])}")
            if isinstance(sample_entry['features'], list) and len(sample_entry['features']) > 0:
                print(f"First feature keys: {sample_entry['features'][0].keys() if isinstance(sample_entry['features'][0], dict) else 'Not a dict'}")
        
        # Process each entry
        for entry in data_list:
            try:
                features_list = []
                
                # Extract features from each hand
                for hand_features in entry['features']:
                    # Landmarks (normalized, 21 points * 3 = 63)
                    landmarks = hand_features['landmarks']
                    # Distances (10 values)
                    distances = hand_features['distances']
                    # Angles (15 values)
                    angles = hand_features['angles']
                    
                    # Combine: 63 + 10 + 15 = 88 features per hand
                    features_list.extend(landmarks)
                    features_list.extend(distances)
                    features_list.extend(angles)
                
                # For single hand: pad to match double hand size
                # For double hand: already has 2 hands worth of features
                expected_size = 2 * (63 + 10 + 15)  # 176 features
                
                if len(features_list) < expected_size:
                    features_list.extend([0] * (expected_size - len(features_list)))
                elif len(features_list) > expected_size:
                    features_list = features_list[:expected_size]
                
                X.append(features_list)
                y.append(entry['mudra_name'])
                
            except Exception as e:
                print(f"Error processing entry: {e}")
                print(f"Entry structure: {entry}")
                continue
        
        if len(X) == 0:
            raise ValueError("No valid data extracted! Check data format.")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Original data shape: {X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_[:10]}..." if len(self.label_encoder.classes_) > 10 else f"Classes: {self.label_encoder.classes_}")
        
        # Data augmentation
        if augment and augment_factor > 0:
            print(f"Applying data augmentation (factor: {augment_factor})...")
            X_aug = []
            y_aug = []
            
            for i in range(len(X)):
                # Add original
                X_aug.append(X[i])
                y_aug.append(y_encoded[i])
                
                # Create augmented samples
                for _ in range(augment_factor):
                    aug_sample = DataAugmenter.augment(X[i])
                    X_aug.append(aug_sample)
                    y_aug.append(y_encoded[i])
            
            X = np.array(X_aug, dtype=np.float32)
            y_encoded = np.array(y_aug)
            
            print(f"Augmented data shape: {X.shape}")
        
        # Calculate class weights to handle imbalance
        unique_classes = np.unique(y_encoded)
        self.class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_encoded
        )
        self.class_weights = dict(enumerate(self.class_weights))
        
        print(f"Class weights calculated for {len(self.class_weights)} classes")
        
        return X, y_encoded
    
    def build_model(self, input_shape, num_classes):
        """Build enhanced neural network with better architecture"""
        
        inputs = keras.Input(shape=(input_shape,))
        
        # First block - larger capacity
        x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        # Second block
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        # Third block
        x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Fourth block
        x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Use Adam optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=200, batch_size=32):
        """Train the model with enhanced callbacks"""
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model
        num_classes = len(np.unique(y))
        self.model = self.build_model(X.shape[1], num_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/best_{self.model_type}_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with class weights
        print("\nStarting training...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Using class weights: {len(self.class_weights)} classes")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Load best model
        if os.path.exists(f'models/best_{self.model_type}_model.h5'):
            self.model = keras.models.load_model(f'models/best_{self.model_type}_model.h5')
            print("\n✓ Loaded best model from checkpoint")
        
        # Evaluate
        print("\nEvaluating model...")
        train_results = self.model.evaluate(X_train, y_train, verbose=0)
        val_results = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n{'='*60}")
        print(f"Training Results:")
        print(f"  Accuracy: {train_results[1]*100:.2f}%")
        print(f"  Top-3 Accuracy: {train_results[2]*100:.2f}%")
        print(f"\nValidation Results:")
        print(f"  Accuracy: {val_results[1]*100:.2f}%")
        print(f"  Top-3 Accuracy: {val_results[2]*100:.2f}%")
        print(f"{'='*60}")
        
        return X_val, y_val
    
    def plot_training_history(self, save_path):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary text
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_top3_acc = self.history.history['val_top_3_accuracy'][-1]
        
        summary_text = f"""
        Final Results:
        
        Training Accuracy: {final_train_acc*100:.2f}%
        Validation Accuracy: {final_val_acc*100:.2f}%
        Top-3 Accuracy: {final_top3_acc*100:.2f}%
        
        Total Epochs: {len(self.history.history['accuracy'])}
        Model Type: {self.model_type.upper()}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.close()
    
    def evaluate_detailed(self, X_val, y_val, save_dir):
        """Generate detailed evaluation metrics"""
        
        print("\nGenerating detailed evaluation...")
        
        # Predictions
        y_pred_probs = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Classification report
        report = classification_report(
            y_val, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Save report
        report_path = os.path.join(save_dir, f'classification_report_{self.model_type}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nClassification Report (Top 10 classes by F1-score):")
        # Sort by f1-score and show top 10
        class_scores = [(name, metrics['f1-score']) 
                       for name, metrics in report.items() 
                       if isinstance(metrics, dict) and 'f1-score' in metrics]
        class_scores.sort(key=lambda x: x[1], reverse=True)
        
        for name, score in class_scores[:10]:
            print(f"  {name:30s}: {score:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        plt.figure(figsize=(max(20, len(self.label_encoder.classes_)), 
                           max(18, len(self.label_encoder.classes_))))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {self.model_type.title()} Hand Mudras', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = os.path.join(save_dir, f'confusion_matrix_{self.model_type}.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {cm_path}")
        plt.close()
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
        class_acc_dict = {
            self.label_encoder.classes_[i]: float(per_class_acc[i])
            for i in range(len(per_class_acc))
        }
        
        # Save per-class accuracy
        acc_path = os.path.join(save_dir, f'per_class_accuracy_{self.model_type}.json')
        with open(acc_path, 'w') as f:
            json.dump(class_acc_dict, f, indent=2)
        
        # Print worst performing classes
        print("\nWorst Performing Classes (Bottom 10):")
        sorted_acc = sorted(class_acc_dict.items(), key=lambda x: x[1])
        for mudra, acc in sorted_acc[:10]:
            print(f"  {mudra:30s}: {acc*100:.2f}%")
    
    def save_model(self, save_dir):
        """Save model and label encoder"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Keras model
        model_path = os.path.join(save_dir, f'mudra_model_{self.model_type}.h5')
        self.model.save(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(save_dir, f'label_encoder_{self.model_type}.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Label encoder saved to: {encoder_path}")
        
        # Save model info
        info = {
            'model_type': self.model_type,
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'input_shape': int(self.model.input_shape[1]),
            'class_weights': {int(k): float(v) for k, v in self.class_weights.items()}
        }
        
        info_path = os.path.join(save_dir, f'model_info_{self.model_type}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Model info saved to: {info_path}")

def main():
    # Paths
    DATA_PATH = "data/processed/landmarks/dataset_enhanced.pkl"
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at: {DATA_PATH}")
        print("\nPlease run: python scripts/1_extract_landmarks.py")
        sys.exit(1)
    
    print("="*70)
    print("ENHANCED MUDRA CLASSIFIER TRAINING")
    print("="*70)
    print(f"\nLoading dataset from: {DATA_PATH}")
    
    try:
        with open(DATA_PATH, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Single hand samples: {len(dataset['single_hand'])}")
    print(f"  Double hand samples: {len(dataset['double_hand'])}")
    
    # Train separate models for single and double hand
    for model_type in ['single', 'double']:
        print(f"\n{'='*70}")
        print(f"TRAINING {model_type.upper()} HAND MUDRA CLASSIFIER")
        print(f"{'='*70}")
        
        if len(dataset[f'{model_type}_hand']) == 0:
            print(f"⚠️  No {model_type} hand data found. Skipping...")
            continue
        
        try:
            # Initialize classifier
            classifier = EnhancedMudraClassifier(model_type=model_type)
            
            # Prepare data with augmentation
            X, y = classifier.prepare_data(
                dataset, 
                augment=True, 
                augment_factor=3  # Increase to 5 for more augmentation
            )
            
            # Train model
            X_val, y_val = classifier.train(
                X, y,
                validation_split=0.2,
                epochs=200,  # Increase to 300 for better results
                batch_size=32
            )
            
            # Plot training history
            classifier.plot_training_history(
                os.path.join(MODEL_DIR, f'training_history_{model_type}.png')
            )
            
            # Detailed evaluation
            classifier.evaluate_detailed(X_val, y_val, MODEL_DIR)
            
            # Save model
            classifier.save_model(MODEL_DIR)
            
            print(f"\n✓ {model_type.title()} hand model training complete!")
            
        except Exception as e:
            print(f"\n❌ ERROR training {model_type} hand model:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("✓ ALL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved in: {MODEL_DIR}/")
    print("\nNext step: Run live detection")
    print("  python scripts/3_live_detection.py")

if __name__ == "__main__":
    main()