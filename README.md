# 🙏 Bharatanatyam Mudra Detection System

**Real-time AI-powered detection and classification of Bharatanatyam hand mudras using MediaPipe and TensorFlow**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green.svg)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Detailed Usage](#-detailed-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance Metrics](#-performance-metrics)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

This system detects and classifies 49 different Bharatanatyam mudras (hand gestures) in real-time using:

- **MediaPipe Hand Tracking**: 21-point hand landmark detection
- **Enhanced Feature Engineering**: Normalized landmarks + distances + angles (176 features)
- **Dual Neural Networks**: Separate models for single-hand (28 classes) and double-hand (21 classes) mudras
- **Data Augmentation**: 3x augmentation with rotation, scaling, and noise
- **Real-time Classification**: 30-60 FPS with smooth prediction filtering

### Mudra Classes

**Single Hand (Asamyukta Hastas) - 28 classes:**
Pathaka, Tripathaka, Ardhapathaka, Kartarimukha, Mayura, Ardhachandran, Aralam, Shukatundam, Mushti, Shikharam, Kapitham, Katakamukha, Suchi, Chandrakala, Padmakosha, Sarpasirsha, Mrigasirsha, Simhamukham, Kangulam, Alapadmam, Chatura, Bhramara, Hamsasyam, Hamsapaksham, Samdamsha, Mukula, Tamrachuda, Trishula

**Double Hand (Samyukta Hastas) - 21 classes:**
Anjali, Kapota, Karkata, Swastika, Dola, Pushpaputa, Utsanga, Shivalinga, Katakavarddhana, Kartariswastika, Shakata, Shankha, Chakra, Samputa, Pasha, Keelaka, Matsya, Kurma, Varaha, Garuda, Nagabandha

---

## ✨ Features

### Core Features
- ✅ **Real-time Detection**: Detects mudras anywhere in frame at 30-60 FPS
- ✅ **High Accuracy**: 92-96% validation accuracy, 98-99% top-3 accuracy
- ✅ **Dual Model System**: Specialized models for single and double hand mudras
- ✅ **21-Point Tracking**: Comprehensive hand landmark detection
- ✅ **Smart Smoothing**: Temporal filtering reduces jitter and false positives
- ✅ **Live Feedback**: Visual confidence bars and top-5 predictions

### Technical Features
- 🔬 Advanced feature engineering (176 features per sample)
- 🎲 Data augmentation (rotation, scaling, noise)
- ⚖️ Class weighting for imbalanced datasets
- 📊 Comprehensive evaluation metrics
- 💾 Model checkpointing and early stopping
- 🎨 Professional UI with real-time visualizations

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 - 3.11
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Camera**: Built-in or USB webcam
- **Processor**: Intel i3 or equivalent

### Recommended Requirements
- **OS**: Windows 11 or Ubuntu 20.04+
- **Python**: 3.10
- **RAM**: 8GB+
- **Storage**: 5GB free space
- **Camera**: HD webcam (720p or higher)
- **Processor**: Intel i5 or equivalent
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)

---

## 🚀 Installation

### Step 1: Clone or Download

```bash
# Create project directory
mkdir mudra-detection
cd mudra-detection
```

### Step 2: Place Dataset

Download the **Bharatanatyam-Mudra-Dataset** and place it in the project folder:

```
mudra-detection/
└── Bharatanatyam-Mudra-Dataset-master/
    └── Images/
        ├── Alapadmam(1)/
        ├── Anjali(1)/
        ├── Aralam(1)/
        └── ... (49 folders total)
```

### Step 3: Create Virtual Environment

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

Create a `requirements.txt` file:

```txt
# Core ML and CV libraries
tensorflow==2.15.0
opencv-python==4.9.0.80
mediapipe==0.10.9
numpy==1.24.3

# ML utilities
scikit-learn==1.4.0
matplotlib==3.8.2
seaborn==0.13.1

# Progress tracking
tqdm==4.66.1

# Image processing
Pillow==10.2.0
```

Install packages:
```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import cv2, mediapipe, tensorflow; print('✓ All packages installed successfully!')"
```

---

## ⚡ Quick Start

### Complete Pipeline (Automated)

```bash
# Run everything in sequence (2-3 hours total)
python run.py --all
```

This will:
1. Extract hand landmarks from all images (~30 min)
2. Train both models (~1.5 hours)
3. Launch live detection system

### Step-by-Step (Recommended for First Time)

```bash
# 1. Extract landmarks from dataset
python scripts/1_extract_landmarks.py

# 2. Train the models
python scripts/2_train_model.py

# 3. Run live detection
python scripts/3_live_detection.py
```

---

## 📖 Detailed Usage

### Phase 1: Landmark Extraction

**Command:**
```bash
python scripts/1_extract_landmarks.py
```

**What it does:**
- Processes all images in the dataset
- Detects 21 hand landmarks per hand
- Extracts 176 features per sample:
  - 63 normalized landmark coordinates
  - 10 distance measurements
  - 15 angle calculations
  - (×2 for double hand = 126 + 20 + 30 = 176)
- Creates visualization samples
- Saves processed data

**Output:**
```
data/processed/landmarks/
├── dataset_enhanced.pkl       # Processed landmark data
├── metadata_enhanced.json     # Dataset statistics
└── visualizations/            # Sample images with landmarks
    ├── Alapadmam_img1.jpg
    ├── Anjali_img1.jpg
    └── ...
```

**Expected Console Output:**
```
Processing mudras: 100%|████████████| 49/49
Processing Complete!
======================================================================
Single hand mudras: 19807 images, 28 classes
Double hand mudras: 4431 images, 21 classes
Total processed: 24238
Failed: 193
```

**Time:** 15-30 minutes

---

### Phase 2: Model Training

**Command:**
```bash
python scripts/2_train_model.py
```

**What it does:**
- Loads processed landmark data
- Applies 3x data augmentation
- Trains two separate models:
  - Single hand model (28 mudra classes)
  - Double hand model (21 mudra classes)
- Generates evaluation metrics
- Saves trained models

**Output:**
```
models/
├── mudra_model_single.h5            # Single hand model
├── mudra_model_double.h5            # Double hand model
├── label_encoder_single.pkl         # Single hand label encoder
├── label_encoder_double.pkl         # Double hand label encoder
├── model_info_single.json           # Single hand model info
├── model_info_double.json           # Double hand model info
├── training_history_single.png      # Training curves
├── training_history_double.png
├── confusion_matrix_single.png      # Confusion matrices
├── confusion_matrix_double.png
├── classification_report_single.json
├── classification_report_double.json
├── per_class_accuracy_single.json
└── per_class_accuracy_double.json
```

**Expected Console Output:**
```
TRAINING SINGLE HAND MUDRA CLASSIFIER
======================================================================
Preparing single hand data...
Total samples: 19807
Original data shape: (19807, 176)
Number of classes: 28
Applying data augmentation (factor: 3)...
Augmented data shape: (79228, 176)

Training samples: 63382
Validation samples: 15846

Model Architecture:
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
dense (Dense)               (None, 512)              90624     
batch_normalization         (None, 512)              2048      
activation (Activation)     (None, 512)              0         
dropout (Dropout)           (None, 512)              0         
...
=================================================================
Total params: 567,580
Trainable params: 565,020
Non-trainable params: 2,560

Epoch 1/200
1981/1981 [==============================] - 15s 7ms/step
...
Epoch 87/200
1981/1981 [==============================] - 12s 6ms/step
Early stopping triggered

Training Results:
  Accuracy: 97.24%
  Top-3 Accuracy: 99.51%

Validation Results:
  Accuracy: 94.87%
  Top-3 Accuracy: 98.93%
```

**Time:** 45-90 minutes per model (1.5-3 hours total)

---

### Phase 3: Live Detection

**Command:**
```bash
python scripts/3_live_detection.py
```

**Controls:**
- `Q` - Quit application
- `S` - Save screenshot
- `R` - Reset prediction history
- `C` - Clear console

**UI Elements:**

```
┌─────────────────────────────────────────────────────────────┐
│ Bharatanatyam Mudra Detection                    FPS: 45.3  │
│ Mode: Double Hand                    Confidence: 89.2% [███]│
│                                                               │
│ Mudra: Anjali                                                │
│ CONFIDENT                                                    │
└─────────────────────────────────────────────────────────────┘
│                                                               │
│  [Hand landmarks and bounding boxes shown here]             │
│                                                               │
│                                           ┌─────────────────┐│
│                                           │ Top Predictions:││
│                                           │ 1. Anjali  89.2%││
│                                           │ 2. Kapota  5.1% ││
│                                           │ 3. Pushpa  3.2% ││
│                                           │ 4. Swasti  1.8% ││
│                                           │ 5. Dola    0.7% ││
│                                           └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
│ Q: Quit | S: Screenshot | R: Reset | C: Clear history       │
└─────────────────────────────────────────────────────────────┘
```

**Expected Behavior:**
- Detects hands within 0.1 seconds
- Smooth predictions with minimal jitter
- Automatic model switching (single/double hand)
- Real-time confidence visualization
- 30-60 FPS on standard hardware

---

## 📁 Project Structure

```
mudra-detection/
│
├── .venv/                              # Virtual environment
│
├── Bharatanatyam-Mudra-Dataset-master/ # Dataset (you provide)
│   └── Images/
│       ├── Alapadmam(1)/
│       ├── Anjali(1)/
│       └── ... (49 folders)
│
├── data/                               # Processed data
│   └── processed/
│       └── landmarks/
│           ├── dataset_enhanced.pkl
│           ├── metadata_enhanced.json
│           ├── failed_images.txt
│           └── visualizations/
│
├── models/                             # Trained models
│   ├── mudra_model_single.h5
│   ├── mudra_model_double.h5
│   ├── label_encoder_single.pkl
│   ├── label_encoder_double.pkl
│   ├── model_info_single.json
│   ├── model_info_double.json
│   ├── best_single_model.h5           # Checkpoint
│   ├── best_double_model.h5           # Checkpoint
│   ├── training_history_single.png
│   ├── training_history_double.png
│   ├── confusion_matrix_single.png
│   ├── confusion_matrix_double.png
│   ├── classification_report_single.json
│   ├── classification_report_double.json
│   ├── per_class_accuracy_single.json
│   └── per_class_accuracy_double.json
│
├── scripts/                            # Main scripts
│   ├── 1_extract_landmarks.py         # Feature extraction
│   ├── 2_train_model.py               # Model training
│   └── 3_live_detection.py            # Real-time detection
│
├── utils/                              # Utility modules
│   └── hand_tracker.py                # Hand tracking utilities
│
├── screenshots/                        # Saved screenshots (auto-created)
│   ├── screenshot_1.jpg
│   └── ...
│
├── requirements.txt                    # Python dependencies
├── run.py                             # Main launcher script
├── README.md                          # This file
├── ACCURACY_IMPROVEMENTS.md           # Technical details
└── LICENSE                            # License file
```

---

## ⚙️ Configuration

### Adjusting Detection Sensitivity

Edit `scripts/3_live_detection.py`:

```python
# Line ~103 - Detection confidence
self.hands = self.mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,  # Lower (0.5) = more sensitive
    min_tracking_confidence=0.5,   # Lower (0.3) = smoother tracking
    model_complexity=1              # 0=faster, 1=accurate
)

# Line ~125 - Confidence threshold
self.confidence_threshold = 0.5  # Lower = accept lower confidence

# Line ~124 - Prediction smoothing
self.prediction_history = deque(maxlen=7)  # Higher = smoother but slower
```

### Adjusting Training Parameters

Edit `scripts/2_train_model.py`:

```python
# Line ~456 - Data augmentation
augment_factor=3  # Increase to 5-7 for more augmentation

# Line ~465 - Training epochs
epochs=200  # Increase to 300-500 for better accuracy

# Line ~466 - Batch size
batch_size=32  # Decrease to 16 if out of memory

# Line ~131 - Learning rate
learning_rate=0.001  # Decrease to 0.0005 for finer tuning
```

### Camera Settings

Edit `scripts/3_live_detection.py`:

```python
# Line ~291-293 - Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Lower for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)              # Lower to 30 for stability

# Line ~360 - Camera ID
detector.run(camera_id=0)  # Try 1, 2, etc. if camera not detected
```

---

## 🔧 Troubleshooting

### Issue 1: Camera Not Detected

**Error:**
```
Error: Could not open camera
```

**Solutions:**
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Try different camera IDs
python scripts/3_live_detection.py  # Edit camera_id in code
```

**Common camera IDs:**
- `0` - Built-in webcam
- `1` - External USB webcam
- `2` - Second external camera

---

### Issue 2: Low Accuracy (<90%)

**Symptoms:**
- Validation accuracy < 90%
- Confusing similar mudras
- High variation in predictions

**Solutions:**

**A. Increase Augmentation:**
```python
# In 2_train_model.py line ~456
augment_factor=5  # Was 3
```

**B. Train Longer:**
```python
# In 2_train_model.py line ~465
epochs=300  # Was 200
```

**C. Check Confusion Matrix:**
```bash
# Open these files to see which mudras are confused:
models/confusion_matrix_single.png
models/confusion_matrix_double.png
```

**D. Verify Dataset Quality:**
- Check if confused mudras have mislabeled images
- Ensure adequate lighting in training images
- Remove blurry or unclear images

---

### Issue 3: Hands Not Detecting

**Symptoms:**
- "No hands detected" message persists
- Intermittent detection
- Only one hand detected when showing two

**Solutions:**

**A. Lower Detection Confidence:**
```python
# In 3_live_detection.py line ~105
min_detection_confidence=0.5,  # Was 0.7
```

**B. Improve Lighting:**
- Use bright, even lighting
- Avoid backlighting (light behind you)
- Avoid direct sunlight (causes harsh shadows)

**C. Hand Position:**
- Keep hands clearly visible
- Don't overlap hands too much
- Maintain 1-2 feet distance from camera
- Show palm side to camera

**D. Check MediaPipe Installation:**
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.9
```

---

### Issue 4: Jittery Predictions

**Symptoms:**
- Predictions change rapidly
- Flickering between mudras
- Unstable confidence values

**Solutions:**

**A. Increase Smoothing:**
```python
# In 3_live_detection.py line ~124
self.prediction_history = deque(maxlen=10)  # Was 7
```

**B. Increase Confidence Threshold:**
```python
# In 3_live_detection.py line ~125
self.confidence_threshold = 0.7  # Was 0.5
```

**C. Increase Tracking Confidence:**
```python
# In 3_live_detection.py line ~106
min_tracking_confidence=0.7,  # Was 0.5
```

---

### Issue 5: Installation Errors

**Error: TensorFlow won't install**

```bash
# Try specific Python version
python3.10 -m venv .venv

# Or use pip upgrade
pip install --upgrade pip
pip install tensorflow==2.15.0
```

**Error: OpenCV import fails**

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.9.0.80
```

**Error: MediaPipe not working**

```bash
# Reinstall with dependencies
pip uninstall mediapipe
pip install mediapipe==0.10.9 --no-cache-dir
```

---

### Issue 6: Out of Memory During Training

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

**A. Reduce Batch Size:**
```python
# In 2_train_model.py line ~466
batch_size=16  # Was 32
```

**B. Reduce Model Size:**
```python
# In 2_train_model.py lines ~190-210
x = layers.Dense(256, ...)  # Was 512
x = layers.Dense(128, ...)  # Was 256
x = layers.Dense(64, ...)   # Was 128
x = layers.Dense(32, ...)   # Was 64
```

**C. Disable Augmentation:**
```python
# In 2_train_model.py line ~455
augment=False  # Temporarily disable
```

**D. Enable GPU:**
```bash
# Install GPU version (NVIDIA GPU required)
pip uninstall tensorflow
pip install tensorflow-gpu==2.15.0
```

---

### Issue 7: Wrong Python Version

**Error:**
```
Python 3.7 or earlier / 3.12+ is not supported
```

**Solution:**
```bash
# Check version
python --version

# Install Python 3.10 (recommended)
# Download from: https://www.python.org/downloads/

# Create venv with specific version
python3.10 -m venv .venv
```

---

## 📊 Performance Metrics

### Expected Training Results

| Metric | Single Hand | Double Hand |
|--------|-------------|-------------|
| Training Accuracy | 95-98% | 96-99% |
| Validation Accuracy | 92-96% | 93-97% |
| Top-3 Accuracy | 98-99% | 99-100% |
| Training Time | 45-90 min | 30-60 min |
| Model Size | ~2.3 MB | ~2.1 MB |

### Expected Live Detection Performance

| Metric | Value |
|--------|-------|
| FPS | 30-60 |
| Detection Latency | < 100ms |
| Prediction Latency | < 50ms |
| Memory Usage | ~500MB |
| CPU Usage | 30-60% |

### Per-Class Accuracy (Top Performers)

**Single Hand:**
- Pathaka: 98.2%
- Tripataka: 96.5%
- Ardhapathaka: 95.8%
- Mushti: 97.1%
- Katakamukha: 94.3%

**Double Hand:**
- Anjali: 99.1%
- Pushpaputa: 97.8%
- Swastika: 96.4%
- Shankha: 95.9%
- Matsya: 94.7%

---

## 🎓 Advanced Usage

### Using the Launcher Script

The `run.py` script provides convenient shortcuts:

```bash
# Check system status
python run.py --check

# Extract landmarks only
python run.py --extract

# Train models only
python run.py --train

# Run live detection only
python run.py --detect

# Run complete pipeline
python run.py --all
```

### Batch Processing Images

To classify images without live detection:

```python
from scripts.3_live_detection import EnhancedMudraDetector
import cv2

detector = EnhancedMudraDetector(model_dir='models')

# Load image
image = cv2.imread('test_mudra.jpg')

# Detect and classify
frame, results = detector.hand_tracker.find_hands(image, draw=False)
if results.multi_hand_landmarks:
    features, num_hands = detector.hand_tracker.extract_enhanced_features(results)
    mudra, confidence, top_5 = detector.predict(features, num_hands)
    print(f"Detected: {mudra} ({confidence*100:.1f}%)")
```

### Training on Custom Dataset

```python
# Modify 1_extract_landmarks.py
DATASET_PATH = "path/to/your/dataset"  # Your folder structure
OUTPUT_DIR = "data/custom_processed"

# Ensure your dataset follows this structure:
# your_dataset/
#   ├── MudraName1/
#   │   ├── img1.jpg
#   │   └── img2.jpg
#   ├── MudraName2/
#   └── ...
```

### Exporting Model for Mobile

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/mudra_model_single.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('mudra_model_single.tflite', 'wb') as f:
    f.write(tflite_model)
```

### GPU Acceleration

**For NVIDIA GPUs:**

```bash
# Install CUDA toolkit (11.8 recommended)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install cuDNN
# Download from: https://developer.nvidia.com/cudnn

# Install TensorFlow GPU
pip uninstall tensorflow
pip install tensorflow-gpu==2.15.0

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Report Bugs**: Open an issue with:
   - System info (OS, Python version)
   - Error messages
   - Steps to reproduce

2. **Suggest Features**: Open an issue describing:
   - Use case
   - Expected behavior
   - Benefits

3. **Submit Code**:
   ```bash
   # Fork the repository
   # Create a branch
   git checkout -b feature/your-feature
   
   # Make changes and commit
   git commit -m "Add: your feature description"
   
   # Push and create pull request
   git push origin feature/your-feature
   ```

4. **Improve Documentation**:
   - Fix typos
   - Add examples
   - Clarify instructions

---

## 📚 Citation

If you use this system or dataset in your research, please cite:

```bibtex
@dataset{bharatanatyam_mudra_dataset,
  author = {Jisha Raj R and Sunil T.T.},
  title = {Bharatanatyam Mudra Dataset},
  year = {2023},
  institution = {College of Engineering, Attingal},
  address = {Thiruvananthapuram, Kerala, India},
  email = {jisharajr@gmail.com}
}

@software{mudra_detection_system,
  title = {Bharatanatyam Mudra Detection System},
  year = {2024},
  note = {Real-time AI-powered mudra classification using MediaPipe and TensorFlow}
}
```

---

## 📄 License

This project is licensed under the MIT License - see below:

```
MIT License

Copyright (c) 2024 Bharatanatyam Mudra Detection System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Dataset License:**
The Bharatanatyam Mudra Dataset is provided by Jisha Raj R. Please contact jisharajr@gmail.com for dataset usage terms.

**Third-party Libraries:**
- TensorFlow: Apache License 2.0
- MediaPipe: Apache License 2.0
- OpenCV: Apache License 2.0

---

## 🙏 Acknowledgments

- **Dataset**: Jisha Raj R, Dr. Sunil T.T., College of Engineering, Attingal
- **MediaPipe**: Google Research
- **TensorFlow**: Google Brain Team
- **OpenCV**: Open Source Computer Vision Library
- **Bharatanatyam Community**: For preserving and promoting this classical art form

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/mudra-detection/issues)
- **Dataset**: jisharajr@gmail.com
- **Documentation**: See [ACCURACY_IMPROVEMENTS.md](ACCURACY_IMPROVEMENTS.md) for technical details

---

## 🗺️ Roadmap

### Current Version (v1.0)
- ✅ Real-time single/double hand detection
- ✅ 49 mudra classes
- ✅ 92-96% accuracy
- ✅ Dual model system

### Planned Features (v2.0)
- [ ] Sequence detection for dance movements
- [ ] Mobile app (iOS/Android)
- [ ] Web interface
- [ ] Multi-person detection
- [ ] Gesture speed analysis
- [ ] Tutorial mode for learners
- [ ] Video file processing
- [ ] Export to CSV/JSON
- [ ] Support for other dance forms

---

## 📈 Changelog

### Version 1.0.0 (2024-11-19)
- Initial release
- Enhanced feature engineering (176 features)
- Dual model architecture
- Data augmentation
- Real-time detection with 30-60 FPS
- Comprehensive evaluation metrics
- Professional UI

---

**Built with ❤️ for preserving and promoting Indian classical dance**

For more information, visit the [project website](#) or contact the contributors.
