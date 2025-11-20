# 🚀 Quick Start Guide - Bharatanatyam Mudra Detection

**Get up and running in 30 minutes!**

---

## ⏱️ Time Estimate

- **Setup**: 10 minutes
- **Landmark Extraction**: 15-30 minutes
- **Training**: 1.5-2 hours
- **Testing**: 2 minutes

**Total**: ~2-3 hours (mostly automated)

---

## 📋 Before You Start

### ✅ Checklist

- [ ] Python 3.8-3.11 installed
- [ ] Webcam available
- [ ] 5GB free disk space
- [ ] Bharatanatyam-Mudra-Dataset downloaded
- [ ] Stable internet (for initial package download)

### 🔍 Verify Python Version

```bash
python --version
# Should show: Python 3.8.x to 3.11.x
```

**If wrong version:**
- Download Python 3.10: https://www.python.org/downloads/
- ⚠️ Check "Add Python to PATH" during installation

---

## 🎯 Installation (10 minutes)

### Step 1: Create Project Folder

**Windows (Command Prompt):**
```cmd
mkdir mudra-detection
cd mudra-detection
```

**macOS/Linux (Terminal):**
```bash
mkdir mudra-detection
cd mudra-detection
```

### Step 2: Place Dataset

1. Download Bharatanatyam-Mudra-Dataset
2. Extract the ZIP file
3. Move the folder to your project:

```
mudra-detection/
└── Bharatanatyam-Mudra-Dataset-master/
    └── Images/
        ├── Alapadmam(1)/
        ├── Anjali(1)/
        └── ... (49 folders)
```

**Verify dataset:**
```bash
# Windows
dir Bharatanatyam-Mudra-Dataset-master\Images

# Mac/Linux
ls Bharatanatyam-Mudra-Dataset-master/Images
```

You should see 49 folders.

### Step 3: Create File Structure

Create these folders:
```bash
# Windows
mkdir scripts utils data models

# Mac/Linux
mkdir -p scripts utils data models
```

### Step 4: Create All Script Files

Create these files with the provided code:

**📄 requirements.txt**
```txt
tensorflow==2.15.0
opencv-python==4.9.0.80
mediapipe==0.10.9
numpy==1.24.3
scikit-learn==1.4.0
matplotlib==3.8.2
seaborn==0.13.1
tqdm==4.66.1
Pillow==10.2.0
```

**📄 scripts/1_extract_landmarks.py** - Copy from artifact

**📄 scripts/2_train_model.py** - Copy from artifact (fixed version)

**📄 scripts/3_live_detection.py** - Copy from artifact

**📄 utils/hand_tracker.py** - Copy from artifact

**📄 run.py** - Copy from artifact

### Step 5: Create Virtual Environment

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in your prompt.

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your prompt.

### Step 6: Install Packages

```bash
pip install -r requirements.txt
```

**This will take 5-10 minutes.** You'll see:
```
Collecting tensorflow==2.15.0
  Downloading tensorflow-2.15.0...
Installing collected packages: ...
Successfully installed tensorflow-2.15.0 opencv-python-4.9.0.80 ...
```

### Step 7: Verify Installation

```bash
python -c "import cv2, mediapipe, tensorflow; print('✓ Success! All packages installed.')"
```

**Expected output:**
```
✓ Success! All packages installed.
```

**If you see errors**, check [Troubleshooting](#troubleshooting) section.

---

## 🎬 Running the System

### 🎯 Option 1: Complete Automated Pipeline (Easiest)

**Run everything with one command:**
```bash
python run.py --all
```

This will:
1. ✅ Extract landmarks (~30 min)
2. ✅ Train both models (~1.5 hours)
3. ✅ Start live detection

**Just wait and watch!** ☕

---

### 🎯 Option 2: Step-by-Step (Recommended First Time)

This gives you more control and lets you check each step.

#### Step 1: Extract Hand Landmarks

```bash
python scripts/1_extract_landmarks.py
```

**What you'll see:**
```
Processing mudras: 100%|█████████████████| 49/49
  Alapadmam: 100%|███████████████████| 568/568
  Anjali: 100%|██████████████████████| 593/593
  ...

======================================================================
Processing Complete!
======================================================================
Single hand mudras: 19807 images, 28 classes
Double hand mudras: 4431 images, 21 classes
Total processed: 24238
Failed: 193

Data saved to: data/processed/landmarks
Visualizations saved to: data/processed/visualizations
```

**Time:** 15-30 minutes

**✅ Success indicators:**
- Total processed > 20,000
- Failed < 500
- Files created in `data/processed/landmarks/`

**❌ If it fails:**
- Check dataset path is correct
- Ensure Images folder contains 49 subfolders
- See [Troubleshooting](#troubleshooting)

#### Step 2: Train the Models

```bash
python scripts/2_train_model.py
```

**What you'll see:**
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
...
=================================================================
Total params: 567,580

Starting training...
Epochs: 200, Batch size: 32

Epoch 1/200
1981/1981 [==============================] - 15s 7ms/step - loss: 2.1543 - accuracy: 0.4321 - val_loss: 1.8234 - val_accuracy: 0.5123
Epoch 2/200
1981/1981 [==============================] - 12s 6ms/step - loss: 1.5432 - accuracy: 0.6234 - val_loss: 1.3456 - val_accuracy: 0.6543
...
Epoch 87/200
1981/1981 [==============================] - 12s 6ms/step - loss: 0.1234 - accuracy: 0.9724 - val_loss: 0.2345 - val_accuracy: 0.9487

Restoring model weights from the end of the best epoch: 72.
Epoch 87: early stopping

Training Results:
  Accuracy: 97.24%
  Top-3 Accuracy: 99.51%

Validation Results:
  Accuracy: 94.87%
  Top-3 Accuracy: 98.93%
```

**Then it trains the double hand model** (similar output).

**Time:** 1.5-2 hours (45-60 min per model)

**✅ Success indicators:**
- Validation accuracy > 90%
- Training completes without errors
- Files created in `models/` folder

**❌ If accuracy is low (<85%):**
- See [Improving Accuracy](#improving-accuracy)
- Check confusion matrix images
- Verify dataset quality

#### Step 3: Run Live Detection

```bash
python scripts/3_live_detection.py
```

**What you'll see:**

```
Enhanced Bharatanatyam Mudra Detection System
======================================================================
Available models:
  Single hand: 28 classes
  Double hand: 21 classes

Camera: 0

Controls:
  Q - Quit
  S - Save screenshot
  R - Reset prediction history
  C - Clear screen
======================================================================

[Camera window opens]
```

**In the camera window:**
- Show your hand(s)
- See landmarks appear (green lines and dots)
- See mudra name and confidence
- See top 5 predictions

**Controls:**
- Press `Q` to quit
- Press `S` to save screenshot
- Press `R` if predictions seem stuck

**Time:** Runs until you quit

---

## 🎨 Understanding the Interface

```
┌───────────────────────────────────────────────────────────┐
│ Bharatanatyam Mudra Detection           FPS: 45.3         │
│ Mode: Single Hand                    Confidence: 94.2%    │
│                                                            │
│ Mudra: Pathaka                                            │
│ CONFIDENT                                                 │
├────────────────────────────────────┬──────────────────────┤
│                                    │ Top Predictions:     │
│                                    │ 1. Pathaka    94.2%  │
│   [Your hand with landmarks]       │ 2. Tripataka   3.1%  │
│   [Green dots and lines]           │ 3. Ardhapat    1.8%  │
│   [Green bounding box]             │ 4. Kapitham    0.5%  │
│                                    │ 5. Mukula      0.4%  │
│                                    │                      │
├────────────────────────────────────┴──────────────────────┤
│ Q: Quit | S: Screenshot | R: Reset | C: Clear            │
└───────────────────────────────────────────────────────────┘
```

### UI Elements:

1. **Mode**: Single Hand or Double Hand (auto-detected)
2. **FPS**: Frames per second (30-60 is good)
3. **Mudra Name**: Detected mudra (green = confident, orange = uncertain)
4. **Confidence**: How sure the model is (aim for >70%)
5. **Hand Landmarks**: 21 green dots showing hand keypoints
6. **Bounding Box**: Green rectangle around detected hand
7. **Top Predictions**: 5 most likely mudras with percentages

---

## 💡 Tips for Best Results

### 1. Lighting
- ✅ Use bright, even lighting
- ✅ Light from front or top
- ❌ Avoid backlighting (light behind you)
- ❌ Avoid direct sunlight (harsh shadows)

### 2. Hand Position
- ✅ Keep hands 1-2 feet from camera
- ✅ Show palm side to camera
- ✅ Keep hands clearly visible
- ❌ Don't overlap hands too much (for double hand mudras)
- ❌ Don't move too fast

### 3. Camera Setup
- ✅ Stable camera position
- ✅ Clean camera lens
- ✅ Good camera angle (slightly above eye level)
- ❌ Avoid cluttered background

### 4. Making Mudras
- ✅ Hold mudra steady for 1-2 seconds
- ✅ Make clear, defined gestures
- ✅ Follow traditional mudra forms
- ❌ Don't make partial or ambiguous gestures

---

## 🎓 Testing the System

### Test Sequence

Try these mudras in order (from easiest to detect):

**Single Hand:**
1. **Pathaka** (flat palm, all fingers together)
2. **Mushti** (closed fist)
3. **Tripataka** (index and middle up, others down)
4. **Ardhachandra** (thumb extended, others curved)
5. **Katakamukha** (thumb, ring, pinky touch; index, middle extended)

**Double Hand:**
1. **Anjali** (prayer hands)
2. **Pushpaputa** (cupped hands together)
3. **Matsya** (fish shape)
4. **Garuda** (bird shape)
5. **Swastika** (crossed wrists)

### Expected Results

| Mudra | Expected Confidence | Notes |
|-------|-------------------|-------|
| Pathaka | 95%+ | Very distinct, easy to detect |
| Mushti | 95%+ | Simple closed fist |
| Anjali | 98%+ | Most recognized mudra |
| Tripataka | 85-95% | May confuse with Ardhapathaka |
| Katakamukha | 80-90% | Complex finger position |

**If confidence is consistently low (<70%):**
- Check lighting
- Hold mudra steadier
- Ensure correct mudra form
- See [Troubleshooting](#troubleshooting)

---

## 🔧 Common Issues & Quick Fixes

### Issue: "Camera Not Found"

**Quick Fix:**
```python
# Edit 3_live_detection.py, line 360
detector.run(camera_id=1)  # Try 1, 2, 3...
```

### Issue: "No hands detected"

**Quick Fixes:**
1. Improve lighting
2. Move hands closer to camera
3. Lower detection threshold:
   ```python
   # Edit 3_live_detection.py, line 105
   min_detection_confidence=0.5  # Was 0.7
   ```

### Issue: "Wrong mudra detected"

**Quick Fixes:**
1. Hold mudra steadier
2. Check mudra form is correct
3. Ensure good lighting
4. Press `R` to reset prediction history

### Issue: "Jittery predictions"

**Quick Fix:**
```python
# Edit 3_live_detection.py, line 124
self.prediction_history = deque(maxlen=10)  # Was 7
```

### Issue: "Low FPS (<20)"

**Quick Fixes:**
1. Close other applications
2. Lower camera resolution:
   ```python
   # Edit 3_live_detection.py, lines 291-292
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
   ```

### Issue: "Out of memory during training"

**Quick Fix:**
```python
# Edit 2_train_model.py, line 466
batch_size=16  # Was 32
```

---

## 📊 Checking Your Results

### After Extraction (Step 1)

**Check:**
```bash
# Should have these files:
data/processed/landmarks/
├── dataset_enhanced.pkl
├── metadata_enhanced.json
└── visualizations/ (with images)
```

**Verify metadata:**
```bash
# Windows
type data\processed\landmarks\metadata_enhanced.json

# Mac/Linux
cat data/processed/landmarks/metadata_enhanced.json
```

**Should show:**
```json
{
  "total_single_hand": 19807,
  "total_double_hand": 4431,
  "total_images": 24238,
  "num_classes_single": 28,
  "num_classes_double": 21
}
```

### After Training (Step 2)

**Check:**
```bash
# Should have these files:
models/
├── mudra_model_single.h5
├── mudra_model_double.h5
├── label_encoder_single.pkl
├── label_encoder_double.pkl
└── confusion_matrix_single.png
```

**View confusion matrix:**
- Open `models/confusion_matrix_single.png`
- Bright diagonal = good
- Dark off-diagonal = confusions between classes

**Check accuracy:**
```bash
# Windows
type models\model_info_single.json

# Mac/Linux
cat models/model_info_single.json
```

**Should show:**
```json
{
  "num_classes": 28,
  "validation_accuracy": 0.94
}
```

### During Live Detection (Step 3)

**Good signs:**
- FPS: 30-60
- Confidence: >70% for clear mudras
- Smooth predictions (not jumping)
- Quick detection (<0.5s)

**Bad signs:**
- FPS: <20
- Confidence: <50% consistently
- Predictions jumping rapidly
- No detection after 5+ seconds

---

## 🎯 Next Steps

### After Basic Setup

1. **Test all mudras** - Try each of the 49 mudras
2. **Save examples** - Press `S` to save screenshots of good detections
3. **Check problematic mudras** - Note which mudras are confused
4. **Review confusion matrix** - See which pairs need improvement

### Improving Performance

1. **Increase augmentation**:
   ```python
   # In 2_train_model.py, line 456
   augment_factor=5  # Was 3
   ```

2. **Train longer**:
   ```python
   # In 2_train_model.py, line 465
   epochs=300  # Was 200
   ```

3. **Adjust detection settings** - See [Configuration](#configuration)

### Advanced Usage

- Process video files
- Batch process images
- Export model for mobile
- Create custom datasets
- Train on specific mudra pairs

See **README.md** for advanced topics.

---

## 📞 Getting Help

### Before Asking for Help

1. ✅ Checked this guide
2. ✅ Checked README.md
3. ✅ Checked ACCURACY_IMPROVEMENTS.md
4. ✅ Verified all files are in correct locations
5. ✅ Tried common fixes above

### When Asking for Help

Include:
- Operating System (Windows/Mac/Linux)
- Python version (`python --version`)
- Error message (full text)
- What you were doing
- What you've tried

### Where to Get Help

- GitHub Issues (for bugs)
- README.md (detailed documentation)
- ACCURACY_IMPROVEMENTS.md (technical details)

---

## ✅ Success Checklist

### Installation Complete ✓
- [ ] Python 3.8-3.11 installed
- [ ] Virtual environment created
- [ ] All packages installed
- [ ] Dataset in correct location
- [ ] All script files created

### Data Processing Complete ✓
- [ ] Landmarks extracted
- [ ] 20,000+ images processed
- [ ] Visualization samples created
- [ ] No critical errors

### Training Complete ✓
- [ ] Single hand model trained
- [ ] Double hand model trained
- [ ] Validation accuracy >90%
- [ ] Model files created
- [ ] Confusion matrices generated

### System Working ✓
- [ ] Camera detected
- [ ] Hands detected
- [ ] Mudras classified correctly
- [ ] Confidence >70% for clear gestures
- [ ] FPS >30

---

## 🎉 You're All Set!

**Congratulations!** You now have a working Bharatanatyam mudra detection system.

### What You Can Do:
- 🎯 Practice mudras with real-time feedback
- 📸 Record your progress with screenshots
- 📊 Analyze accuracy with confusion matrices
- 🎓 Learn new mudras with visual guidance
- 🤖 Build applications using the models

### Share Your Results:
- Take screenshots of successful detections
- Note which mudras work best
- Report any issues or suggestions
- Help improve the system!

---

**Happy Dancing! 💃🕺**

For detailed information, see **README.md**  
For technical details, see **ACCURACY_IMPROVEMENTS.md**
