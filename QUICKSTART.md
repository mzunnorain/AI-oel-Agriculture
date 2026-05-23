# Quick Start Guide

## 5-Minute Setup for Smart Agriculture Decision Support System

### Prerequisites

- Python 3.8+
- pip package manager
- 2GB disk space
- Internet connection (for first-time setup)

---

## Step 1: Clone & Navigate (1 min)

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-agriculture-dss.git
cd smart-agriculture-dss

# Verify directory structure
ls -la
# Expected: data/, src/, models/, results/, *.py files, README.md
```

---

## Step 2: Create Virtual Environment (1 min)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Verify activation (should show 'venv' in prompt)
```

---

## Step 3: Install Dependencies (2 min)

```bash
pip install -r requirements.txt

# Verify installation
python -c "import pandas, sklearn, matplotlib; print('✓ All dependencies installed')"
```

---

## Step 4: Prepare Dataset (30 sec)

```bash
# Download/create sample dataset
python prepare_dataset.py

# Expected output:
# Creating sample dataset...
# Sample dataset created at data/agricultural_data.csv
# Dataset shape: (2200, 9)
```

---

## Step 5: Train Models (3-5 min)

```bash
# Train all three ML models
python train_models.py

# Expected output:
# Loading and preprocessing data...
# ============================================================
# TRAINING ALL MODELS
# ============================================================
# 1. Training Decision Tree Classifier...
# 2. Training KMeans Clustering...
# 3. Training Linear Regression...
# ✓ All models trained and saved successfully!
```

**What happens:**
- Models trained on 1,760 samples (80%)
- Evaluated on 440 test samples (20%)
- Serialized models saved to `models/` directory
- Performance metrics logged to `results/model_metrics.json`

---

## Step 6: Launch Interactive GUI (Now!)

```bash
# Start the Tkinter application
python src/gui.py

# GUI appears with 4 tabs:
# 1. Prediction - Enter soil parameters and get recommendations
# 2. Model Metrics - View model performance statistics
# 3. Visualizations - See charts (feature importance, clusters, residuals)
# 4. About - System information
```

---

## Example Prediction Walkthrough

### Input Example

In the **Prediction Tab**, enter these values:

```
Nitrogen:     80 mg/kg
Phosphorus:   50 mg/kg
Potassium:    100 mg/kg
Temperature:  25°C
Humidity:     70%
pH:           6.5
Rainfall:     100 mm
```

Click **"Predict"** button

### Expected Output

```
======================================================================
INTEGRATED AGRICULTURAL DECISION SUPPORT OUTPUT
======================================================================

1. CROP RECOMMENDATION (Decision Tree Classifier)
   Recommended Crop: Rice
   Confidence: 97.5%

2. SOIL ZONE CLASSIFICATION (KMeans Clustering)
   Assigned Cluster: 1
   Agronomic Guidance: Zone 1 - Optimize management practices

3. CROP YIELD PREDICTION (Linear Regression)
   Predicted Yield: 3850 kg/hectare
   Confidence Bounds: [3670 - 4030]

======================================================================
```

---

## Troubleshooting Quick Reference

### Issue: "Module not found"

```bash
# Solution: Reinstall requirements
pip install --upgrade -r requirements.txt
pip list  # Verify installation
```

### Issue: "agricultural_data.csv not found"

```bash
# Solution: Create sample dataset
python prepare_dataset.py
ls data/  # Verify file exists
```

### Issue: "Models not found during prediction"

```bash
# Solution: Train models first
python train_models.py
ls models/  # Should show 3 .pkl files
```

### Issue: "GUI won't open on Ubuntu/Linux"

```bash
# Solution: Install tkinter
sudo apt-get install python3-tk
python src/gui.py
```

### Issue: "ImportError: No module named sklearn"

```bash
# Solution: Check Python version and reinstall
python --version  # Should be 3.8+
pip uninstall scikit-learn
pip install scikit-learn==1.3.0
```

---

## File Structure Reference

```
smart-agriculture-dss/
├── data/
│   └── agricultural_data.csv        ← Your dataset goes here
├── src/
│   ├── preprocessing.py             ← Data pipeline (don't modify)
│   ├── models.py                    ← ML models (don't modify)
│   ├── gui.py                       ← Tkinter interface (run this)
│   └── utils.py                     ← Utilities (don't modify)
├── models/
│   ├── decision_tree_model.pkl      ← Auto-generated after training
│   ├── kmeans_model.pkl
│   └── linear_regression_model.pkl
├── results/
│   └── model_metrics.json           ← Auto-generated metrics
├── train_models.py                  ← Run this to train
├── prepare_dataset.py               ← Run this first (once)
├── requirements.txt                 ← Dependencies
└── README.md                        ← Full documentation
```

---

## Next Steps

### Option 1: Modify Input Parameters (GUI)

Edit `src/gui.py` line 73-74 to add/remove input fields:

```python
input_params = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 
               'Humidity', 'pH', 'Rainfall', 'Soil_Type']  # Add/remove here
```

### Option 2: Use Your Own Dataset

1. Prepare CSV with columns: N, P, K, temperature, humidity, ph, rainfall, label, yield
2. Place in `data/agricultural_data.csv`
3. Run `python train_models.py`
4. Launch GUI with your trained models

### Option 3: Programmatic API Usage

```python
from src.models import ModelFactory
from src.preprocessing import DataPreprocessor
import numpy as np

# Load models
factory = ModelFactory()
factory.load_all_models()

# Make prediction
X_input = np.array([[80, 50, 100, 25, 70, 6.5, 100]])

crop = factory.dt_classifier.predict(X_input)[0]
cluster = factory.kmeans_clusterer.predict(X_input)[0]
yield_pred = factory.lr_regressor.predict(X_input)[0]

print(f"Crop: {crop}, Zone: {cluster}, Yield: {yield_pred:.2f}")
```

---

## Performance Expectations

**Model Training Time:**
- Dataset preparation: ~10 sec
- Decision Tree training: ~1 sec
- KMeans training: ~2 sec
- Linear Regression: ~1 sec
- **Total: ~20 seconds**

**GUI Prediction Latency:**
- Per prediction: ~50-100 ms
- Visualization generation: ~500-1000 ms

**Memory Usage:**
- Training: ~200 MB (peak)
- Runtime: ~50 MB (GUI running)
- Models on disk: ~300 KB total

---

## Documentation Links

- **Full README:** [README.md](README.md)
- **Technical Report:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- **Data Dictionary:** [data/data_dictionary.md](data/data_dictionary.md)
- **GitHub Repository:** https://github.com/yourusername/smart-agriculture-dss

---

## Support & Questions

- Check [README.md](README.md) for detailed documentation
- Review [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for methodology
- Refer to [Troubleshooting Guide](#troubleshooting-quick-reference)

---

## Key Commands Cheat Sheet

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Data preparation
python prepare_dataset.py

# Model training
python train_models.py

# Launch GUI
python src/gui.py

# Clean cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Check directory structure
tree -L 2  # On Linux/macOS
# or manually verify with ls -R
```

---

## Common Customizations

### Change Number of Clusters

Edit `src/models.py` line 87:

```python
self.model = KMeans(n_clusters=5)  # Change 3 to desired number
```

### Modify Input Feature List

Edit `src/gui.py` line 73:

```python
input_params = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 
               'Humidity', 'pH', 'Rainfall']  # Add/remove features
```

### Adjust Tree Depth

Edit `src/models.py` line 18:

```python
self.model = DecisionTreeClassifier(max_depth=15)  # Increase for deeper tree
```

---

**Happy farming! 🌾**

*For issues or questions, refer to README.md or open a GitHub issue.*

---

**Last Updated:** May 20, 2024  
**Status:** Ready for Deployment  
**Tested On:** Python 3.8, 3.9, 3.10, 3.11
