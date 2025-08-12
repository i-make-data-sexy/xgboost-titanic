# Titanic Survival Analysis Dashboard & XGBoost Predictor

A Flask-based interactive dashboard visualizing survival patterns from the Titanic dataset using Plotly Express, with XGBoost machine learning for predictive analysis.

## Features

### Visualizations
- **Four Interactive Dashboard Charts:**
  - Bar chart showing survivor counts by passenger class
  - Donut chart displaying survival rates by gender
  - Bar chart presenting survival rates by age group
  - Line chart illustrating survival rates by family size

### Machine Learning
- **XGBoost Predictive Model:**
  - Automated feature engineering (FamilySize, Title extraction, Age groups)
  - Hyperparameter tuning with GridSearchCV
  - Cross-validation for robust evaluation
  - Feature importance analysis
  - ROC curves and confusion matrices
  - Model persistence for production use

### Design
- **Modern Interface:**
  - Clean, responsive layout with CSS Grid
  - Custom brand colors (#FFA500, #8BB42D, #0273BE, #E90555)
  - Interactive Plotly charts with hover effects
  - Professional styling with shadows and rounded corners

## Getting Started

### Prerequisites

- Python 3.11+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [your-repo-url]
   cd XGBoost
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   #### Mac-Specific Requirements

Note: Mac users need to install OpenMP before using XGBoost:

```bash
# Install OpenMP runtime
brew install libomp

# If you don't have Homebrew, install it first:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


5. **Ensure your data file is in place:**
   - Place `titanic.csv` in the `data/raw/` directory

### Running the Application

#### Step 1: Train the Model (Required First Time)

```bash
# Train the XGBoost model
python train.py
```

**What happens:**
- Loads and preprocesses the Titanic dataset
- Trains XGBoost with hyperparameter tuning (~30 seconds for basic, ~2-3 minutes for full tuning)
- Displays three visualization windows (feature importance, confusion matrix, ROC curve)
- Saves model to `data/models/xgboost_model_v1.0.pkl`
- Saves encoders to `data/models/encoders.pkl`
- Creates `data/processed/titanic_processed.csv`

**Expected output:**
```
============================================================
                    TITANIC ML PIPELINE
============================================================

1. LOADING DATA
----------------------------------------
✓ Loaded 891 passengers

2. PREPROCESSING
----------------------------------------
Starting data preprocessing...
Creating family features...
✓ Model saved to: data/models/xgboost_model_v1.0.pkl
✓ Test Accuracy: 0.8324
✓ Test AUC Score: 0.8651
```

#### Step 2: Run the Dashboard

```bash
# Start the Flask server
python app.py
```

**Expected output:**
```
INFO | Starting Flask app on 127.0.0.1:5000
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

#### Step 3: View the Dashboard

Open your browser and navigate to: `http://localhost:5000`

You'll see four interactive charts:
- Survivor counts by passenger class
- Survival rates by gender (donut chart)
- Survival rates by age group
- Survival rates by family size

### Quick Test Mode

```bash
# If you just want to see everything work quickly:

# Terminal 1
python train.py  # Takes ~30 seconds

# Terminal 2 (after training completes)
python app.py

# Browser
# Go to http://localhost:5000
```

### Testing Predictions

After training, you can test individual predictions:

```python
# Start Python interactive mode
python

>>> import models, joblib, config
>>> 
>>> # Load the trained model
>>> model = models.load_model()
>>> encoders = joblib.load(config.SAVED_MODELS_DIR / "encoders.pkl")
>>> 
>>> # Test a passenger
>>> passenger = {
...     "Pclass": 1,
...     "Sex": "female",
...     "Age": 30,
...     "SibSp": 1,
...     "Parch": 0,
...     "Fare": 100,
...     "Embarked": "C",
...     "Name": "Mrs. Test Passenger",
...     "Cabin": "C123"
... }
>>> 
>>> result = models.predict_single(model, passenger, encoders)
>>> print(f"Survived: {result['survived']}")
>>> print(f"Survival probability: {result['survival_probability']:.1%}")
```

### Troubleshooting

#### Module Not Found Errors
```bash
# If you get "ModuleNotFoundError: No module named 'xgboost'"
pip install xgboost pandas numpy scikit-learn plotly flask joblib
```

#### File Not Found Errors
```bash
# If you get "FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/titanic.csv'"
# Make sure your directory structure is correct:
ls data/raw/  # Should show titanic.csv

# If directories don't exist:
mkdir -p data/raw data/models data/processed
```

#### Port Already in Use
```bash
# If you get "Address already in use" error
# Option 1: Kill the process using port 5000
lsof -i :5000  # Mac/Linux - shows what's using the port
kill -9 <PID>  # Replace <PID> with the process ID

# Option 2: Use a different port
# Edit config.py and change:
FLASK_PORT = 5001  # Or any other available port
```

#### Charts Not Displaying During Training
If the Plotly charts don't open automatically during training:
1. Check your default browser settings
2. Look for the charts in any open browser windows
3. The training will still complete even if you close the charts

#### Model Not Found When Running Dashboard
```bash
# If app.py runs but you see "Model not found" errors:
# You need to train the model first:
python train.py

# This creates the required files:
# - data/models/xgboost_model_v1.0.pkl
# - data/models/encoders.pkl
```

## Project Structure

```
XGBoost/
├── data/
│   ├── models/              # Saved trained models (.pkl files)
│   ├── processed/           # Processed/cleaned datasets
│   └── raw/                 # Original untouched data
│       └── titanic.csv
├── static/
│   ├── css/
│   │   └── styles.css       # Dashboard styling
│   ├── img/
│   │   └── favicon.png      # Site favicon (defaults to Annielytics)
│   └── js/
│       └── dashboard.js     # Chart rendering logic
├── templates/
│   └── index.html           # Dashboard HTML template
├── app.py                   # Flask application (routes & endpoints)
├── app_structure.py         # Utility to display project structure
├── config.py                # Central configuration & settings
├── dashboard.py             # Dashboard chart creation functions
├── models.py                # ML model training & prediction
├── preprocessing.py         # Data cleaning & feature engineering
├── train.py                 # Main training orchestration script
└── README.md                # This file
```

## Technical Implementation

### Machine Learning Pipeline

1. **Data Preprocessing (`preprocessing.py`):**
   - Feature engineering: FamilySize, IsAlone, Title extraction
   - Missing value handling with transparent strategies
   - Label encoding for categorical variables
   - Age and fare binning for better patterns

2. **Model Training (`models.py`):**
   - Basic XGBoost baseline model
   - Hyperparameter tuning with GridSearchCV
   - 5-fold cross-validation
   - Model persistence with joblib
   - Comprehensive evaluation metrics

3. **Configuration (`config.py`):**
   - Centralized settings for paths and parameters
   - Model versioning
   - Feature column definitions
   - Brand color definitions

### Dashboard Components

1. **Flask Backend (`app.py` & `dashboard.py`):**
   - Loads and processes Titanic dataset
   - Creates four Plotly Express visualizations
   - Handles JSON serialization for frontend consumption
   - Provides endpoints for predictions and model info

2. **Frontend (`index.html` & `dashboard.js`):**
   - Responsive grid layout for charts
   - JavaScript to render Plotly charts from JSON data
   - Binary data decoding for proper chart display

3. **Styling (`styles.css`):**
   - Modern, clean design with custom containers
   - Responsive CSS Grid layout
   - Overflow handling to prevent chart bleeding

## API Endpoints

- `GET /` - Main dashboard page
- `GET /predict` - Prediction form (if implemented)
- `POST /predict` - Make predictions for new passengers
- `GET /model-info` - Get information about trained model
- `POST /retrain` - Trigger model retraining

## 🐛 Plotly Binary Encoding Issue & Solution

### The Problem

When using Plotly 5.x with Flask, chart data was being binary-encoded for optimization, resulting in data like:
```json
{
  "y": {
    "dtype": "i2",
    "bdata": "iABXAHcA"
  }
}
```

Instead of the expected clean arrays:
```json
{
  "y": [136, 87, 119]
}
```

This caused charts to display incorrect values (e.g., showing 0-2 survivors when segmented by class instead of 136, 87, 119).

### Solutions Attempted

1. **Disable binary encoding in Python (didn't work):**
   ```python
   import plotly.io as pio
   pio.json.config.default_engine = "json"
   ```

2. **Manual conversion to clean JSON (didn't work):**
   ```python
   def fig_to_clean_json(fig):
       fig_dict = fig.to_dict()
       # Convert numpy arrays to lists
       # ...
   ```

### The Working Solution

Decode the binary data on the frontend in JavaScript (`dashboard.js`):

```javascript
function decodeBinaryData(obj) {
    // Base64 decode helper
    function base64ToArray(base64, dtype) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        
        // Convert based on dtype
        if (dtype === 'i2') return new Int16Array(bytes.buffer);
        if (dtype === 'f8') return new Float64Array(bytes.buffer);
        // ... other types
    }
    
    // Recursively process and decode
    if (obj && typeof obj === 'object') {
        if (obj.bdata && obj.dtype) {
            const decoded = base64ToArray(obj.bdata, obj.dtype);
            return Array.from(decoded);
        }
        // ... recursive processing
    }
}
```

This solution intercepts the binary-encoded data before it reaches Plotly and converts it back to regular JavaScript arrays.

## Model Performance

The XGBoost model typically achieves:
- **Accuracy:** ~82-84%
- **ROC AUC:** ~0.86-0.88
- **Cross-validation score:** ~0.85 (±0.03)

Key predictive features (by importance):
1. Passenger class
2. Sex
3. Fare
4. Age
5. Family size

## Customization

### Brand Colors

The dashboard uses a consistent color palette defined in `config.py`:
- **Orange:** `#FFA500` - Primary accent color
- **Green:** `#8BB42D` - Secondary color
- **Blue:** `#0273BE` - Main visualization color
- **Pink:** `#E90555` - Accent color

To change colors, update the `BRAND_COLORS` dictionary in `config.py`.

### Model Parameters

Adjust hyperparameter tuning in `config.py`:
```python
HYPERPARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.1],
    ...
}
```

### Feature Engineering

Add new features in `preprocessing.py`:
- Create new engineered features in dedicated functions
- Update `FEATURE_COLUMNS` in `config.py`
- Retrain the model with `python train.py`

## Data Insights

The analysis reveals several key patterns from the Titanic disaster:
- **Class matters:** First-class passengers had significantly higher survival rates (63% vs 24% for third class)
- **Women and children first:** 74.2% of females survived vs 18.9% of males
- **Age impact:** Children had the highest survival rate at 58%
- **Family size:** Mid-sized families (3-4 people) had better survival rates
- **Title importance:** "Mrs" and "Miss" had much higher survival rates than "Mr"

## Requirements

```txt
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
plotly==5.15.0
Flask==2.3.2
joblib==1.3.1
```

## Future Enhancements

- [ ] Add real-time prediction interface
- [ ] Implement model versioning and A/B testing
- [ ] Add more ensemble methods (Random Forest, LightGBM)
- [ ] Create automated retraining pipeline
- [ ] Add data drift detection
- [ ] Implement feature selection algorithms
- [ ] Add explainability with SHAP values

## Acknowledgments

- Dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- Visualization library: [Plotly Express](https://plotly.com/python/plotly-express/)
- Web framework: [Flask](https://flask.palletsprojects.com/)
- ML framework: [XGBoost](https://xgboost.readthedocs.io/)

---

**Note:** This project was created as a demonstration of combining data visualization techniques with machine learning for predictive analysis. The binary encoding issue and its JavaScript-based solution represent a real-world challenge when integrating Plotly with web frameworks. The modular structure makes it easy to adapt for production use cases like utility company analytics.