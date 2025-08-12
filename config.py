"""
Configuration settings for the Titanic ML project.
All paths, parameters, and constants are defined here.
"""

# ========================================================================
#  Imports
# ========================================================================

from pathlib import Path


# ========================================================================
#   Project Paths
# ========================================================================

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw/titanic.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed/titanic_processed.csv"

# Model directory for saved models
SAVED_MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ========================================================================
#   Model Settings
# ========================================================================

# Model versioning
MODEL_VERSION = "v1.0"
MODEL_FILENAME = f"xgboost_model_{MODEL_VERSION}.pkl"

# Target and feature columns
TARGET_COLUMN = "Survived"

# Features (i.e., columns) we'll use for training
FEATURE_COLUMNS = [
    "Pclass",
    "Sex_Encoded",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_Encoded",
    "FamilySize",
    "IsAlone",
    "Title_Encoded",
    "HasCabin",
    "FareBin_Encoded",      
    "AgeGroup_Encoded" 
]

# ========================================================================
#   Training Settings
# ========================================================================

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation (5 is sweet spot for most datasets; if millions of records try 3)
CV_FOLDS = 5

# XGBoost default parameters
XGBOOST_DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# Hyperparameter tuning grid (simplified for faster training)
HYPERPARAM_GRID = {
    "n_estimators": [100, 200],             # How many trees in the forst
    "max_depth": [3, 4, 5],                 # How tall each tree can grow (3-5 is sweet spot for most data to prevent overfitting)
    "learning_rate": [0.01, 0.1],           # 0.01 for careful learning, 0.1 for quick learning
    "subsample": [0.8, 1.0],                # What % of data each tree sees (<1 helps prevent overfitting); 0.8 = each tree sees 80% of data
    "colsample_bytree": [0.8, 1.0]          # What % of features each tree uses (<1.0 means each tree uses random features, like Random Forest)
}

# ========================================================================
#   Visualization Settings
# ========================================================================

# Set brand colors for visualizations
BRAND_COLORS = {
    "orange": "#FFA500",
    "green": "#8BB42D",
    "blue": "#0273BE",
    "pink": "#E90555"
}

# Color sequence for Plotly
COLOR_SEQUENCE = [
    BRAND_COLORS["blue"],
    BRAND_COLORS["orange"],
    BRAND_COLORS["green"],
    BRAND_COLORS["pink"]
]

# ========================================================================
#   Flask Settings
# ========================================================================

FLASK_DEBUG = True
FLASK_PORT = 5000
FLASK_HOST = "127.0.0.1"