"""
Main Flask application for the Titanic ML Dashboard.
Simplified to focus on routing and coordination.
"""

# ========================================================================
#   Imports
# ========================================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import logging
import config
import dashboard
import models
import preprocessing
import joblib
from pathlib import Path

# ========================================================================
#   Flask Configuration
# ========================================================================

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ========================================================================
#   Data Loading
# ========================================================================

def load_titanic_data():
    """
    Loads the Titanic dataset from the configured path.
    
    Returns:
        pd.DataFrame: Titanic dataset
    """
    
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
        logger.info(f"Loaded {len(df)} records from Titanic dataset")
        return df
    except FileNotFoundError:
        logger.error(f"Could not find data file at {config.RAW_DATA_PATH}")
        raise


# ========================================================================
#   Routes
# ========================================================================

@app.route("/")
def index():
    """
    Main dashboard route - displays visualizations.
    """
    
    try:
        # Load data
        df = load_titanic_data()
        
        # Create all dashboard charts
        charts = dashboard.create_all_dashboard_charts(df)
        
        # Convert charts to JSON for frontend
        chart_json = {
            name: dashboard.convert_chart_to_json(chart)
            for name, chart in charts.items()
        }
        
        # Render template with chart data
        return render_template(
            "index.html",
            class_chart=chart_json["class_chart"],
            gender_chart=chart_json["gender_chart"],
            age_chart=chart_json["age_chart"],
            family_chart=chart_json["family_chart"]
        )
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return f"Error: {str(e)}", 500


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Prediction endpoint - can display form or process predictions.
    """
    
    if request.method == "GET":
        # Display prediction form
        return render_template("predict.html")
    
    # Process prediction request
    try:
        # Load model and encoders
        model_path = config.SAVED_MODELS_DIR / config.MODEL_FILENAME
        encoder_path = config.SAVED_MODELS_DIR / "encoders.pkl"
        
        if not model_path.exists():
            return jsonify({"error": "Model not found. Please train the model first."}), 404
        
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        
        # Get passenger data from request
        passenger_data = request.json
        
        # Make prediction
        result = models.predict_single(model, passenger_data, encoders)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def model_info():
    """
    Displays information about the trained model.
    """
    
    try:
        # Check if model exists
        model_path = config.SAVED_MODELS_DIR / config.MODEL_FILENAME
        
        if not model_path.exists():
            return jsonify({"status": "No model trained yet"}), 404
        
        # Load model
        model = joblib.load(model_path)
        
        # Get model parameters
        model_info = {
            "model_type": "XGBoost Classifier",
            "version": config.MODEL_VERSION,
            "parameters": model.get_params(),
            "n_features": model.n_features_in_ if hasattr(model, "n_features_in_") else "Unknown",
            "feature_names": config.FEATURE_COLUMNS
        }
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Endpoint to trigger model retraining.
    """
    
    try:
        # Import training script
        from train import train_pipeline
        
        # Get training parameters from request
        tune_hyperparameters = request.json.get("tune_hyperparameters", True)
        
        # Run training pipeline
        logger.info("Starting model retraining...")
        results = train_pipeline(tune_hyperparameters=tune_hyperparameters)
        
        # Return training results
        return jsonify({
            "status": "success",
            "metrics": {
                "accuracy": float(results["metrics"]["accuracy"]),
                "roc_auc": float(results["metrics"]["roc_auc"]),
                "cv_mean": float(results["cv_scores"]["mean"]),
                "cv_std": float(results["cv_scores"]["std"])
            }
        })
        
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ========================================================================
#   Error Handlers
# ========================================================================

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(e)}")
    return render_template("500.html"), 500


# ========================================================================
#   Main Entry Point
# ========================================================================

if __name__ == "__main__":
    logger.info(f"Starting Flask app on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(
        debug=config.FLASK_DEBUG,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT
    )