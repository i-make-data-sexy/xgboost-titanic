"""
Main Flask application for the Titanic MLindex_dashboard.
Simplified to focus on routing and coordination.
"""

# ========================================================================
#   Imports
# ========================================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import logging
import config
import index_dashboard
import models_dashboard
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
        charts = index_dashboard.create_all_dashboard_charts(df)
        
        # Convert charts to JSON for frontend
        chart_json = {
            name: index_dashboard.convert_chart_to_json(chart)
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
    

# Add this route to your app.py file

@app.route("/model-performance")
def model_performance():
    """
    Display model performance metrics and evaluation charts.
    """
    
    try:
        # Check if model exists
        model_path = config.SAVED_MODELS_DIR / config.MODEL_FILENAME
        encoder_path = config.SAVED_MODELS_DIR / "encoders.pkl"
        
        if not model_path.exists():
            return render_template("error.html", 
                                 message="No trained model found. Please train the model first."), 404
        
        # Load model and encoders
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        
        # Load data for evaluation
        df = load_titanic_data()
        
        # Preprocess data (using saved encoders)
        df_processed, _ = preprocessing.preprocess_data(
            df, 
            fit_encoders=False,
            encoders=encoders
        )
        
        # Get features and target
        X = preprocessing.get_feature_matrix(df_processed)
        y = df_processed[config.TARGET_COLUMN]
        
        # Split data (using same seed for consistency)
        X_train, X_test, y_train, y_test = models_dashboard.split_data(X, y)
        
        # Get predictions for metrics
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred) * 100
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate additional metrics
        precision = precision_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = models_dashboard.cross_validate_model(model, X, y)
        
        # Create visualization charts
        importance_fig = models_dashboard.plot_feature_importance(model, X.columns.tolist())
        confusion_fig = models_dashboard.plot_confusion_matrix(cm)
        roc_fig = models_dashboard.plot_roc_curve(y_test, y_pred_proba)
        
        # Convert charts to JSON
        importance_json = index_dashboard.convert_chart_to_json(importance_fig)
        confusion_json = index_dashboard.convert_chart_to_json(confusion_fig)
        roc_json = index_dashboard.convert_chart_to_json(roc_fig)
        
        # Render template with all data
        return render_template(
            "model-performance.html",
            importance_chart=importance_json,
            confusion_chart=confusion_json,
            roc_chart=roc_json,
            accuracy=f"{accuracy:.1f}",
            roc_auc=f"{roc_auc:.3f}",
            cv_score=f"{cv_scores['mean']:.3f} ± {cv_scores['std']:.3f}",
            # Pass additional metrics
            precision=f"{precision:.1f}",
            recall=f"{recall:.1f}",
            f1_score=f"{f1:.3f}"
        )
        
    except Exception as e:
        logger.error(f"Error creating model performance dashboard: {str(e)}")
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
        result = models_dashboard.predict_single(model, passenger_data, encoders)
        
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
        # Add detailed logging
        logger.info("=" * 60)
        logger.info("RETRAIN ENDPOINT CALLED")
        logger.info("=" * 60)
        
        # Import training script
        from train import train_pipeline
        
        # Get training parameters from request
        tune_hyperparameters = request.json.get("tune_hyperparameters", True)
        logger.info(f"Hyperparameter tuning: {tune_hyperparameters}")
        
        # Run training pipeline
        logger.info("Starting model retraining via web interface...")  
        results = train_pipeline(tune_hyperparameters=tune_hyperparameters, show_plots=False)
        logger.info("Training pipeline completed")  
        
        # Log what we're returning
        logger.info("Returning JSON response - no tabs should have opened")
        
        # Return training results
        return jsonify({
            "status": "success",
            "metrics": {
                # Format accuracy to 1 decimal place
                "accuracy": round(float(results["metrics"]["accuracy"]) * 100, 1),
                "roc_auc": float(results["metrics"]["roc_auc"]),
                "cv_mean": float(results["cv_scores"]["mean"]),
                "cv_std": float(results["cv_scores"]["std"])
            }
        })
        
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        logger.error(f"Full traceback:", exc_info=True)  # Full error details
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