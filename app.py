"""
Main Flask application for the Titanic MLindex_dashboard.
Simplified to focus on routing and coordination.
"""

# ========================================================================
#   Imports
# ========================================================================

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import logging
import config
import index_dashboard
import models_dashboard
import preprocessing
import joblib
import json
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
            roc_auc=f"{roc_auc:.2f}",
            cv_score=f"{cv_scores['mean']:.2f} ± {cv_scores['std']:.2f}",
            # Pass additional metrics
            precision=f"{precision:.1f}",
            recall=f"{recall:.1f}",
            f1_score=f"{f1:.2f}"
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


@app.route('/model-info')
def model_info():
    # Load model
    model_path = config.SAVED_MODELS_DIR / config.MODEL_FILENAME
    
    # Get the actual model information
    if not os.path.exists(model_path):
        return "No model found", 404
    
    model = joblib.load(model_path)
    
    # Prepare the data
    feature_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else []
    parameters = model.get_params()
    
    # Convert to formatted JSON strings
    feature_names_json = json.dumps(feature_names, indent=2)
    parameters_json = json.dumps(parameters, indent=2)
    
    # NEW: Extract the actual values for the metric cards
    n_estimators = parameters.get('n_estimators', 100)
    max_depth = parameters.get('max_depth', 3)
    learning_rate = parameters.get('learning_rate', 0.1)
    
    # Render the template with the data
    return render_template('model-info.html',
        model_type="XGBoost",
        n_features=model.n_features_in_,
        version="v1.0",
        n_estimators=n_estimators,  # NEW: Added this
        max_depth=max_depth,  # NEW: Added this
        learning_rate=learning_rate,  # NEW: Added this
        feature_names_json=feature_names_json,
        parameters_json=parameters_json
    )
    
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

@app.route("/download-data")
def download_data():
    """
    Generate a ZIP file with all CSV data for dashboard prototyping.
    """
    from flask import Response
    import io
    import zipfile
    
    try:
        # Load and preprocess data
        df = load_titanic_data()
        
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # 1. ORIGINAL DATA FOR EXPLORATION DASHBOARD
            # This is the clean data for your class, gender, age, and family charts
            exploration_df = df.copy()
            exploration_df['Survived_Label'] = exploration_df['Survived'].map({0: 'Did Not Survive', 1: 'Survived'})
            exploration_df['Sex_Label'] = exploration_df['Sex']
            exploration_df['Pclass_Label'] = exploration_df['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})
            
            # Add age groups for age distribution chart
            exploration_df['Age_Group'] = pd.cut(exploration_df['Age'], 
                                                  bins=[0, 12, 18, 35, 50, 80], 
                                                  labels=['Child', 'Teen', 'Young Adult', 'Middle Age', 'Senior'])
            
            # Add family size categories
            exploration_df['SibSp_Parch_Total'] = exploration_df['SibSp'] + exploration_df['Parch']
            exploration_df['Family_Size'] = exploration_df['SibSp_Parch_Total'].apply(
                lambda x: 'Alone' if x == 0 else 'Small (1-3)' if x <= 3 else 'Large (4+)'
            )
            
            csv_data = exploration_df.to_csv(index=False)
            zip_file.writestr('01_exploration_dashboard_data.csv', csv_data)
            
            # 2. PREPROCESSED DATA WITH ALL FEATURES
            df_processed, encoders = preprocessing.preprocess_data(df)
            df_processed['Survived_Label'] = df_processed['Survived'].map({0: 'Did Not Survive', 1: 'Survived'})
            
            csv_data = df_processed.to_csv(index=False)
            zip_file.writestr('02_preprocessed_features.csv', csv_data)
            
            # 3. COMBINED MODEL PERFORMANCE DATA (with predictions AND feature importance)
            model_path = config.SAVED_MODELS_DIR / config.MODEL_FILENAME
            encoder_path = config.SAVED_MODELS_DIR / "encoders.pkl"
            
            if model_path.exists():
                model = joblib.load(model_path)
                encoders = joblib.load(encoder_path)
                
                # Get preprocessed data and predictions
                df_processed, _ = preprocessing.preprocess_data(df, fit_encoders=False, encoders=encoders)
                X = preprocessing.get_feature_matrix(df_processed)
                y = df_processed[config.TARGET_COLUMN]
                
                # Split data
                X_train, X_test, y_train, y_test = models_dashboard.split_data(X, y)
                
                # Get predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # FIX: Reset index to avoid mismatch
                X_test_reset = X_test.reset_index(drop=True)
                y_test_reset = y_test.reset_index(drop=True)
                
                # Create model performance dataframe with proper indexing
                model_perf_df = pd.DataFrame({
                    'Actual_Survived': y_test_reset.values,
                    'Predicted_Survived': y_pred,
                    'Survival_Probability': y_pred_proba,
                    'Actual_Label': y_test_reset.map({0: 'Did Not Survive', 1: 'Survived'}),
                    'Predicted_Label': pd.Series(y_pred).map({0: 'Did Not Survive', 1: 'Survived'}),
                    'Correct_Prediction': (y_test_reset.values == y_pred).astype(int)
                })
                
                # Add all features to the performance data
                for col in X_test_reset.columns:
                    model_perf_df[col] = X_test_reset[col].values
                
                csv_data = model_perf_df.to_csv(index=False)
                zip_file.writestr('03_model_predictions.csv', csv_data)
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': model.feature_names_in_,
                    'Importance_Score': model.feature_importances_,
                    'Importance_Percentage': model.feature_importances_ * 100
                }).sort_values('Importance_Score', ascending=False)
                
                csv_data = importance_df.to_csv(index=False)
                zip_file.writestr('04_feature_importance.csv', csv_data)
                
                # 4. CONFUSION MATRIX DATA
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                
                cm_df = pd.DataFrame({
                    'Actual': ['Did Not Survive', 'Did Not Survive', 'Survived', 'Survived'],
                    'Predicted': ['Did Not Survive', 'Survived', 'Did Not Survive', 'Survived'],
                    'Count': [cm[0,0], cm[0,1], cm[1,0], cm[1,1]],
                    'Category': ['True Negative', 'False Positive', 'False Negative', 'True Positive']
                })
                
                csv_data = cm_df.to_csv(index=False)
                zip_file.writestr('05_confusion_matrix.csv', csv_data)
            
            # 5. RAW ORIGINAL DATA
            csv_data = df.to_csv(index=False)
            zip_file.writestr('06_titanic_raw.csv', csv_data)
        
        # Prepare the ZIP file for download
        zip_buffer.seek(0)
        
        return Response(
            zip_buffer.getvalue(),
            mimetype="application/zip",
            headers={"Content-Disposition": "attachment; filename=titanic_dashboard_data.zip"}
        )
        
    except Exception as e:
        logger.error(f"Error generating ZIP file: {str(e)}")
        return f"Error: {str(e)}", 500

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