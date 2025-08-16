"""
Main training script for the Titanic XGBoost model.
Run this to train and save your model.
"""

# ========================================================================
#   Imports
# ========================================================================

import pandas as pd
import config
import preprocessing
import models_dashboard
import logging

# Disable Plotly auto-show behavior
import plotly.io as pio
pio.renderers.default = None  # Prevents any automatic rendering


# ========================================================================
#   Instantiate Logger
# ========================================================================

logger = logging.getLogger(__name__)


# ========================================================================
#   Main Training Pipeline
# ========================================================================

def train_pipeline(tune_hyperparameters=True, show_plots=True):
    """
    Complete training pipeline from raw data to saved model.
    
    Args:
        tune_hyperparameters (bool): Whether to tune hyperparameters
        show_plots (bool): Whether to display plots in new windows
        
    Returns:
        dict: Dictionary containing model, metrics, and encoders
    """
    
    # Add logging
    logger.info("Starting train_pipeline function")
    
    print("=" * 60)
    print(" " * 20 + "TITANIC ML PIPELINE")
    print("=" * 60)
    
    # Load raw data
    print("\n1. LOADING DATA")
    print("-" * 40)
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"✓ Loaded {len(df)} passengers")
    
    # Preprocess data
    print("\n2. PREPROCESSING")
    print("-" * 40)
    df_processed, encoders = preprocessing.preprocess_data(df)
    
    # Save processed data for reference
    df_processed.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"✓ Saved processed data to {config.PROCESSED_DATA_PATH}")
    
    # Prepare features and target
    print("\n3. PREPARING FEATURES")
    print("-" * 40)
    X = preprocessing.get_feature_matrix(df_processed)
    y = df_processed[config.TARGET_COLUMN]
    print(f"✓ Feature matrix shape: {X.shape}")
    
    # Split data
    print("\n4. SPLITTING DATA")
    print("-" * 40)
    X_train, X_test, y_train, y_test = models_dashboard.split_data(X, y)
    
    # Train model
    print("\n5. TRAINING MODEL")
    print("-" * 40)
    
    if tune_hyperparameters:
        # Train with hyperparameter tuning
        model = models_dashboard.tune_hyperparameters(X_train, y_train)
    else:
        # Train basic model
        model = models_dashboard.train_basic_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    print("\n6. EVALUATING MODEL")
    print("-" * 40)
    metrics = models_dashboard.evaluate_model(model, X_test, y_test)
    
    # Cross-validation
    print("\n7. CROSS-VALIDATION")
    print("-" * 40)
    cv_scores = models_dashboard.cross_validate_model(model, X, y)
    print(f"✓ CV AUC: {cv_scores['mean']:.4f} (+/- {cv_scores['std']:.4f})")
    
    # Save model
    print("\n8. SAVING MODEL")
    print("-" * 40)
    model_path = models_dashboard.save_model(model)
    
    # Save encoders
    import joblib
    encoder_path = config.SAVED_MODELS_DIR / "encoders.pkl"
    joblib.dump(encoders, encoder_path)
    print(f"✓ Encoders saved to: {encoder_path}")
    
    # Create visualizations
    print("\n9. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Add detailed logging
    logger.info("Creating visualizations - NO show() should be called")
    
    # Feature importance
    logger.info("Creating feature importance plot...")  
    fig_importance = models_dashboard.plot_feature_importance(
        model, 
        X.columns.tolist()
    )
    logger.info("Feature importance plot created successfully")  
    
    # Confusion matrix
    logger.info("Creating confusion matrix...")  
    fig_cm = models_dashboard.plot_confusion_matrix(metrics["confusion_matrix"])
    logger.info("Confusion matrix created successfully") 
    
    # ROC curve with optimal threshold point (elbow)
    logger.info("Creating ROC curve...")  
    fig_roc = models_dashboard.plot_roc_curve(y_test, metrics["probabilities"])
    logger.info("ROC curve created successfully")  
    # fig_roc.show()  # Should be commented out
    
    # Log completion
    logger.info("All visualizations created without showing")
    print("✓ Visualizations created (view at /model-performance)")
    
    # Only show plots if requested (not when called from Flask)
    if show_plots:
        logger.info("Showing plots in new windows (show_plots=True)")
        fig_importance.show()
        fig_cm.show()
        fig_roc.show()
    else:
        logger.info("NOT showing plots (show_plots=False)")
    
    print("✓ Visualizations created (view at /model-performance)")
    
    print("\n" + "=" * 60)
    print(" " * 20 + "TRAINING COMPLETE!")
    print("=" * 60)
    
    return {
        "model": model,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "encoders": encoders,
        "feature_names": X.columns.tolist()
    }


# ========================================================================
#   Quick Test Predictions
# ========================================================================

def test_predictions():
    """
    Tests the saved model with sample predictions.
    """
    
    print("\nTESTING PREDICTIONS")
    print("-" * 40)
    
    # Load model and encoders
    model = models_dashboard.load_model()
    
    import joblib
    encoders = joblib.load(config.SAVED_MODELS_DIR / "encoders.pkl")
    
    # Test passenger 1: Rich woman in first class
    passenger1 = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 35,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "C",
        "Name": "Mrs. Test Passenger",
        "Cabin": "C123"
    }
    
    result1 = models_dashboard.predict_single(model, passenger1, encoders)
    print(f"\nPassenger 1 (First class woman):")
    print(f"  Survived: {result1['survived']}")
    print(f"  Probability: {result1['survival_probability']:.2%}")
    
    # Test passenger 2: Poor man in third class
    passenger2 = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7,
        "Embarked": "S",
        "Name": "Mr. Test Passenger",
        "Cabin": None
    }
    
    result2 = models_dashboard.predict_single(model, passenger2, encoders)
    print(f"\nPassenger 2 (Third class man):")
    print(f"  Survived: {result2['survived']}")
    print(f"  Probability: {result2['survival_probability']:.2%}")


# ========================================================================
#   Run Training
# ========================================================================

if __name__ == "__main__":
    # Train the model
    results = train_pipeline(tune_hyperparameters=True, show_plots=True)
    
    # Test predictions
    test_predictions()