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
import models


# ========================================================================
#   Main Training Pipeline
# ========================================================================

def train_pipeline(tune_hyperparameters=True):
    """
    Complete training pipeline from raw data to saved model.
    
    Args:
        tune_hyperparameters (bool): Whether to tune hyperparameters
        
    Returns:
        dict: Dictionary containing model, metrics, and encoders
    """
    
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
    X_train, X_test, y_train, y_test = models.split_data(X, y)
    
    # Train model
    print("\n5. TRAINING MODEL")
    print("-" * 40)
    
    if tune_hyperparameters:
        # Train with hyperparameter tuning
        model = models.tune_hyperparameters(X_train, y_train)
    else:
        # Train basic model
        model = models.train_basic_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    print("\n6. EVALUATING MODEL")
    print("-" * 40)
    metrics = models.evaluate_model(model, X_test, y_test)
    
    # Cross-validation
    print("\n7. CROSS-VALIDATION")
    print("-" * 40)
    cv_scores = models.cross_validate_model(model, X, y)
    print(f"✓ CV AUC: {cv_scores['mean']:.4f} (+/- {cv_scores['std']:.4f})")
    
    # Save model
    print("\n8. SAVING MODEL")
    print("-" * 40)
    model_path = models.save_model(model)
    
    # Save encoders
    import joblib
    encoder_path = config.SAVED_MODELS_DIR / "encoders.pkl"
    joblib.dump(encoders, encoder_path)
    print(f"✓ Encoders saved to: {encoder_path}")
    
    # Create visualizations
    print("\n9. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Feature importance
    fig_importance = models.plot_feature_importance(
        model, 
        X.columns.tolist()
    )
    fig_importance.show()
    
    # Confusion matrix
    fig_cm = models.plot_confusion_matrix(metrics["confusion_matrix"])
    fig_cm.show()
    
    # ROC curve
    fig_roc = models.plot_roc_curve(y_test, metrics["probabilities"])
    fig_roc.show()
    
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
    model = models.load_model()
    
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
    
    result1 = models.predict_single(model, passenger1, encoders)
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
    
    result2 = models.predict_single(model, passenger2, encoders)
    print(f"\nPassenger 2 (Third class man):")
    print(f"  Survived: {result2['survived']}")
    print(f"  Probability: {result2['survival_probability']:.2%}")


# ========================================================================
#   Run Training
# ========================================================================

if __name__ == "__main__":
    # Train the model
    results = train_pipeline(tune_hyperparameters=True)
    
    # Test predictions
    test_predictions()