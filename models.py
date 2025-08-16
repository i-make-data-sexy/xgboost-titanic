"""
Model training, prediction, and evaluation functions.
Handles XGBoost model lifecycle.
"""

# ========================================================================
#   Imports
# ========================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import config
import logging


# ========================================================================
#   Instantiate Logger
# ========================================================================

logger = logging.getLogger(__name__)


# ========================================================================
#   Data Splitting
# ========================================================================

def split_data(X, y, test_size=None, random_state=None):
    """
    Splits data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        test_size (float): Proportion for testing
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    # Use config defaults if not provided
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y    # Keep same ratio of survivors in both sets
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Survival rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


# ========================================================================
#   Model Training
# ========================================================================

def train_basic_model(X_train, y_train, X_test=None, y_test=None, use_class_weight=False):
    """
    Trains a basic XGBoost model with default parameters.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Optional test features for evaluation
        y_test (pd.Series): Optional test target
        use_class_weight (bool): Whether to balance class weights
        
    Returns:
        xgb.XGBClassifier: Trained model
    """
    
    print("\nTraining basic XGBoost model...")
    
    # Calculate class weight if requested
    if use_class_weight:
        # Calculate ratio of negative to positive class
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        print(f"Using class weight: {scale_pos_weight:.2f} (addressing {pos_count}/{neg_count} imbalance)")
    else:
        scale_pos_weight = 1.0
    
    # Create model with default parameters
    model = xgb.XGBClassifier(
        **config.XGBOOST_DEFAULT_PARAMS,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight  # Class weight parameter
    )
    
    # Prepare evaluation set if test data provided
    eval_set = [(X_test, y_test)] if X_test is not None else None
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    print("✓ Basic model trained successfully")
    
    return model

def tune_hyperparameters(X_train, y_train, param_grid=None, cv_folds=None, use_class_weight=False):
    """
    Performs hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (dict): Parameters to search
        cv_folds (int): Number of cross-validation folds
        use_class_weight (bool): Whether to balance class weights
        
    Returns:
        xgb.XGBClassifier: Best tuned model
    """
    
    print("\nTuning hyperparameters (this may take a minute)...")
    
    # Use defaults if not provided
    if param_grid is None:
        param_grid = config.HYPERPARAM_GRID
    if cv_folds is None:
        cv_folds = config.CV_FOLDS
    
    # Calculate class weight if requested
    if use_class_weight:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        print(f"Using class weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0
    
    # Base model for grid search
    base_model = xgb.XGBClassifier(
        **config.XGBOOST_DEFAULT_PARAMS,
        scale_pos_weight=scale_pos_weight
    )
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
    print("✓ Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    
    return grid_search.best_estimator_

def cross_validate_model(model, X, y, cv_folds=None):
    """
    Performs cross-validation on a model.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        y (pd.Series): Target
        cv_folds (int): Number of folds
        
    Returns:
        dict: Cross-validation scores
    """
    
    if cv_folds is None:
        cv_folds = config.CV_FOLDS
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X, y,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    return {
        "mean": cv_scores.mean(),
        "std": cv_scores.std(),
        "scores": cv_scores
    }


# ========================================================================
#   Model Evaluation
# ========================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Evaluation metrics
    """
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "predictions": y_pred,
        "probabilities": y_pred_proba
    }
    
    # Print evaluation results
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Did Not Survive", "Survived"]
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
    print(f"FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")
    
    # Add confusion matrix to metrics
    metrics["confusion_matrix"] = cm
    
    return metrics


# ========================================================================
#   Model Visualization (using Plotly)
# ========================================================================

def plot_feature_importance(model, feature_names):
    """
    Creates a Plotly feature importance chart.
    
    Args:
        model: Trained XGBoost model
        feature_names (list): List of feature names
        
    Returns:
        plotly.graph_objects.Figure: Feature importance chart
    """
    # Start logging
    logger.info("plot_feature_importance called - should NOT open new tab")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance * 100  # Convert to percentage
    }).sort_values("Importance", ascending=True)
    
    # Use Plotly Express
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance Scores",
        text="Importance",
        color_discrete_sequence=[config.BRAND_COLORS["blue"]],
        labels={"Importance": "Importance (%)"}
    )
    
    # Format text on bars as percentages
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hoverinfo='skip',
        hovertemplate=None,                          # Disable tooltips (hovertemplate) because there's no additional info
    )
    
    max_x = importance_df["Importance"].max()
    
    fig.update_layout(
        xaxis=dict(
            title="",
            showticklabels=False,
            zeroline=True,              # Show the zero line
            zerolinewidth=1,            # Make it thin (adjust as needed: 0.5 for thinner)
            zerolinecolor="#DEDEDE",  # Light gray color
            showline=False,
            showgrid=False,
            range=[0, max_x * 1.15]
        ),
        yaxis_title="",
        margin=dict(l=120, r=50, t=50, b=70),  # More space for labels
        margin_pad=5,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # Log status
    logger.info("Returning feature importance figure without showing")
    
    return fig

def plot_confusion_matrix(cm, labels=["Did Not Survive", "Survived"]):
    """
    Creates a Plotly confusion matrix heatmap.
    
    Args:
        cm (np.array): Confusion matrix
        labels (list): Class labels
        
    Returns:
        plotly.graph_objects.Figure: Confusion matrix heatmap
    """
    
    # Start logging
    logger.info("plot_confusion_matrix called - should NOT open new tab")
    
    # Debug the confusion matrix values
    logger.debug(f"Confusion matrix shape: {cm.shape}")
    logger.debug(f"Confusion matrix values:\n{cm}")
    
    # Create hover text for each cell
    hover_text = np.array([
        [
            f"<b>True Negatives (TN)</b><br>Correctly predicted non-survivors<br><br>Predicted: {labels[0]}<br>Actual: {labels[0]}<br><br>Count: {cm[0,0]}",
            f"<b>False Positives (FP)</b><br>Missed non-survivors<br><br>Predicted: {labels[1]}<br>Actual: {labels[0]}<br><br>Count: {cm[0,1]}"
        ],
        [
            f"<b>False Negatives (FN)</b><br>Missed survivors<br><br>Predicted: {labels[0]}<br>Actual: {labels[1]}<br><br>Count: {cm[1,0]}",
            f"<b>True Positives (TP)</b><br>Correctly predicted non-survivors<br><br>Predicted: {labels[1]}<br>Actual: {labels[1]}<br><br>Count: {cm[1,1]}"
        ]
    ])
    
    # Create two separate arrays - one for colors, one for display
    # The z array determines colors, text array shows values
    z_for_colors = [[1, 0], [0, 1]]  
    
    # Custom colorscale - only two colors needed
    colorscale = [
        [0, "#D3D3D3"],                       # Gray for value 0 (incorrect)
        [1, config.BRAND_COLORS["blue"]]        # Blue for value 1 (correct)
    ]
    
    # Create text annotations for display
    text_display = [[str(cm[i,j]) for j in range(2)] for i in range(2)]
    
    # Use go.Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_for_colors,                           # Use binary matrix for colors
        text=text_display,                        # Use string version of values
    texttemplate="%{text}",                       # Show the values
        textfont={"size": 14, "color": "white"},  
        x=labels,
        y=labels,
        colorscale=colorscale,
        showscale=False,
        hovertext=hover_text,
        hoverinfo="text",
        zmin=0,                                   # Explicitly set min
        zmax=1                                    # Explicitly set max
    ))
    
    # Calculate accuracy metrics
    total = cm.sum()
    correct = cm[0,0] + cm[1,1]
    incorrect = cm[0,1] + cm[1,0]
    accuracy = (correct / total) * 100
    error_rate = (incorrect / total) * 100
    
    # Update layout
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(
            title="Predicted",
            side="bottom",
            tickmode="array",
            tickvals=[0, 1],
            ticktext=labels
        ),
        yaxis=dict(
            title="Actual",
            tickmode="array",
            tickvals=[0, 1],
            ticktext=labels,
            autorange="reversed"
        ),
        margin_pad=5,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=60, b=180)
    )
    
    # Add interpretation annotation
    fig.add_annotation(
        text=(
            "<b>How to Read:</b><br>"
            f"• Blue cells = Correct predictions ({correct}/{total} = {accuracy:.1f}%)<br>"
            f"• Gray cells = Incorrect predictions ({incorrect}/{total} = {error_rate:.1f}%)<br>"
            f"• Model Accuracy: {accuracy:.1f}%"
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.2,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left"
    )
    
    # Extra logging
    logger.info("Returning confusion matrix figure without showing")
    logger.debug(f"z_for_colors matrix: {z_for_colors}")
    logger.debug(f"Actual CM values: {cm}")
    
    return fig

def find_optimal_threshold(y_true, y_scores, method="youden"):
    """
    Finds the optimal classification threshold using various methods.
    
    Args:
        y_true (np.array): True labels
        y_scores (np.array): Predicted probabilities
        method (str): Method to find threshold
            - "youden": Maximizes Youden's J statistic (TPR - FPR)
            - "closest": Point closest to top-left corner
            - "f1": Maximizes F1 score
        
    Returns:
        dict: Optimal threshold and metrics
    """
    
    from sklearn.metrics import roc_curve, f1_score
    import numpy as np
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    if method == "youden":
        # Youden's J statistic: maximize (sensitivity + specificity - 1)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
    elif method == "closest":
        # Find point closest to (0,1) - the perfect classifier
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        
    elif method == "f1":
        # Find threshold that maximizes F1 score
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        optimal_idx = np.argmax(f1_scores)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get optimal values
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    # Calculate performance at this threshold
    y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
    
    return {
        "threshold": optimal_threshold,
        "fpr": optimal_fpr,
        "tpr": optimal_tpr,
        "sensitivity": optimal_tpr,  # True Positive Rate
        "specificity": 1 - optimal_fpr,  # True Negative Rate
        "accuracy": np.mean(y_true == y_pred_optimal),
        "method": method
    }

def plot_roc_curve(y_true, y_scores):
    """
    Creates a Plotly ROC curve with the optimal threshold point highlighted.
    
    Args:
        y_true (np.array): True labels
        y_scores (np.array): Predicted probabilities
        
    Returns:
        plotly.graph_objects.Figure: ROC curve with optimal point
    """
    
    from sklearn.metrics import roc_curve, auc
    
    # Start logging
    logger.info("plot_roc_curve called - should NOT open new tab")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (the "elbow")
    optimal = find_optimal_threshold(y_true, y_scores, method="youden")
    
    # Create DataFrame for Plotly Express
    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr,
        "Model": f"ROC Curve (AUC = {roc_auc:.2f})"
    })
    
    # Add diagonal reference line
    diagonal_df = pd.DataFrame({
        "False Positive Rate": [0, 1],
        "True Positive Rate": [0, 1],
        "Model": "Random Classifier"
    })
    
    # Combine DataFrames
    plot_df = pd.concat([roc_df, diagonal_df])
    
    # Create plot
    fig = px.line(
        plot_df,
        x="False Positive Rate",
        y="True Positive Rate",
        color="Model",
        title="ROC Curve",
        color_discrete_map={
            f"ROC Curve (AUC = {roc_auc:.2f})": config.BRAND_COLORS["blue"],
            "Random Classifier": "gray"
        },
    )
    
    # Add optimal threshold point with a star
    fig.add_scatter(
        x=[optimal["fpr"]],
        y=[optimal["tpr"]],
        mode="markers",
        marker=dict(
            size=12,
            color=config.BRAND_COLORS["orange"],
            symbol="star"
        ),
        name=f"Optimal (threshold={optimal['threshold']:.2f})",
        hovertemplate=(
            f"<b>Optimal Threshold (Elbow)</b><br>"
            f"Threshold: {optimal['threshold']:.2f}<br>"
            f"Sensitivity: {optimal['sensitivity']:.2f}<br>"
            f"Specificity: {optimal['specificity']:.2f}<br>"
            f"False Positive Rate: {optimal['fpr']:.2f}<br>"
            f"True Positive Rate: {optimal['tpr']:.2f}<br>"
            f"<br>" 
            f"Interpretation:<br>" 
            f"<i>This is the best balance point where the model<br>"            # Non-technical explanation starts
            f"correctly identifies the most actual survivors<br>"
            f"while minimizing false positives (i.e., those the model <br>"
            f"predicted to survive who actually died). At this threshold,<br>"
            f"the model catches {optimal['sensitivity']:.0%} of survivors with only {optimal['fpr']:.0%} <br>"
            f"false positives.</i><br>"          # Explanation ends
            "<extra></extra>"
        )
    )
    
    # ROC curve line
    fig.update_traces(
        line=dict(width=2),
        selector=dict(name=f"ROC Curve (AUC = {roc_auc:.2f})"),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "True Positive Rate: %{y:.2f}<br>"              
            "False Positive Rate: %{x:.2f}<br>"              
            "<br>"                                           
            "<b>Interpretation:</b><br>"                                                         # Interpretation starts
            "<i>At this threshold setting, the model would<br>"
            "correctly identify %{y:.0%} of actual survivors<br>"
            "(True Positive Rate) but would also incorrectly<br>"
            "flag %{x:.0%} of non-survivors as survivors<br>"
            "(False Positive Rate).</i><br>"
            "<br>"
            f"<i>An Area Under the Curve (AUC) score of {roc_auc:.2f}<br>"
            f"means this model is {'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'acceptable' if roc_auc > 0.7 else 'moderate'} at distinguishing <br>"
            f"between survivors and non-survivors. <br>" 
            f"(Perfect = 1, Random = 0.5, Excellent > 0.9, <br>"
            f"Good > 0.8, Acceptable > 0.7)</i><br>"                                                  # AUC explanation
            "<extra></extra>"                                                                     # Interpretation ends  
        )
    )
    
    # Random classifier line
    fig.update_traces(
        line=dict(width=1, dash="dash"),
        selector=dict(name="Random Classifier"),
        hovertemplate=(
            "<b>Random Classifier (Baseline)</b><br>"
            "True Positive Rate: %{y:.2f}<br>"
            "False Positive Rate: %{x:.2f}<br>"
            "<br>"
            "<b>Interpretation:</b><br>"
            "<i>This diagonal line represents a model with<br>"
            "no predictive power—like flipping a coin. <br>"
            "It shows what would happen if we randomly<br>"
            "guessed who survived.</i><br>"
            "<br>"
            "<i>Our model's ROC curve suggests it's learning <br>"
            "real patterns, not just guessing. The farther it is above <br>"
            "this reference line the better the model! </i>🏆<br>"
            "<extra></extra>"
        )
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=0.6, y=0.1),
        xaxis=dict(gridcolor="rgba(200,200,200,0.3)", range=[0, 1]),
        yaxis=dict(gridcolor="rgba(200,200,200,0.3)", range=[0, 1])
    )
    
    # Log before returning
    logger.info("Returning ROC curve figure without showing")
    
    return fig

# ========================================================================
#   Model Persistence
# ========================================================================

def save_model(model, filepath=None):
    """
    Saves a trained model to disk.
    
    Args:
        model: Trained model to save
        filepath (str/Path): Where to save the model
        
    Returns:
        Path: Path where model was saved
    """
    
    if filepath is None:
        filepath = config.SAVED_MODELS_DIR / config.MODEL_FILENAME
    
    filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    print(f"✓ Model saved to: {filepath}")
    
    return filepath

def load_model(filepath=None):
    """
    Loads a trained model from disk.
    
    Args:
        filepath (str/Path): Path to saved model
        
    Returns:
        Loaded model
    """
    
    if filepath is None:
        filepath = config.SAVED_MODELS_DIR / config.MODEL_FILENAME
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found at {filepath}")
    
    # Load model
    model = joblib.load(filepath)
    print(f"✓ Model loaded from: {filepath}")
    
    return model


# ========================================================================
#   Prediction Functions
# ========================================================================

def predict_single(model, passenger_data, encoders):
    """
    Makes a prediction for a single passenger.
    
    Args:
        model: Trained model
        passenger_data (dict): Passenger information
        encoders (dict): Label encoders for categorical features
        
    Returns:
        dict: Prediction results
    """
    
    # Convert to DataFrame
    df = pd.DataFrame([passenger_data])
    
    # Import preprocessing to avoid circular import
    import preprocessing
    
    # Preprocess the data
    df_processed, _ = preprocessing.preprocess_data(
        df, 
        fit_encoders=False,
        encoders=encoders
    )
    
    # Get features
    X = preprocessing.get_feature_matrix(df_processed)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return {
        "survived": bool(prediction),
        "survival_probability": float(probability[1]),
        "death_probability": float(probability[0])
    }

def predict_batch(model, df, encoders):
    """
    Makes predictions for multiple passengers.
    
    Args:
        model: Trained model
        df (pd.DataFrame): Passenger data
        encoders (dict): Label encoders
        
    Returns:
        pd.DataFrame: Original data with predictions added
    """
    
    # Import preprocessing
    import preprocessing
    
    # Preprocess the data
    df_processed, _ = preprocessing.preprocess_data(
        df.copy(),
        fit_encoders=False,
        encoders=encoders
    )
    
    # Get features
    X = preprocessing.get_feature_matrix(df_processed)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Add to original dataframe
    df["Predicted_Survived"] = predictions
    df["Survival_Probability"] = probabilities
    
    return df