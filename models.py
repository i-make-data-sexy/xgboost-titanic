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

def train_basic_model(X_train, y_train, X_test=None, y_test=None):
    """
    Trains a basic XGBoost model with default parameters.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Optional test features for evaluation
        y_test (pd.Series): Optional test target
        
    Returns:
        xgb.XGBClassifier: Trained model
    """
    
    print("\nTraining basic XGBoost model...")
    
    # Create model with default parameters
    model = xgb.XGBClassifier(
        **config.XGBOOST_DEFAULT_PARAMS,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1
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


def tune_hyperparameters(X_train, y_train, param_grid=None, cv_folds=None):
    """
    Performs hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (dict): Parameters to search
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        xgb.XGBClassifier: Best tuned model
    """
    
    print("\nTuning hyperparameters (this may take a minute)...")
    
    # Use defaults if not provided
    if param_grid is None:
        param_grid = config.HYPERPARAM_GRID
    if cv_folds is None:
        cv_folds = config.CV_FOLDS
    
    # Base model for grid search
    base_model = xgb.XGBClassifier(**config.XGBOOST_DEFAULT_PARAMS)
    
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
        target_names=["Not Survived", "Survived"]
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
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance * 100
    }).sort_values("Importance", ascending=True)
    
    # Create bar chart
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance Scores",
        text="Importance",
        color_discrete_sequence=[config.BRAND_COLORS["blue"]]
    )
    
    # Format text on bars
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )
    
    fig.update_layout(
        xaxis=dict(
            title="",
            showgrid=False,
            showticklabels=False
        ),
        yaxis_title="",
        height=600,
        width=1000,
        margin=dict(l=100, r=50, t=50, b=50),
        margin_pad=5,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # Add the vertical line at x=0 (the axis spine)
    fig.update_xaxes(
        zeroline=True,              # Show the zero line
        zerolinewidth=1,            # Make it thin (adjust as needed: 0.5 for thinner)
        zerolinecolor="#DEDEDE",  # Light gray color
        showline=False,
        showgrid=False
    )
    
    return fig


def plot_confusion_matrix(cm, labels=["Not Survived", "Survived"]):
    """
    Creates a Plotly confusion matrix heatmap.
    
    Args:
        cm (np.array): Confusion matrix
        labels (list): Class labels
        
    Returns:
        plotly.graph_objects.Figure: Confusion matrix heatmap
    """
    
    # Create hover text with classification labels
    hover_text = [
        ["True Negatives (TN)", "False Positives (FP)"],
        ["False Negatives (FN)", "True Positives (TP)"]
    ]
    
    # Create detailed hover template
    hover_details = []
    for i in range(2):
        row_details = []
        for j in range(2):
            count = cm[i, j]
            classification = hover_text[i][j]
            actual = labels[i]
            predicted = labels[j]
            
            # Build hover text
            text = (
                f"<b>{classification}</b><br>"
                f"Count: {count}<br>"
                f"Actual: {actual}<br>"
                f"Predicted: {predicted}<br>"
            )
            
            # Add interpretation
            if i == 0 and j == 0:  # TN
                text += "<br>✅ Correctly predicted death"
            elif i == 0 and j == 1:  # FP
                text += "<br>❌ Incorrectly predicted survival<br>(False hope)"
            elif i == 1 and j == 0:  # FN
                text += "<br>❌ Missed a survivor<br>(Surprise survival)"
            else:  # TP
                text += "<br>✅ Correctly predicted survival"
            
            row_details.append(text)
        hover_details.append(row_details)
    
    # Create a color matrix - 1 for correct (diagonal), 0 for incorrect (off-diagonal)
    color_matrix = np.zeros_like(cm, dtype=float)
    np.fill_diagonal(color_matrix, 1)
    
    # Create custom colorscale - pink (0) to green (1)
    colorscale = [
        [0, config.BRAND_COLORS["pink"]],  # Incorrect predictions
        [1, config.BRAND_COLORS["green"]]  # Correct predictions
    ]
    
    # Use go.Heatmap instead of px.imshow for more control
    fig = go.Figure(data=go.Heatmap(
        z=color_matrix,  # Use color matrix for colors
        text=cm,  # Use actual values for text
        texttemplate="%{text}",  # Show the values
        textfont={"size": 20},
        x=labels,
        y=labels,
        colorscale=colorscale,
        showscale=False,
        customdata=hover_details,
        hovertemplate="%{customdata}<extra></extra>"
    ))
    
    # NEW: Calculate accuracy metrics
    total = cm.sum()
    correct = cm[0,0] + cm[1,1]  # Diagonal sum (TN + TP)
    incorrect = cm[0,1] + cm[1,0]  # Off-diagonal sum (FP + FN)
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
            autorange="reversed"            # Flip y-axis to match standard confusion matrix
        ),
        width=700,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=60, b=180)
    )
    
    # Add interpretation annotation in bottom-left corner
    fig.add_annotation(
        text=(
            "<b>How to Read:</b><br>"
            f"• Green cells = Correct predictions ({correct}/{total} = {accuracy:.1f}%)<br>"
            f"• Pink cells = Incorrect predictions ({incorrect}/{total} = {error_rate:.1f}%)<br>"
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
    
    return fig


def plot_roc_curve(y_true, y_scores):
    """
    Creates a Plotly ROC curve.
    
    Args:
        y_true (np.array): True labels
        y_scores (np.array): Predicted probabilities
        
    Returns:
        plotly.graph_objects.Figure: ROC curve
    """
    
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve (include thresholds)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create DataFrame for Plotly Express
    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr,
        "Model": f"ROC Curve (AUC = {roc_auc:.2f})"
    })
    
    # Add diagonal reference line data for the Random Classifier
    diagonal_df = pd.DataFrame({
        "False Positive Rate": [0, 1],
        "True Positive Rate": [0, 1],
        "Model": "Random Classifier"
    })
    
    # Combine both DataFrames
    plot_df = pd.concat([roc_df, diagonal_df])
    
    # Create plot using Plotly Express
    fig = px.line(
        plot_df,
        x="False Positive Rate",
        y="True Positive Rate",
        color="Model",
        title="ROC Curve",
        color_discrete_map={
            f"ROC Curve (AUC = {roc_auc:.2f})": config.BRAND_COLORS["blue"],
            "Random Classifier": "gray"
        }
    )
    
    # ROC curve annotation styling
    fig.update_traces(
        line=dict(width=2),
        selector=dict(name=f"ROC Curve (AUC = {roc_auc:.2f})")
    )
    
    # Random Classifier annotation styling
    fig.update_traces(
        line=dict(
            width=1, 
            dash="dash"),
        selector=dict(name="Random Classifier")
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=0.6, y=0.1),
        margin_pad=5,
        xaxis=dict(
            gridcolor="rgba(200,200,200,0.3)", 
            range=[0, 1]),
        yaxis=dict(
            gridcolor="rgba(200,200,200,0.3)", 
            range=[0, 1])
    )
    
    fig.update_xaxes(
        zeroline=True,              # Show the vertical zero line
        zerolinewidth=0.5,          # Make it thin 
        zerolinecolor="#DEDEDE",  # Light gray color
        showline=False,
        showgrid=False
    )
    
    fig.update_yaxes(
        zeroline=True,              # Show the horizontal zero line
        zerolinewidth=0.5,          # Make it thin 
        zerolinecolor="#DEDEDE",  # Light gray color
        showline=False,
        showgrid=False
    )
    
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