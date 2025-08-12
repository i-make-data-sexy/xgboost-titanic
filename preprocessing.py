"""
Data preprocessing functions for the Titanic dataset.
Handles feature engineering, missing values, and encoding.
"""

# ========================================================================
#   Imports
# ========================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import config

# ========================================================================
#   Feature Engineering
# ========================================================================

def create_family_features(df):
    """
    Creates family-related features from existing columns.
    
    Args:
        df (pd.DataFrame): DataFrame with SibSp and Parch columns
        
    Returns:
        pd.DataFrame: DataFrame with new family features
    """
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Family size (matching your dashboard)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    
    # Is alone feature
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    
    return df


def extract_title(df):
    """
    Extracts and simplifies titles from passenger names.
    
    Args:
        df (pd.DataFrame): DataFrame with Name column
        
    Returns:
        pd.DataFrame: DataFrame with Title column added
    """
    
    df = df.copy()
    
    # Extract title from name
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    
    # Group rare titles together
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Countess": "Rare",
        "Ms": "Miss",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Mme": "Mrs",
        "Capt": "Rare",
        "Sir": "Rare"
    }
    
    df["Title"] = df["Title"].map(title_mapping).fillna("Rare")
    
    return df


def create_age_groups(df):
    """
    Creates age group categories matching your dashboard.
    
    Args:
        df (pd.DataFrame): DataFrame with Age column
        
    Returns:
        pd.DataFrame: DataFrame with AgeGroup column
    """
    
    df = df.copy()
    
    # Handle missing Age values gracefully
    if df["Age"].notna().sum() == 0:  # All ages are NaN
        df["AgeGroup"] = "Unknown"
        return df
    
    # Define bins matching your dashboard
    age_bins = [0, 12, 18, 35, 60, 100]
    age_labels = ["Child", "Teen", "Young Adult", "Middle Age", "Senior"]
    
    # Create age groups (NaN values will remain NaN in the result)
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
    
    return df


def create_fare_bins(df):
    """
    Creates fare categories using quartiles.
    
    Args:
        df (pd.DataFrame): DataFrame with Fare column
        
    Returns:
        pd.DataFrame: DataFrame with FareBin column
    """
    
    df = df.copy()
    
    # Skip binning if we have too few unique values (like single predictions)
    if len(df) < 4 or df["Fare"].nunique() < 4:
        # For small datasets or predictions, assign based on absolute values
        def assign_fare_category(fare):
            if fare < 10:
                return "Low"
            elif fare < 30:
                return "Medium"
            elif fare < 100:
                return "High"
            else:
                return "Very High"
        
        df["FareBin"] = df["Fare"].apply(assign_fare_category)
    else:
        # Create fare bins using quartiles for larger datasets
        df["FareBin"] = pd.qcut(
            df["Fare"], 
            q=4, 
            labels=["Low", "Medium", "High", "Very High"],
            duplicates="drop"
        )
    
    return df


def create_cabin_feature(df):
    """
    Creates a binary feature for cabin information.
    
    Args:
        df (pd.DataFrame): DataFrame with Cabin column
        
    Returns:
        pd.DataFrame: DataFrame with HasCabin column
    """
    
    df = df.copy()
    
    # Binary feature: has cabin info or not
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    
    return df


# ========================================================================
#   Missing Value Handling
# ========================================================================

def fill_missing_ages(df):
    """
    Fills missing age values using group medians.
    
    Args:
        df (pd.DataFrame): DataFrame with Age column
        
    Returns:
        pd.DataFrame: DataFrame with filled Age values
    """
    
    df = df.copy()
    
    # Fill missing ages with median by Pclass and Sex
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Keep remaining missing values as NaN for transparency
    # XGBoost handles NaN values natively
    
    return df


def fill_missing_embarked(df):
    """
    Keeps missing Embarked values as NaN for transparency.
    
    Args:
        df (pd.DataFrame): DataFrame with Embarked column
        
    Returns:
        pd.DataFrame: DataFrame with Embarked column unchanged
    """
    
    df = df.copy()
    
    # Keep missing Embarked as NaN - XGBoost handles it
    # Only 2 missing values in Titanic dataset
    print(f"Note: {df['Embarked'].isna().sum()} missing Embarked values kept as NaN")
    
    return df


def fill_missing_fare(df):
    """
    Fills missing Fare values using class medians.
    
    Args:
        df (pd.DataFrame): DataFrame with Fare column
        
    Returns:
        pd.DataFrame: DataFrame with filled Fare values
    """
    
    df = df.copy()
    
    # Fill missing fares with median by Pclass
    df["Fare"] = df.groupby("Pclass")["Fare"].transform(
        lambda x: x.fillna(x.median())
    )
    
    # If still missing, use overall median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    return df


# ========================================================================
#   Encoding Functions
# ========================================================================

def encode_categorical_features(df, fit_encoders=True, encoders=None):
    """
    Encodes categorical features for model training.
    
    Args:
        df (pd.DataFrame): DataFrame with categorical features
        fit_encoders (bool): Whether to fit new encoders or use existing
        encoders (dict): Existing encoders to use if fit_encoders=False
        
    Returns:
        tuple: (encoded DataFrame, dictionary of encoders)
    """
    
    df = df.copy()
    
    # Initialize encoders dictionary if fitting
    if fit_encoders:
        encoders = {}
    
    # Define categorical columns to encode
    categorical_columns = {
        "Sex": "Sex_Encoded",
        "Embarked": "Embarked_Encoded",
        "Title": "Title_Encoded",
        "FareBin": "FareBin_Encoded",    
        "AgeGroup": "AgeGroup_Encoded"    
    }
    
    for original_col, encoded_col in categorical_columns.items():
        if original_col in df.columns:
            
            if fit_encoders:
                # Create and fit new encoder
                encoder = LabelEncoder()
                df[encoded_col] = encoder.fit_transform(df[original_col])
                encoders[original_col] = encoder
                
            else:
                # Use existing encoder
                if original_col in encoders:
                    # Handle unseen categories
                    encoder = encoders[original_col]
                    df[encoded_col] = df[original_col].apply(
                        lambda x: encoder.transform([x])[0] 
                        if x in encoder.classes_ else -1
                    )
    
    return df, encoders


# ========================================================================
#   Main Preprocessing Pipeline
# ========================================================================

def preprocess_data(df, fit_encoders=True, encoders=None):
    """
    Complete preprocessing pipeline for the Titanic data.
    
    Args:
        df (pd.DataFrame): Raw Titanic DataFrame
        fit_encoders (bool): Whether to fit new encoders
        encoders (dict): Existing encoders for prediction
        
    Returns:
        tuple: (processed DataFrame, encoders dictionary)
    """
    
    print("Starting data preprocessing...")
    
    # Feature engineering
    print("Creating family features...")
    df = create_family_features(df)
    
    print("Extracting titles...")
    df = extract_title(df)
    
    print("Creating cabin feature...")
    df = create_cabin_feature(df)
    
    # Handle missing values
    print("Filling missing values...")
    df = fill_missing_ages(df)
    df = fill_missing_embarked(df)
    df = fill_missing_fare(df)
    
    # Create derived features that depend on filled values
    print("Creating age groups...")
    df = create_age_groups(df)
    
    print("Creating fare bins...")
    df = create_fare_bins(df)
    
    # Encode categorical features
    print("Encoding categorical features...")
    df, encoders = encode_categorical_features(df, fit_encoders, encoders)
    
    print("Preprocessing complete!")
    
    return df, encoders


def get_feature_matrix(df, feature_cols=None):
    """
    Extracts feature matrix for model training.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
        feature_cols (list): List of feature columns to use
        
    Returns:
        pd.DataFrame: Feature matrix
    """
    
    if feature_cols is None:
        feature_cols = config.FEATURE_COLUMNS
    
    # Filter to only columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # Warn if any columns are missing
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
    
    return df[available_cols]