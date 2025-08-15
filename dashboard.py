"""
Dashboard creation functions for the Titanic analysis.
All visualization logic lives here.
"""

import pandas as pd
import plotly.express as px
import config

# ========================================================================
#   Data Preparation for Dashboard
# ========================================================================

def prepare_dashboard_data(df):
    """
    Prepares the data specifically for dashboard visualizations.
    
    Args:
        df (pd.DataFrame): Raw Titanic data
        
    Returns:
        pd.DataFrame: Data with additional columns for visualization
    """
    
    # Add FamilySize feature
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    
    # Add age groups for visualization
    age_bins = [0, 12, 18, 35, 60, 100]
    age_labels = ["Child", "Teen", "Young Adult", "Middle Age", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
    
    return df


# ========================================================================
#   Individual Chart Creation Functions
# ========================================================================

def create_class_survival_chart(df):
    """
    Creates a horizontal bar chart showing survivor count by passenger class.
    
    Args:
        df (pd.DataFrame): Titanic dataset
        
    Returns:
        plotly.express.Figure: Bar chart
    """
    
    # Filter to survivors only
    df_survivors = df[df["Survived"] == 1].copy()
    
    # Aggregate by class
    class_counts = (
        df_survivors.groupby("Pclass")
        .size()
        .reset_index(name="Count")
    )
    
    # Map class numbers to names
    class_names = {1: "First", 2: "Second", 3: "Third"}
    class_counts["Class"] = class_counts["Pclass"].map(class_names)
    
    # Create bar chart
    fig = px.bar(
        class_counts,
        x="Count",
        y="Class",
        orientation="h",
        title="Survivor Count by Class",
        text="Count",
        color_discrete_sequence=[config.BRAND_COLORS["blue"]]
    )
    
    # Customize appearance
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y} Class</b><br>Survivors: %{x}<extra></extra>"
    )
    
    fig.update_yaxes(type="category", autorange="reversed")
    # fig.update_xaxes(visible=False)
    
    fig.update_xaxes(
        # visible=False,
        zeroline=True,                          # Show the zero line
        zerolinewidth=1,                        # Make it thin (adjust as needed: 0.5 for thinner)
        zerolinecolor="#DEDEDE",              # Light gray color
        showline=False,
        showgrid=False,
        title="",
        showticklabels=False
    )
    
    # Extend x-axis for data labels
    max_count = class_counts["Count"].max()
    fig.update_layout(
        yaxis_title="",
        xaxis=dict(range=[0, max_count * 1.1]),
        showlegend=False,
        margin_pad=5,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=60, b=40, l=80, r=40),
        height=350
    )
    
    return fig


def create_gender_survival_donut(df):
    """
    Creates a donut chart showing survival rate by gender.
    
    Args:
        df (pd.DataFrame): Titanic dataset
        
    Returns:
        plotly.express.Figure: Donut chart
    """
    
    # Calculate survival rate by gender
    gender_survival = (
        df.groupby("Sex")["Survived"]
        .mean()
        .reset_index()
    )
    gender_survival["Survived"] = gender_survival["Survived"] * 100
    gender_survival["Sex_Display"] = gender_survival["Sex"].str.capitalize()
    
    # Create donut chart
    fig = px.pie(
        gender_survival,
        values="Survived",
        names="Sex_Display",
        hole=0.5,
        title="Survival Rate by Gender",
        color_discrete_sequence=[config.BRAND_COLORS["orange"], config.BRAND_COLORS["blue"]]
    )
    
    # Customize appearance
    fig.update_traces(
        texttemplate="%{label}<br>%{value:.0f}%",
        textfont_size=14,
        textfont_color="white",
        marker=dict(line=dict(color="white", width=2)),
        hovertemplate="<b>%{label}</b><br>Survival Rate: %{value:.1f}%<extra></extra>"
    )
    
    fig.update_layout(
        paper_bgcolor="white",
        margin=dict(t=60, b=40, l=40, r=40),
        height=350,
        showlegend=False
    )
    
    return fig

def create_family_survival_line(df):
    """
    Creates a line chart showing survival rate by family size.
    
    Args:
        df (pd.DataFrame): Titanic dataset with FamilySize column
        
    Returns:
        plotly.express.Figure: Line chart
    """
    
    # Calculate survival rate by family size
    family_survival = (
        df.groupby("FamilySize")["Survived"]
        .mean()
        .reset_index()
    )
    family_survival["Survived"] = family_survival["Survived"] * 100
    
    # Create line chart using scatter with lines mode
    fig = px.scatter(
        family_survival,
        x="FamilySize",
        y="Survived",
        title="Survival Rate by Family Size",
        color_discrete_sequence=[config.BRAND_COLORS["blue"]]
    )
    
    # Convert to line chart with markers
    fig.update_traces(
        mode="lines+markers",
        marker=dict(size=10, color=config.BRAND_COLORS["orange"]),
        line=dict(width=3, color=config.BRAND_COLORS["blue"]),
        hovertemplate="Family Size: %{x}<br>Survival Rate: %{y:.0f}%<extra></extra>"
    )
    
    fig.update_layout(
        yaxis=dict(
            title="",
            gridcolor="rgba(200,200,200,0.3)",
            ticksuffix="%",
            zeroline=False,                     # Let the grid line at 0% be the only line
            showgrid=True,
        ),
        xaxis=dict(
            title="",
            gridcolor="rgba(200,200,200,0.3)",
            range=[0, family_survival["FamilySize"].max() + 1]
        ),
        margin_pad=5,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=60, b=40, l=60, r=40),
        height=350
    )
    
    return fig


def create_age_survival_chart(df):
    """
    Creates a 100% stacked bar chart showing survival rate by age group.
    
    Args:
        df (pd.DataFrame): Titanic dataset with AgeGroup column
        
    Returns:
        plotly.express.Figure: Stacked bar chart
    """
    
    # Create survival counts by age group
    age_survival = (
        df.groupby(["AgeGroup", "Survived"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    
    # Calculate percentages within each age group
    age_totals = (
        age_survival.groupby("AgeGroup")["Count"]
        .sum()
        .reset_index(name="Total")
    )
    age_survival = age_survival.merge(age_totals, on="AgeGroup")
    age_survival["Percentage"] = (age_survival["Count"] / age_survival["Total"]) * 100
    
    # Map survived values to readable labels
    age_survival["Status"] = age_survival["Survived"].map({0: "Did Not Survive", 1: "Survived"})
    
    # Calculate survival rate for sorting
    survival_rates = (
        age_survival[age_survival["Survived"] == 1]
        [["AgeGroup", "Percentage"]]
        .rename(columns={"Percentage": "SurvivalRate"})
    )
    age_survival = age_survival.merge(survival_rates, on="AgeGroup")
    
    # Sort by survival rate
    age_survival = age_survival.sort_values(
        ["SurvivalRate", "Survived"], 
        ascending=[False, False]
    )
    
    # Create stacked bar chart
    fig = px.bar(
        age_survival,
        x="AgeGroup",
        y="Percentage",
        color="Status",
        title="Survival Rate by Age Group",
        text="Percentage",
        color_discrete_map={
            "Survived": config.BRAND_COLORS["blue"],
            "Did Not Survive": "#B8D4E8"  # Light blue
        }
    )
    
    # Update traces individually to ensure proper hover data
    for trace in fig.data:
        # Get the status from the trace name (which matches the color column)
        status = trace.name
        
        # Update this specific trace with its correct hover template
        trace.update(
            texttemplate="%{text:.0f}%",
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>" + status + ": %{y:.0f}%<extra></extra>"
        )
    
    fig.update_layout(
        showlegend=True,
        legend_title_text="",
        yaxis=dict(
            title="", 
            zeroline=True,                      # Show the zero line
            zerolinewidth=1,                    # Make it thin (adjust as needed: 0.5 for thinner)
            zerolinecolor="#DEDEDE",          # Light gray color
            showline=False,
            showgrid=False,
            showticklabels=False),
        xaxis_title="",
        margin_pad=5,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=60, b=40, l=40, r=40),
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# ========================================================================
#   Main Dashboard Creation
# ========================================================================

def create_all_dashboard_charts(df):
    """
    Creates all dashboard charts from the Titanic data.
    
    Args:
        df (pd.DataFrame): Raw Titanic dataset
        
    Returns:
        dict: Dictionary containing all charts
    """
    
    # Prepare data
    df = prepare_dashboard_data(df)
    
    # Create all charts
    charts = {
        "class_chart": create_class_survival_chart(df),
        "gender_chart": create_gender_survival_donut(df),
        "age_chart": create_age_survival_chart(df),
        "family_chart": create_family_survival_line(df)
    }
    
    return charts


def convert_chart_to_json(fig):
    """
    Converts a Plotly figure to clean JSON for the frontend.
    
    Args:
        fig: Plotly figure
        
    Returns:
        str: JSON string
    """
    
    import json
    import numpy as np
    
    # Get figure dictionary
    fig_dict = fig.to_dict()
    
    # Clean numpy types for JSON serialization
    def clean_dict(obj):
        if isinstance(obj, dict):
            return {k: clean_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_dict(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    cleaned = clean_dict(fig_dict)
    return json.dumps(cleaned)