# src/geometry_metrics.py
import pandas as pd
import numpy as np

def calculate_herfindahl(df: pd.DataFrame, group_cols=['DeliveryDate', 'HourEnding', 'AncillaryType']) -> pd.DataFrame:
    """
    Calculates the Herfindahl-Hirschman Index (H_k) for Day-Ahead awards.
    
    Formula: H_k = Sum( (q_i / Q_total)^2 )
    High H_k (> 0.4) indicates a 'convex' or brittle supply stack.
    """
    # Ensure we are working with numeric quantities
    df['AwardedMW'] = pd.to_numeric(df['AwardedMW'], errors='coerce').fillna(0)
    
    # Calculate total Q per hour/product
    total_q = df.groupby(group_cols)['AwardedMW'].transform('sum')
    
    # Calculate market share squared for each block
    df['share_sq'] = (df['AwardedMW'] / total_q) ** 2
    
    # Sum squares to get HHI
    hhi = df.groupby(group_cols)['share_sq'].sum().reset_index()
    hhi.rename(columns={'share_sq': 'H_k'}, inplace=True)
    
    return hhi

def calculate_stack_convexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnoses 'Stack Convexity' by comparing the marginal price block vs. average price.
    
    Convexity Ratio = Max_Price / Volume_Weighted_Avg_Price
    High ratio indicates a 'hockey stick' geometry.
    """
    # VWAP Calculation
    df['revenue'] = df['AwardedMW'] * df['ClearingPrice']
    grouped = df.groupby(['DeliveryDate', 'HourEnding', 'AncillaryType'])
    
    metrics = grouped.agg(
        Total_MW=('AwardedMW', 'sum'),
        Max_Price=('ClearingPrice', 'max'),
        Total_Rev=('revenue', 'sum')
    ).reset_index()
    
    metrics['VWAP'] = metrics['Total_Rev'] / metrics['Total_MW']
    metrics['Convexity_Ratio'] = metrics['Max_Price'] / metrics['VWAP']
    
    return metrics