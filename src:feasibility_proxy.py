# src/feasibility_proxy.py
import pandas as pd

def detect_participation_exhaustion(rt_awards: pd.DataFrame, lookback_window=12) -> pd.DataFrame:
    """
    Identifies resources that abruptly cease participation after sustained deployment.
    
    Logic:
    1. 'Active': Award > 0 in previous intervals.
    2. 'Depleted': Award drops to 0 while Prices (MCPC) remain High.
    3. 'Withholding Flag': If Price is High but resource did NOT discharge prior (No SoC depletion).
    """
    # Sort by resource and time
    df = rt_awards.sort_values(['ResourceName', 'SCEDTimestamp']).copy()
    
    # Calculate rolling discharge (proxy for SoC usage)
    df['rolling_discharge'] = df.groupby('ResourceName')['BasePoint'].transform(
        lambda x: x.rolling(window=lookback_window).sum()
    )
    
    # Define "High Price" regime (e.g., > $100)
    df['high_price_env'] = df['MCPC_ECRS'] > 100
    
    # Detect dropouts
    df['prev_award'] = df.groupby('ResourceName')['AwardedMW'].shift(1)
    df['dropout'] = (df['prev_award'] > 0) & (df['AwardedMW'] == 0)
    
    # Classify failure mode
    # If dropout happens + High Price + High Prior Discharge -> Physical Exhaustion (Feasibility)
    # If dropout happens + High Price + LOW Prior Discharge -> Potential Withholding
    
    def classify_event(row):
        if not row['dropout'] or not row['high_price_env']:
            return 'Normal'
        if row['rolling_discharge'] > 50: # Threshold for "Active"
            return 'Feasibility_Exhaustion'
        else:
            return 'Strategic_Withholding_Risk'

    df['failure_mode'] = df.apply(classify_event, axis=1)
    return df