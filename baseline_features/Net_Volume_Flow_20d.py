import pandas as pd
import numpy as np

def calculate_Net_Volume_Flow_20d():
    # Load the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Sort the dataframe by index (datetime, instrument) to ensure correct time series order
    df = df.sort_index()
    
    # Calculate the direction of the price movement
    # Sign function: 1 if Close > Open, -1 if Close < Open, 0 if Close == Open
    price_diff = df['$close'] - df['$open']
    direction = np.sign(price_diff)
    
    # Calculate signed volume
    signed_volume = df['$volume'] * direction
    
    # Calculate rolling sum of signed volume (Numerator)
    # Window size is 20, min_periods is 20 to ensure we have a full window
    numerator = signed_volume.groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=20).sum()
    )
    
    # Calculate rolling sum of total volume (Denominator)
    denominator = df['$volume'].groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=20).sum()
    )
    
    # Calculate the factor value
    factor_value = numerator / denominator
    
    # Replace infinite values with NaN (can happen if total volume is 0)
    factor_value = factor_value.replace([np.inf, -np.inf], np.nan)
    
    # Create the result dataframe with the required column name
    result_df = pd.DataFrame(factor_value)
    result_df.columns = ['Net_Volume_Flow_20d']
    
    # Save the result to result.h5
    result_df.to_hdf("result.h5", key="data")

if __name__ == "__main__":
    calculate_Net_Volume_Flow_20d()