import pandas as pd
import numpy as np

def calculate_Return_ZScore_20D():
    # Load the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Sort index to ensure correct time series operations
    df = df.sort_index()
    
    # Calculate daily simple returns: r_t = (Close_t - Close_{t-1}) / Close_{t-1}
    # Using pct_change() is equivalent to the formula.
    # We group by instrument to ensure we don't mix returns between different stocks.
    close = df['$close']
    ret = close.groupby(level='instrument').pct_change()
    
    # Calculate the 20-day rolling mean of returns
    # window=20, min_periods=20 ensures we use exactly 20 data points as per formula
    rolling_mean = ret.groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=20).mean()
    )
    
    # Calculate the 20-day rolling standard deviation of returns
    # Pandas std() uses ddof=1 by default, which matches the formula's denominator (N-1) = 19
    rolling_std = ret.groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=20).std()
    )
    
    # Calculate Z-Score: F_t = (r_t - mu_{t,20}) / sigma_{t,20}
    z_score = (ret - rolling_mean) / rolling_std
    
    # Create the result dataframe
    result_df = z_score.to_frame("Return_ZScore_20D")
    
    # Save the result to result.h5
    result_df.to_hdf("result.h5", key="data")

if __name__ == "__main__":
    calculate_Return_ZScore_20D()