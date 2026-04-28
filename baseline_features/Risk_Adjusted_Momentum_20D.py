import pandas as pd
import numpy as np

def calculate_Risk_Adjusted_Momentum_20D():
    # Load the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Sort index to ensure correct time series operations
    df = df.sort_index()
    
    # Get close price
    close = df['$close']
    
    # 1. Calculate R_{t, 20}: The cumulative return over the past 20 trading days.
    # Formula: (Close_t - Close_{t-20}) / Close_{t-20}
    # Using pct_change(20) grouped by instrument
    cumulative_return = close.groupby(level='instrument').pct_change(20)
    
    # 2. Calculate sigma_{t, 20}: The standard deviation of daily logarithmic returns over the past 20 trading days.
    # Step 2a: Calculate daily logarithmic returns r_t = ln(Close_t / Close_{t-1})
    prev_close = close.groupby(level='instrument').shift(1)
    log_ret = np.log(close / prev_close)
    
    # Step 2b: Calculate rolling standard deviation
    # window=20, min_periods=20 ensures exactly 20 data points are used
    # std() uses ddof=1 by default, matching the formula 1/(N-1)
    rolling_std = log_ret.groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=20).std()
    )
    
    # 3. Calculate the factor: F_t = R_{t, 20} / sigma_{t, 20}
    factor_value = cumulative_return / rolling_std
    
    # Create the result dataframe
    result_df = factor_value.to_frame("Risk_Adjusted_Momentum_20D")
    
    # Save the result to result.h5
    result_df.to_hdf("result.h5", key="data")

if __name__ == "__main__":
    calculate_Risk_Adjusted_Momentum_20D()