import pandas as pd
import numpy as np

def calculate_Return_Sign_Autocorrelation_20D():
    # Load the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Reset index to get 'datetime' and 'instrument' as columns for sorting
    df = df.reset_index()
    
    # Sort by instrument and datetime to ensure correct time-series operations
    df = df.sort_values(by=['instrument', 'datetime'])
    
    # Calculate daily returns r_t
    df['ret'] = df.groupby('instrument')['$close'].pct_change()
    
    # Calculate the sign of the daily return
    # np.sign returns 1 for positive, -1 for negative, 0 for zero, and NaN for NaN.
    df['sign'] = np.sign(df['ret'])
    
    # Calculate the rolling autocorrelation of the sign series with lag 1
    # The formulation asks for Corr(sign(r_{t-i}), sign(r_{t-i-1})) for i=0..19
    # This is equivalent to rolling correlation between the series and its lag-1 version over a window of 20.
    # We use min_periods=20 to ensure we have a full 20-day window.
    
    def rolling_autocorr(x, window=20):
        return x.rolling(window=window, min_periods=window).corr(x.shift(1))
    
    df['Return_Sign_Autocorrelation_20D'] = df.groupby('instrument')['sign'].transform(rolling_autocorr)
    
    # Set the index back to ['datetime', 'instrument'] as required
    result = df.set_index(['datetime', 'instrument'])[['Return_Sign_Autocorrelation_20D']]
    
    # Save the result to result.h5
    result.to_hdf("result.h5", key="data")

if __name__ == "__main__":
    calculate_Return_Sign_Autocorrelation_20D()