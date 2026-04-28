import pandas as pd
import numpy as np

def calculate_Return_Kurtosis_20D():
    # Load the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Sort index to ensure correct time series operations
    df = df.sort_index()
    
    # Calculate daily arithmetic returns: r_t = (Close_t - Close_{t-1}) / Close_{t-1}
    close = df['$close']
    prev_close = close.groupby(level='instrument').shift(1)
    ret = (close - prev_close) / prev_close
    
    # Calculate rolling mean of returns over the past 20 days
    rolling_mean = ret.groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=20).mean()
    )
    
    # Calculate deviations from the mean
    dev = ret - rolling_mean
    
    # Calculate the 4th moment (numerator part): mean of (deviation)^4
    # Formula part: (1/20) * sum((r - r_bar)^4)
    m4 = dev.pow(4).groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=20).mean()
    )
    
    # Calculate the squared 2nd moment (denominator part): (mean of (deviation)^2)^2
    # Formula part: ((1/20) * sum((r - r_bar)^2))^2
    m2 = dev.pow(2).groupby(level='instrument').transform(
        lambda x: x.rolling(window=20, min_periods=20).mean()
    )
    
    # Calculate Kurtosis: m4 / (m2)^2
    kurtosis = m4 / (m2 ** 2)
    
    # Create the result dataframe
    result_df = kurtosis.to_frame("Return_Kurtosis_20D")
    
    # Save the result to result.h5
    result_df.to_hdf("result.h5", key="data")

if __name__ == "__main__":
    calculate_Return_Kurtosis_20D()