import pandas as pd
import numpy as np

def calculate_Volatility_20D():
    # Load the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Sort index to ensure correct time series operations
    df = df.sort_index()
    
    # Extract adjusted close prices
    # The data description states it is adjusted daily price data
    close = df['$close']
    
    # Unstack to perform calculations across instruments (wide format)
    # Resulting index: datetime, columns: instruments
    close_unstacked = close.unstack(level='instrument')
    
    # Calculate daily logarithmic returns: r_t = ln(P_t / P_{t-1})
    # log(P_t) - log(P_{t-1})
    log_returns = np.log(close_unstacked) - np.log(close_unstacked.shift(1))
    
    # Calculate rolling standard deviation over a 20-day window
    # Pandas rolling().std() uses ddof=1 by default, which matches the sample standard deviation formula
    # Formula: sqrt(1/(N-1) * sum((r - mean)^2))
    volatility = log_returns.rolling(window=20).std()
    
    # Stack back to the original MultiIndex format (datetime, instrument)
    factor_series = volatility.stack()
    
    # Name the series according to the factor definition
    factor_series.name = "Volatility_20D"
    
    # Convert to DataFrame
    result_df = factor_series.to_frame()
    
    # Ensure index names are correct
    result_df.index.names = ['datetime', 'instrument']
    
    # Save the result to result.h5
    result_df.to_hdf("result.h5", key="data")

if __name__ == "__main__":
    calculate_Volatility_20D()