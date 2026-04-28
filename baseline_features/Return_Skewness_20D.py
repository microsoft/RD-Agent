import pandas as pd
import numpy as np

def calculate_return_skewness_20d():
    # Read the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Sort by instrument and datetime
    df = df.sort_index()
    
    # Calculate daily returns: r_t = C_t / C_{t-1} - 1
    df['return'] = df.groupby(level='instrument')['$close'].pct_change()
    
    # Define population skewness function
    # Skewness = (1/N) * sum(((r - mean) / std)^3)
    def pop_skewness(x):
        mean = np.mean(x)
        std = np.std(x, ddof=0)  # population standard deviation
        if std == 0 or np.isnan(std):
            return np.nan
        return np.mean((x - mean) ** 3) / (std ** 3)
    
    # Calculate rolling skewness over 20-day window
    df['Return_Skewness_20D'] = df.groupby(level='instrument')['return'].transform(
        lambda x: x.rolling(window=20, min_periods=20).apply(pop_skewness, raw=True)
    )
    
    # Prepare the result
    result = df[['Return_Skewness_20D']].copy()
    
    # Save to HDF5 file
    result.to_hdf("result.h5", key="data")
    
    return result

if __name__ == "__main__":
    calculate_return_skewness_20d()