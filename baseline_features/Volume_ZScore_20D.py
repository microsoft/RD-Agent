import pandas as pd
import numpy as np

def calculate_Volume_ZScore_20D():
    # Load the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Sort the dataframe by datetime to ensure the time-series operations are correct
    # The index is MultiIndex with levels ['datetime', 'instrument']
    df = df.sort_index(level='datetime')
    
    # Reshape the volume data to wide format (dates as index, instruments as columns)
    # This avoids index alignment issues associated with groupby().rolling()
    volume_wide = df['$volume'].unstack(level='instrument')
    
    # Calculate 20-day rolling mean and standard deviation
    rolling_mean = volume_wide.rolling(window=20).mean()
    rolling_std = volume_wide.rolling(window=20).std()
    
    # Calculate the Z-Score: (Volume - Moving Average) / Moving Standard Deviation
    factor_values_wide = (volume_wide - rolling_mean) / rolling_std
    
    # Replace infinite values with NaN (occurs if std is 0)
    factor_values_wide.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Reshape back to long format (MultiIndex: datetime, instrument)
    factor_values = factor_values_wide.stack()
    
    # Create the result dataframe
    result_df = factor_values.to_frame('Volume_ZScore_20D')
    
    # Save the result to result.h5
    result_df.to_hdf("result.h5", key="data")

if __name__ == "__main__":
    calculate_Volume_ZScore_20D()