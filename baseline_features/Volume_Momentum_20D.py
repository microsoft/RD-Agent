import pandas as pd

def calculate_Volume_Momentum_20D():
    # Load the daily price and volume data
    df = pd.read_hdf("daily_pv.h5", key="data")
    
    # Sort the dataframe by instrument and datetime to ensure correct rolling calculation
    df = df.sort_index()
    
    # Calculate the 5-day moving average of volume for each instrument (Numerator)
    # window=5 corresponds to the average of current day and previous 4 days
    # min_periods=5 ensures the average is calculated only when 5 days of data are available
    vol_ma_5 = df.groupby(level='instrument')['$volume'].transform(
        lambda x: x.rolling(window=5, min_periods=5).mean()
    )
    
    # Calculate the denominator: 5-day average volume shifted back by 20 days
    # This represents the average volume for the window [t-20, t-24]
    vol_ma_5_lag20 = vol_ma_5.groupby(level='instrument').shift(20)
    
    # Calculate the factor value: Current 5-day avg volume / 5-day avg volume 20 days ago
    factor_values = vol_ma_5 / vol_ma_5_lag20
    
    # Create the result dataframe with the required format
    result_df = factor_values.to_frame('Volume_Momentum_20D')
    
    # Save the result to result.h5
    result_df.to_hdf("result.h5", key="data")

if __name__ == "__main__":
    calculate_Volume_Momentum_20D()