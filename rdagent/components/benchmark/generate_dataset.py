import pandas as pd
import numpy as np
import random
import string

def create_new_hdf5_file(file_path, new_path):
    """ Create a new HDF5 file with random data. """
    # Load the dataset
    data = pd.read_hdf(file_path, key='data')

    columns = [] # TODO select the column we want to keep
    selected_data = data[columns]

    # Generate new data for each column
    new_data = pd.DataFrame(index=selected_data.index)

    for column in selected_data.columns:
        if column == 'B/P':
            mean = selected_data[column].mean().values[0]
            std = selected_data[column].std().values[0]
        else:
            mean = selected_data[column].mean()
            std = selected_data[column].std()
        new_data[column] = np.random.normal(mean, std, size=selected_data.shape[0])

    # Save the new dataset
    new_data.to_hdf(new_path, key='data', mode='w')

    print("New dataset created and saved successfully!")

def change_head(path):
    data = pd.read_hdf(path, key='data')
    columns = [
        'B/P', 'ROE',
        'TotalCapitalStock', 'TradableACapital', 'TotalMarketValue', 'TradableMarketValue', 'StockPrice', 
        'E/P', 'ECut/P', 'EBIT/EV', 'EBITDA/EV', 'ROA_Q', 'MACrossover', 'QuarterlyUnrestrictedShareholdersRatioChange', 
        'HSZZ_ALPHA_3M', 'ROE_Q', 'ROA_TTM', 'S_ROAG', 'ExternalFinancingScale_2Y', 'ConMarketConf_5D', 
        'NetProfitSequentialQuarterlyChange', 'SemiannualUnrestrictedShareholdersRatioChange', 'STD_12M', 
        'LarSmaDiffSellValue', 'OperatingCashFlowRatio', 'TurnoverRate_30D', 'HSZZ_R2_3M', 'Sales_Growth_3Y', 
        'PricePosition_30D', 'NetProfitMargin', 'OperatingProfitYOY', 'SalesToCashRatio', 
        'FutureUnrestrictedRatio_3M', 'HSZZ_ALPHA_12M', 'Idiosyncrasy', 'RatingChange', 'TSKEW', 
        'WeeklyConsensusChangeJR_1W', 'HSZZ_BETA_3M', 'PricePosition_180D', 'MedSellValue', 
        'UnlimitedShareholdersAverageAmount', 'T_ROEG', 'QuarterlyAverageShareholdersRatioChange', 
        'FixedAssetTurnover', 'MonthlyRatingChange_1M', 'FutureUnrestrictedRatio_6M', 'TurnoverRate_30D_90D', 
        'Sales2EV', 'ILLIQ_1M', 'Profit_Growth_TTM', 'HighLow_1M', 'OperationCash_TTM', 
        'FutureUnrestrictedRatio_6MOver30DAvgTurnover', 'TurnoverRate_30D_180D', 'GrossProfitMargin', 
        'AnalystMomentumScore', 'ShareExpansionRatio_2Y', 'ROIC', 'TurnoverRate_60D', 
        'ExternalFinancingAdjustedGrowthRate', 'Weighted_Strength_3M', 'Weighted_Strength_1M', 
        'FutureUnrestrictedRatio_1MOver30DAvgTurnover', 'OperatingCashFlowOverRevenue_TTM', 'ConMarketConf_10D', 
        'HSZZ_ResidualStd_3M', 'RevenueSequentialYOY', 'RevenueYOY', 'EXTE'
    ]

    data.columns = columns
    data.to_hdf(path, key='data', mode='w')
    print("Head changed successfully!")

def view_hdf5_file(filename):
    with pd.HDFStore(filename, 'r') as store:
        print("Keys in the file:", store.keys())
        data = store['data']
        print(data.head())
        print("\nSummary statistics:\n", data.describe())  
        print(data.index)



if __name__ == '__main__':
    path = ''
    new_path = ''
    create_new_hdf5_file(file_path=path, new_path=new_path)
    change_head(new_path)
    view_hdf5_file(new_path)