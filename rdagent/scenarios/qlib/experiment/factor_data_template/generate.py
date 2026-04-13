# import qlib
# from rdagent.utils.qlib import get_qlib_data_path

# qlib.init(provider_uri=str(get_qlib_data_path()))

# from qlib.data import D

# instruments = D.instruments()
# fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
# data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

# data.to_hdf("./daily_pv_all.h5", key="data")


# fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
# data = (
#     (
#         D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
#         .swaplevel()
#         .sort_index()
#     )
#     .swaplevel()
#     .loc[data.reset_index()["instrument"].unique()[:100]]
#     .swaplevel()
#     .sort_index()
# )

# data.to_hdf("./daily_pv_debug.h5", key="data")


import os
import qlib
import pandas as pd
from rdagent.utils.qlib import get_qlib_data_path
from qlib.data import D

# =========================
# Init
# =========================
qlib.init(provider_uri=str(get_qlib_data_path()))

# =========================
# Config
# =========================
FIELDS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]

START_DATE_ALL = "2008-12-29"

DEBUG_START = "2018-01-01"
DEBUG_END = "2019-12-31"
DEBUG_STOCKS = 100

SAVE_ALL = "./daily_pv_all.h5"
SAVE_DEBUG = "./daily_pv_debug.h5"

DTYPE = "float32"


# =========================
# Utils
# =========================
def fetch_data(instruments, fields, start_time=None, end_time=None):
    df = D.features(
        instruments,
        fields,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    return df.swaplevel().sort_index()


def save_hdf(df, path):
    df.to_hdf(path, key="data")
    print(f"[✓] Saved: {path} | shape={df.shape}")


# =========================
# Instruments
# =========================
instruments = D.instruments()


# =========================
# Full dataset (with cache)
# =========================
if os.path.exists(SAVE_ALL):
    print("[✓] Using cached full dataset")
    data_all = pd.read_hdf(SAVE_ALL, key="data")
else:
    data_all = fetch_data(
        instruments,
        FIELDS,
        start_time=START_DATE_ALL,
    ).astype(DTYPE)

    save_hdf(data_all, SAVE_ALL)


# =========================
# Debug dataset
# =========================
if os.path.exists(SAVE_DEBUG):
    print("[✓] Using cached debug dataset")
    data_debug = pd.read_hdf(SAVE_DEBUG, key="data")
else:
    data_debug = fetch_data(
        instruments,
        FIELDS,
        start_time=DEBUG_START,
        end_time=DEBUG_END,
    ).astype(DTYPE)

    stock_list = (
        data_debug.index.get_level_values("instrument")
        .unique()[:DEBUG_STOCKS]
    )

    data_debug = data_debug.loc[stock_list]

    save_hdf(data_debug, SAVE_DEBUG)