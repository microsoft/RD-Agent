import qlib

from rdagent.utils.qlib import get_qlib_data_path

qlib.init(provider_uri=get_qlib_data_path())

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = (
    (
        D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

data.to_hdf("./daily_pv_debug.h5", key="data")
