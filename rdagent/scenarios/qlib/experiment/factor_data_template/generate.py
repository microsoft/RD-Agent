import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
# Debug slice: pick instruments that actually trade in the debug window.
# (Taking the first 100 of the full-history panel breaks on universes with
# delisted securities: names dead before 2018 are absent from this window.)
data_debug = (
    D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
    .sort_index()
)
keep = data_debug.index.get_level_values("instrument").unique()[:100]
data_debug = data_debug.loc[keep].swaplevel().sort_index()

data_debug.to_hdf("./daily_pv_debug.h5", key="data")
