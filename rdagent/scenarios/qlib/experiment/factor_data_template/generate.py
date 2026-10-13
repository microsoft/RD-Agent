import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
sub = (
    D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
    .swaplevel()
    .sort_index()
    .swaplevel()
)
# Some instruments in the all-time data were delisted before 2018 and are absent from the subset.
# Filter before capping so delisted names don't shrink the debug file below 100 instruments.
present = set(sub.index.get_level_values("instrument"))
target = [i for i in data.index.get_level_values("instrument").unique() if i in present][:100]
data = sub.loc[target].swaplevel().sort_index()

data.to_hdf("./daily_pv_debug.h5", key="data")
