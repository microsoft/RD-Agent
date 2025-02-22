import importlib
import math

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class BaseGraph:
    _name = None

    def __init__(
        self, df: pd.DataFrame = None, layout: dict = None, graph_kwargs: dict = None, name_dict: dict = None, **kwargs
    ):
        """

        :param df:
        :param layout:
        :param graph_kwargs:
        :param name_dict:
        :param kwargs:
            layout: dict
                go.Layout parameters
            graph_kwargs: dict
                Graph parameters, eg: go.Bar(**graph_kwargs)
        """
        self._df = df

        self._layout = dict() if layout is None else layout
        self._graph_kwargs = dict() if graph_kwargs is None else graph_kwargs
        self._name_dict = name_dict

        self.data = None

        self._init_parameters(**kwargs)
        self._init_data()

    def _init_data(self):
        """

        :return:
        """
        if self._df.empty:
            raise ValueError("df is empty.")

        self.data = self._get_data()

    def _init_parameters(self, **kwargs):
        """

        :param kwargs
        """

        # Instantiate graphics parameters
        self._graph_type = self._name.lower().capitalize()

        # Displayed column name
        if self._name_dict is None:
            self._name_dict = {_item: _item for _item in self._df.columns}

    @staticmethod
    def get_instance_with_graph_parameters(graph_type: str = None, **kwargs):
        """

        :param graph_type:
        :param kwargs:
        :return:
        """
        try:
            _graph_module = importlib.import_module("plotly.graph_objs")
            _graph_class = getattr(_graph_module, graph_type)
        except AttributeError:
            _graph_module = importlib.import_module("qlib.contrib.report.graph")
            _graph_class = getattr(_graph_module, graph_type)
        return _graph_class(**kwargs)

    def _get_layout(self) -> go.Layout:
        """

        :return:
        """
        return go.Layout(**self._layout)

    def _get_data(self) -> list:
        """

        :return:
        """

        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type, x=self._df.index, y=self._df[_col], name=_name, **self._graph_kwargs
            )
            for _col, _name in self._name_dict.items()
        ]
        return _data

    @property
    def figure(self) -> go.Figure:
        """

        :return:
        """
        _figure = go.Figure(data=self.data, layout=self._get_layout())
        # NOTE: Use the default theme from plotly version 3.x, template=None
        _figure["layout"].update(template=None)
        return _figure


class SubplotsGraph:
    """Create subplots same as df.plot(subplots=True)

    Simple package for `plotly.tools.subplots`
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        kind_map: dict = None,
        layout: dict = None,
        sub_graph_layout: dict = None,
        sub_graph_data: list = None,
        subplots_kwargs: dict = None,
        **kwargs,
    ):
        """

        :param df: pd.DataFrame

        :param kind_map: dict, subplots graph kind and kwargs
            eg: dict(kind='Scatter', kwargs=dict())

        :param layout: `go.Layout` parameters

        :param sub_graph_layout: Layout of each graphic, similar to 'layout'

        :param sub_graph_data: Instantiation parameters for each sub-graphic
            eg: [(column_name, instance_parameters), ]

            column_name: str or go.Figure

            Instance_parameters:

                - row: int, the row where the graph is located

                - col: int, the col where the graph is located

                - name: str, show name, default column_name in 'df'

                - kind: str, graph kind, default `kind` param, eg: bar, scatter, ...

                - graph_kwargs: dict, graph kwargs, default {}, used in `go.Bar(**graph_kwargs)`

        :param subplots_kwargs: `plotly.tools.make_subplots` original parameters

                - shared_xaxes: bool, default False

                - shared_yaxes: bool, default False

                - vertical_spacing: float, default 0.3 / rows

                - subplot_titles: list, default []
                    If `sub_graph_data` is None, will generate 'subplot_titles' according to `df.columns`,
                    this field will be discarded


                - specs: list, see `make_subplots` docs

                - rows: int, Number of rows in the subplot grid, default 1
                    If `sub_graph_data` is None, will generate 'rows' according to `df`, this field will be discarded

                - cols: int, Number of cols in the subplot grid, default 1
                    If `sub_graph_data` is None, will generate 'cols' according to `df`, this field will be discarded


        :param kwargs:

        """

        self._df = df
        self._layout = layout
        self._sub_graph_layout = sub_graph_layout

        self._kind_map = kind_map
        if self._kind_map is None:
            self._kind_map = dict(kind="Scatter", kwargs=dict())

        self._subplots_kwargs = subplots_kwargs
        if self._subplots_kwargs is None:
            self._init_subplots_kwargs()

        self.__cols = self._subplots_kwargs.get("cols", 2)  # pylint: disable=W0238
        self.__rows = self._subplots_kwargs.get(  # pylint: disable=W0238
            "rows", math.ceil(len(self._df.columns) / self.__cols)
        )

        self._sub_graph_data = sub_graph_data
        if self._sub_graph_data is None:
            self._init_sub_graph_data()

        self._init_figure()

    def _init_sub_graph_data(self):
        """

        :return:
        """
        self._sub_graph_data = []
        self._subplot_titles = []

        for i, column_name in enumerate(self._df.columns):
            row = math.ceil((i + 1) / self.__cols)
            _temp = (i + 1) % self.__cols
            col = _temp if _temp else self.__cols
            res_name = column_name.replace("_", " ")
            _temp_row_data = (
                column_name,
                dict(
                    row=row,
                    col=col,
                    name=res_name,
                    kind=self._kind_map["kind"],
                    graph_kwargs=self._kind_map["kwargs"],
                ),
            )
            self._sub_graph_data.append(_temp_row_data)
            self._subplot_titles.append(res_name)

    def _init_subplots_kwargs(self):
        """

        :return:
        """
        # Default cols, rows
        _cols = 2
        _rows = math.ceil(len(self._df.columns) / 2)
        self._subplots_kwargs = dict()
        self._subplots_kwargs["rows"] = _rows
        self._subplots_kwargs["cols"] = _cols
        self._subplots_kwargs["shared_xaxes"] = False
        self._subplots_kwargs["shared_yaxes"] = False
        self._subplots_kwargs["vertical_spacing"] = 0.3 / _rows
        self._subplots_kwargs["print_grid"] = False
        self._subplots_kwargs["subplot_titles"] = self._df.columns.tolist()

    def _init_figure(self):
        """

        :return:
        """
        self._figure = make_subplots(**self._subplots_kwargs)

        for column_name, column_map in self._sub_graph_data:
            if isinstance(column_name, go.Figure):
                _graph_obj = column_name
            elif isinstance(column_name, str):
                temp_name = column_map.get("name", column_name.replace("_", " "))
                kind = column_map.get("kind", self._kind_map.get("kind", "Scatter"))
                _graph_kwargs = column_map.get("graph_kwargs", self._kind_map.get("kwargs", {}))
                _graph_obj = BaseGraph.get_instance_with_graph_parameters(
                    kind,
                    **dict(
                        x=self._df.index,
                        y=self._df[column_name],
                        name=temp_name,
                        **_graph_kwargs,
                    ),
                )
            else:
                raise TypeError()

            row = column_map["row"]
            col = column_map["col"]

            self._figure.add_trace(_graph_obj, row=row, col=col)

        if self._sub_graph_layout is not None:
            for k, v in self._sub_graph_layout.items():
                self._figure["layout"][k].update(v)

        # NOTE: Use the default theme from plotly version 3.x: template=None
        self._figure["layout"].update(template=None)
        self._figure["layout"].update(self._layout)

    @property
    def figure(self):
        return self._figure


def _calculate_maximum(df: pd.DataFrame, is_ex: bool = False):
    """

    :param df:
    :param is_ex:
    :return:
    """
    if is_ex:
        end_date = df["cum_ex_return_wo_cost_mdd"].idxmin()
        start_date = df.loc[df.index <= end_date]["cum_ex_return_wo_cost"].idxmax()
    else:
        end_date = df["return_wo_mdd"].idxmin()
        start_date = df.loc[df.index <= end_date]["cum_return_wo_cost"].idxmax()
    return start_date, end_date


def _calculate_mdd(series):
    """
    Calculate mdd

    :param series:
    :return:
    """
    return series - series.cummax()


def _calculate_report_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df = raw_df.copy(deep=True)
    index_names = df.index.names
    df.index = df.index.strftime("%Y-%m-%d")

    report_df = pd.DataFrame()

    report_df["cum_bench"] = df["bench"].cumsum()
    report_df["cum_return_wo_cost"] = df["return"].cumsum()
    report_df["cum_return_w_cost"] = (df["return"] - df["cost"]).cumsum()
    # report_df['cum_return'] - report_df['cum_return'].cummax()
    report_df["return_wo_mdd"] = _calculate_mdd(report_df["cum_return_wo_cost"])
    report_df["return_w_cost_mdd"] = _calculate_mdd((df["return"] - df["cost"]).cumsum())

    report_df["cum_ex_return_wo_cost"] = (df["return"] - df["bench"]).cumsum()
    report_df["cum_ex_return_w_cost"] = (df["return"] - df["bench"] - df["cost"]).cumsum()
    report_df["cum_ex_return_wo_cost_mdd"] = _calculate_mdd((df["return"] - df["bench"]).cumsum())
    report_df["cum_ex_return_w_cost_mdd"] = _calculate_mdd((df["return"] - df["cost"] - df["bench"]).cumsum())
    # return_wo_mdd , return_w_cost_mdd,  cum_ex_return_wo_cost_mdd, cum_ex_return_w

    report_df["turnover"] = df["turnover"]
    report_df.sort_index(ascending=True, inplace=True)

    report_df.index.names = index_names
    return report_df


def report_figure(df: pd.DataFrame) -> list | tuple:
    """

    :param df:
    :return:
    """

    # Get data
    report_df = _calculate_report_data(df)

    # Maximum Drawdown
    max_start_date, max_end_date = _calculate_maximum(report_df)
    ex_max_start_date, ex_max_end_date = _calculate_maximum(report_df, True)

    index_name = report_df.index.name
    _temp_df = report_df.reset_index()
    _temp_df.loc[-1] = 0
    _temp_df = _temp_df.shift(1)
    _temp_df.loc[0, index_name] = "T0"
    _temp_df.set_index(index_name, inplace=True)
    _temp_df.iloc[0] = 0
    report_df = _temp_df

    # Create figure
    _default_kind_map = dict(kind="Scatter", kwargs={"mode": "lines+markers"})
    _temp_fill_args = {"fill": "tozeroy", "mode": "lines+markers"}
    _column_row_col_dict = [
        ("cum_bench", dict(row=1, col=1)),
        ("cum_return_wo_cost", dict(row=1, col=1)),
        ("cum_return_w_cost", dict(row=1, col=1)),
        ("return_wo_mdd", dict(row=2, col=1, graph_kwargs=_temp_fill_args)),
        ("return_w_cost_mdd", dict(row=3, col=1, graph_kwargs=_temp_fill_args)),
        ("cum_ex_return_wo_cost", dict(row=4, col=1)),
        ("cum_ex_return_w_cost", dict(row=4, col=1)),
        ("turnover", dict(row=5, col=1)),
        ("cum_ex_return_w_cost_mdd", dict(row=6, col=1, graph_kwargs=_temp_fill_args)),
        ("cum_ex_return_wo_cost_mdd", dict(row=7, col=1, graph_kwargs=_temp_fill_args)),
    ]

    _subplot_layout = dict()
    for i in range(1, 8):
        # yaxis
        _subplot_layout.update({"yaxis{}".format(i): dict(zeroline=True, showline=True, showticklabels=True)})
        _show_line = i == 7
        _subplot_layout.update({"xaxis{}".format(i): dict(showline=_show_line, type="category", tickangle=45)})

    _layout_style = dict(
        height=1200,
        title=" ",
        shapes=[
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": max_start_date,
                "y0": 0.55,
                "x1": max_end_date,
                "y1": 1,
                "fillcolor": "#d3d3d3",
                "opacity": 0.3,
                "line": {
                    "width": 0,
                },
            },
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": ex_max_start_date,
                "y0": 0,
                "x1": ex_max_end_date,
                "y1": 0.55,
                "fillcolor": "#d3d3d3",
                "opacity": 0.3,
                "line": {
                    "width": 0,
                },
            },
        ],
    )

    _subplot_kwargs = dict(
        shared_xaxes=True,
        vertical_spacing=0.01,
        rows=7,
        cols=1,
        row_width=[1, 1, 1, 3, 1, 1, 3],
        print_grid=False,
    )
    figure = SubplotsGraph(
        df=report_df,
        layout=_layout_style,
        sub_graph_data=_column_row_col_dict,
        subplots_kwargs=_subplot_kwargs,
        kind_map=_default_kind_map,
        sub_graph_layout=_subplot_layout,
    ).figure
    return figure
