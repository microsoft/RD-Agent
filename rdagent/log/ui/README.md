# start web app

in `RD-Agent/` folder, run `streamlit run rdagent/log/ui/app.py --server.port 14000`

## custom windows

in `rdagent.log.ui.web.py`

StWindow is the base window, can show common log messages like logger in terminal.

Windows can be nested.

Simply, just write a window for one log object type, and use `isinstance()` to display messages in different windows.

### some base windows

- StWindow
- LLMWindow
- CodeWindow

### multi tabs window

More convenient to use nested windows

- ProgressTabsWindow
- ObjectsTabsWindow

### main trace window

- QlibFactorTraceWindow

## TODOS

- make it like a living trace
- display real living trace
- Window Styles
