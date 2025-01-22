```
export PYTHONPATH=$(pwd):PYTHONPATH

dotenv run -- python rdagent/app/data_science/loop.py --competition tabular-playground-series-dec-2021

streamlit run rdagent/log/ui/llm_st.py --server.port=10880 -- --log_dir ./log
```