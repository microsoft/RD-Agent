```
export PYTHONPATH=$(pwd):PYTHONPATH

dotenv run -- python rdagent/app/data_science/loop.py --competition tabular-playground-series-dec-2021

# aerial-cactus-identification
# spooky-author-identification 
# tabular-playground-series-dec-2021

streamlit run rdagent/log/ui/llm_st.py --server.port=10883 -- --log_dir ./log

dotenv run -- python rdagent/log/mle_summary.py grade_summary --log_folder=./log

dotenv run -- python scripts/exp/researcher/utils.py check --log_trace_path=$dir

dotenv run -- streamlit run rdagent/log/ui/dsapp.py

ls -lat /home/xiaoyang/repos/batch_ctrl/all_projects

ls -lat /home/xiaoyang/repos/batch_ctrl/all_projects/ep*/log.bak/log_202502111648/

cd /home/xiaoyang/repos/batch_ctrl/all_projects
```