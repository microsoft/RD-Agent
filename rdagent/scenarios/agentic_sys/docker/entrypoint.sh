

mkdir -p /env
cd /env
uv sync
uv add pip
source .venv/bin/activate

mkdir -p /workspace
cd /workspace

git clone https://github.com/your-username/deep_research_bench.git
cd deep_research_bench
pip install -r requirements.txt
