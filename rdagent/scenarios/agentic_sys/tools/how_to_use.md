from pathlib import Path
from rdagent.scenarios.agentic_sys.tools.web_search import create_web_search_tool

# Initialize
config_path = Path(__file__).parent / "tools" / "search_config.yaml"
search_tool = create_web_search_tool(config_path)

# Search for hypothesis
results = search_tool.search_for_hypothesis(
    task_description="Improve agentic system",
    current_gaps=["information gathering"],
    context={'weak_dimension': 'comprehensiveness'}
)

# Process results
for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Relevance: {result['relevance']}")