"""
DeepResearch Bench Dataset Loader for Agentic System
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ResearchTask:
    """
    Research task from DeepResearch Bench
    """
    task_id: str
    title: str
    description: str
    domain: str
    difficulty: str
    evaluation_metrics: Dict[str, Any]
    input_data: Optional[Dict] = None
    expected_output: Optional[Dict] = None
    metadata: Optional[Dict] = None

class DeepResearchBenchLoader:
    """
    Load and manage DeepResearch Bench loader
    """
    def __init__(self, data_path, cache_dir):
        """
        Initialize DeepResearch Bench Loader

        Args:
            data_path: Path to local dataset (if already download)
            cache_dir: Directory to cache downloaded data
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents = True, exist_ok = True)
        self.tasks: List[ResearchTask] = []

    def load_dataset(self, subset:str):
        """
        Load DeepResearch Bench dataset
        Args:
            subset: Dataset subset to load (e.g., 'easy', 'medium', 'hard')
        """
        logger.info(f"Loading DeepResearch Bench dataset (subset={subset})")
        if self.data_path and self.data_path.exists():
            #load from local path
            self.tasks = self.load_from_local(self.data_path, subset)
        else:
            #download from remote
            self.tasks = self.download_and_load(subset)
        logger.info(f"Loaded {len(self.tasks)} tasks from DeepResearch Bench (subset={subset})")
        return self.tasks
    
    def load_from_local(self, data_path: Path, subset: str):
        "load dataset from local path"
        tasks = []
        #assume JSON format
        json_files = list(data_path.glob(".json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                #Parse task data
                if isinstance(data,list):
                    for item in data:
                        task = self.parse_task(item)
                        if self.matches_subset(task, subset):
                            tasks.append(task)
                else:
                    task = self.parse_task(data)
                    if self.matches_subset(task, subset):
                        tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to load task from {json_file}: {e}")
        return tasks
    
    def download_and_load(self, subset):
        "download dataset from DeepResearch Bench and load"
        base_url = "https://github.com/Ayanami0730/deep_research_bench"

        tasks = []
        cache_file = self.cache_dir / f"tasks_{subset}.json"

        #Check cache first
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return [self.parse_task(item) for item in cached_data]
        try: 
            #Download task list
            logger.info(f"Downloading tasks from {base_url}")
            response = requests.get(f"{base_url}/tree/main/data/{subset}_data/{subset}.jsonl",timeout=30)
            response.raise_for_status()
            data = response.json()
            tasks_data = data.get('tasks', [])

            #Parse tasks
            for item in tasks_data:
                task = self.parse_task(item)
                tasks.append(task)

            #Cache downloaded data
            with open(cache_file, 'w') as f:
                json.dump(tasks_data, f)

            logger.info(f"Downloaded and cached {len(tasks)} tasks")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            tasks = self.create_mock_tasks(subset)

        return tasks
    
    def parse_task(self, data: Dict) -> ResearchTask:
        """Parse task data into ResearchTask object"""
        return ResearchTask(
            task_id=data.get('id', 'unknown'),
            title=data.get('title', ''),
            description=data.get('description', ''),
            domain=data.get('domain', 'general'),
            difficulty=data.get('difficulty', 'medium'),
            evaluation_criteria=data.get('evaluation_criteria', {}),
            input_data=data.get('input_data'),
            expected_output=data.get('expected_output'),
            metadata=data.get('metadata', {})
        )
    
    def matches_subset(self, task: ResearchTask, subset: str) -> bool:
        """Check if task matches requested subset"""
        if subset == "all":
            return True
        return task.difficulty.lower() == subset.lower()
    
    def create_mock_tasks(self, subset: str) -> List[ResearchTask]:
        """Create mock tasks for testing when download fails"""
        logger.warning("Creating mock tasks for testing")
        
        mock_tasks = [
            {
                'id': 'mock_001',
                'title': 'Literature Review Synthesis',
                'description': 'Synthesize findings from multiple research papers',
                'domain': 'research_synthesis',
                'difficulty': 'medium',
                'evaluation_criteria': {
                    'completeness': 0.3,
                    'coherence': 0.3,
                    'accuracy': 0.4
                },
                'input_data': {
                    'papers': ['paper1.pdf', 'paper2.pdf', 'paper3.pdf'],
                    'query': 'What are the main findings on topic X?'
                }
            },
            {
                'id': 'mock_002',
                'title': 'Hypothesis Generation',
                'description': 'Generate research hypotheses based on existing literature',
                'domain': 'hypothesis_generation',
                'difficulty': 'hard',
                'evaluation_criteria': {
                    'novelty': 0.4,
                    'feasibility': 0.3,
                    'clarity': 0.3
                }
            },
            {
                'id': 'mock_003',
                'title': 'Experiment Design',
                'description': 'Design an experiment to test a given hypothesis',
                'domain': 'experiment_design',
                'difficulty': 'easy',
                'evaluation_criteria': {
                    'validity': 0.4,
                    'completeness': 0.3,
                    'practicality': 0.3
                }
            }
        ]
        
        return [self._parse_task(task) for task in mock_tasks 
                if self._matches_subset(self._parse_task(task), subset)]
    
    def get_task_by_id(self, task_id: str) -> Optional[ResearchTask]:
        """Get specific task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_tasks_by_domain(self, domain: str):
        """Get tasks filtered by domain"""
        return [task for task in self.tasks if task.domain == domain]
    
    def get_tasks_by_difficulty(self, difficulty: str):
        """Get tasks filtered by difficulty"""
        return [task for task in self.tasks 
                if task.difficulty.lower() == difficulty.lower()]


                


