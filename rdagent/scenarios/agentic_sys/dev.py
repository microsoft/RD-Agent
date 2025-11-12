from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.log import rdagent_logger as logger
from pathlib import Path

# TODO:  We only list the dummy coder and runner here.
# If we want to implement the a comprehensive agentic system R&D Agent, we need to implement it with CoSTEER.


class AgenticSysCoder(Developer[Experiment]):

    def __init__(self, scen):
        self.scen = scen

    def develop(self, exp: Experiment) -> Experiment:
        # TODO: implement the coder
        '''
        generate code based on experiment assumption
        '''
        logger.info("Starting code generation for the experiment")

        try:
            #acquire workspace 
            ws_path = self._get_workspace_path(exp)
            ws_path.mkdir(parents=True, exist_ok=True)
            #generate code
            self._generate_files(ws_path, exp)
            logger.info(f"Code generation as workspace at {ws_path}")

        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            exp.exception = e
        
        return exp

    def _get_workspace_path(self, exp: Experiment):
        '''
        Get workspace path for the experiment
        '''
        if hasattr(exp, 'experiment_workspace') and exp.experiment_workspace:
            return Path(exp.experiment_workspace.workspace_path)
        
        base = Path("./workspace")
        base.mkdir(exist_ok=True)

    def _generate_files(self, ws_path, exp):
        '''
        Generate necessary files for the agentic system experiment
        '''
        # Dummy agent code
        (ws_path / "agent.py").write_text(
            self._get_agent_template(exp)
        )

        # train.py (execute entry point)
        (ws_path / "train.py").write_text(
            self._get_train_template()
        )

        #requirements.txt
        (ws_path / "requirements.txt").write_text(
            "#Add dependencies here\n"
        )

    def _get_agent_template(self, exp):
        "generate agent code template"
        hypothesis = getattr(exp, 'hypothesis', 'Improve system performance')
        return f'''"""
        Agentic System Implementation
        Hypothesis: {hypothesis}
        """
        import time
        from typing import Dict, List, Any

        class AgenticSystem:
            """Agentic System for task execution"""
            def __init__(self):
                self.name = "AgenticSystem"
                self.task_count = 0
            def run_tasks(self, tasks: Dict[str, Any]) -> Dict[str, Any]:
                """Run tasks and return results"""
                start_time = time.time()
                try: 
                    task_id = task.get('id', self.task_count)
                    self.task_count += 1
                    result = {{
                        "task_id": task_id,
                        "status": "True",
                        "time": time.time() - start_time,
                        "error": None
                    }}
                except Exception as e:
                    result = {{
                        "task_id": task_id,
                        "status": False,
                        "time": time.time() - start_time,
                        "error": str(e)
                    }}
                return result
            def run_tasks(self, tasks: List[Dict] = None): 
                """Run multiple tasks and collect results"""
                if tasks is None:
                    tasks = [
                         {{"id": i, "type": "test", "data": f"sample{{i}}"}}
                        for i in range(10)
                    ]
                results = []
                for task in tasks: 
                    result.append(self.run_task(task))
                # Calculate metrics
                success_count = sum(1 for r in results if r["success"])
                total_time = sum(r["time"] for r in results)
                error_count = sum(1 for r in results if r["error"])

                return {{
                    "success_rate": success_count / len(results) if result else 0,
                    "avg_time": total_time / len(results) if results else 0,
                    "error_count": error_count,
                    "total_tasks": len(results)
                }}
        '''
    
    def _get_train_template(self):
        """generate train.py template"""
        return '''"""
        Training/Execution script for Agentic System, this is the entry point 
        that will be executed by the runner.
        """
        import json
        import sys
        from pathlib import Path
        from agent import AgenticSystem
        def main():
            """Main execution function"""
            try:
                print("Starting Agentic System execution...")
                # Initialize agent
                agent = AgenticSystem()
                # Run tasks
                results = agent.run_tasks()
                #Print for logging
                print("execution completed")
                print(f"Success Rate: {results['success_rate']}")
                print(f"Average Time: {results['avg_time']}")
                print(f"Error Count: {results['error_count']}")
                print(f"Total Tasks: {results['total_tasks']}")

            except Exception as e:
                print(f"Execution failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if __name__ == "__main__":
            main()
        '''
        

class AgenticSysRunner(Developer[Experiment]):

    def __init__(self, scen):
        self.scen = scen

    def develop(self, exp: Experiment) -> Experiment:
        # TODO: implement the runner
        """
        execute the experiment
        steps: 
        1. acquire workspace
        2. execute train.py
        3. load result file
        """
        pass
