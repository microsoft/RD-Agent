from pathlib import Path
from rdagent.core.experiment import Experiment

# convert code into executable experiment and output standard experiment result
class AgenticSysExperiment(Experiment):
    def __init__(self, sub_tasks=None, based_experiments=None, experiment_workspace=None):
        super().__init__(sub_tasks=sub_tasks, based_experiments=based_experiments)
        if experiment_workspace is not None:
            self.experiment_workspace = experiment_workspace

    def run(self, code:str):
        """
        Run the experiment with the given code.
        Step: 
        1. Prepare Experiment Environment
        2. Run Agent code
        3. Collect Performance Metrics
        4. record log
        """
        code_path = self.workspace / "agent.py"
        code_path.write_text(code)

        #construct running script
        run_script = f"""
        import sys
        import json
        try:
            sys.path.insert(0, '{self.workspace}')
            from agent import AgenticSystem
            
            agent = AgenticSystem()
            results = agent.run_tasks()
            
            # output structured results
            print("=== EXECUTION RESULTS ===")
            print(f"Success Rate: {{results['success_rate']}}")
            print(f"Average Time: {{results['avg_time']}}")
            print(f"Error Count: {{results['error_count']}}")
            print(f"Total Tasks: {{results['total_tasks']}}")
            print("=== END RESULTS ===")
            
            # output JSON format for parsing
            print("=== JSON RESULTS ===")
            print(json.dumps(results))
            print("=== END JSON ===")
            
        except Exception as e:
            print(f"ERROR: {{str(e)}}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        """

        # use runner to execute
        result = self.runner.run(
            script = run_script,
            timeout = 300,  # 5 minutes timeout
            capture_output = True
        )

        # parse output for metrics
        metrics = self.parse_output(result.stdout)
        return Experiment(
            success = result.returncode == 0,
            metrics = metrics,
            logs = result.stdout,
            errors = result.stderr
        )
    
    #parse all the metrics from stdout
    def parse_output(self, stdout: str):
        metrics = {
            "success_rate": 0.0,
            "avg_time": float('inf'),
            "error_count": 1,
            "total_tasks": 0
        }
        
        try:
            # try to extract JSON block first
            import json
            import re
            json_match = re.search(r'=== JSON RESULTS ===\n(.*?)\n=== END JSON ===', stdout, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group(1))
                metrics.update(result_data)
                return metrics
            
            # fallback to text parsing
            for line in stdout.splitlines():
                if "Success Rate:" in line:
                    metrics["success_rate"] = float(line.split(":")[1].strip())
                elif "Average Time:" in line:
                    metrics["avg_time"] = float(line.split(":")[1].strip())
                elif "Error Count:" in line:
                    metrics["error_count"] = int(line.split(":")[1].strip())
                elif "Total Tasks:" in line:
                    metrics["total_tasks"] = int(line.split(":")[1].strip())
                    
        except Exception as e:
            print(f"Failed to parse output: {e}")
            
        return metrics
