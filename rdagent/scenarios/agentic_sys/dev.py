import sys
from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.log import rdagent_logger as logger
from pathlib import Path

# TODO:  We only list the dummy coder and runner here.
# If we want to implement the a comprehensive agentic system R&D Agent, we need to implement it with CoSTEER.


class AgenticSysCoder(Developer[Experiment]):
    #generate code for agentic system experiment
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
            ws_path = self.get_workspace_path(exp)
            ws_path.mkdir(parents=True, exist_ok=True)
            #generate code
            self.generate_files(ws_path, exp)
            logger.info(f"Code generation as workspace at {ws_path}")

        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            exp.exception = e
        
        return exp

    def get_workspace_path(self, exp: Experiment):
        '''
        Get workspace path for the experiment
        '''
        if hasattr(exp, 'experiment_workspace') and exp.experiment_workspace:
            return Path(exp.experiment_workspace.workspace_path)
        
        base = Path("./workspace")
        base.mkdir(exist_ok=True)
        return base / f"exp_{exp.id}"

    def generate_files(self, ws_path, exp):
        '''
        Generate necessary files for the agentic system experiment
        '''
        # Dummy agent code
        (ws_path / "agent.py").write_text(
            self.get_agent_template(exp)
        )

        # train.py (execute entry point)
        (ws_path / "train.py").write_text(
            self.get_train_template()
        )

        #requirements.txt
        (ws_path / "requirements.txt").write_text(
            "#Add dependencies here\n"
        )

    def get_agent_template(self, exp):
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
                        "success": "True",
                        "time": time.time() - start_time,
                        "error": None
                    }}
                except Exception as e:
                    result = {{
                        "task_id": task_id,
                        "success": False,
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
    
    def get_train_template(self):
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
    """execute code generated by AgenticSysCoder"""

    def __init__(self, scen):
        self.scen = scen

    def develop(self, exp: Experiment) -> Experiment:
        # TODO: implement the runner
        """
        execute the experiment
        steps: 
        1. acquire workspace
        2. execute test.py
        3. parse output
        4. collect performance metrics
        5. record logs
        """
        logger.info("Starting experiment execution")
        try: 
            # acquire workspace
            ws_path = self.get_workspace_path(exp)
            logger.info(f"Using workspace at {ws_path}")
            # validate necessary files
            self.validate_workspace(ws_path)
            #execute experiment
            stdout, stderr = self.execute_experiment(ws_path)
            #parse result
            result = self.parse_execution_output(stdout, stderr)
            exp.result = result
            # record execution logs
            self._log_execution_results(exp, result)
            logger.info("Experiment completed successfully")
        except Exception as e:
            logger.error(f"Experiment execution failed: {str(e)}")
            exp.exception = e
            exp.result = self._create_error_result(str(e))
        return exp
    
    def get_workspace_path(self, exp):
        '''
        Get workspace path for the experiment
        '''
        if hasattr(exp, 'experiment_workspace') and exp.experiment_workspace:
            return Path(exp.experiment_workspace.workspace_path)
        # Default workspace path
        return Path("./workspace") / f"exp_{exp.id}"
    
    def validate_workspace(self, ws_path: Path):
        """Validate necessary files in the workspace"""
        if not ws_path.exists():
            raise FileNotFoundError(f"Workspace path {ws_path} does not exist.")
        
        # examine necessary files
        required_files = ["train.py", "agent.py"]
        missing_files = []

        for file_name in required_files:
            file_path = ws_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files in workspace {ws_path}: {', '.join(missing_files)}")
        
        logger.info("workspace validation passed: {ws_path}")

    def execute_experiment(self, ws_path: Path, timeout: int = 300):
        """Execute the experiment by running train.py"""
        import subprocess

        cmd = [sys.executable, "train.py"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ws_path
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError(f"Experiment execution timed out after {timeout} seconds.")
        except Exception as e:
            logger.error(f"Execution failed with exception: {str(e)}")
            raise RuntimeError(f"Execution error: {str(e)}")
        
    def prepare_environment(self):
        """Prepare execution environment"""
        import os
        env = os.environ.copy()
        # Add any necessary environment variables here
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{os.getcwd()}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = os.getcwd()
        return env
    
    def parse_execution_output(self, stdout: str, stderr: str):
        """Parse execution output to extract performance metrics"""
        try:
            #method1, lookup structured output
            result = self.parse_structured_output(stdout)
            if result:
                return result
            
            #method2, lookup result file
            result = self.parse_result_file()
            if result:
                return result
            
            #method3, parse from stdout
            result = self.parse_text_output(stdout)
            if result:
                return result
            
            logger.warning("Could not parse execution output, using default result ")
            return self.create_default_result(
                success=False,
                reason = "Could not parse output"
            )
        
        except Exception as e:
            logger.error(f"Failed to parse output: {e}")
            return self._create_error_result(f"Parsing error: {e}")
    
    def parse_structured_output(self, stdout:str):
        """Parse structured JSON output """
        try:
            import json
            import re
            # Look for JSON blocks in stdout
            json_pattern = r'\{[^{}]*"success_rate"[^{}]*\}'
            matches = re.findall(json_pattern, stdout, re.DOTALL)
            
            for match in matches:
                try:
                    result = json.loads(match)
                    # Validate result format
                    if self.validate_result_format(result):
                        logger.info("Successfully parsed structured output")
                        return result
                except json.JSONDecodeError:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse structured output: {e}")
            return None


    def parse_text_output(self, stdout: str):
        """Parse text output using regex"""
        try:
            import re
            
            # Extract metrics using regex
            success_match = re.search(r'Success Rate:\s*([0-9.]+)', stdout, re.IGNORECASE)
            time_match = re.search(r'Average Time:\s*([0-9.]+)', stdout, re.IGNORECASE)
            error_match = re.search(r'Error Count:\s*([0-9]+)', stdout, re.IGNORECASE)
            task_match = re.search(r'Total Tasks:\s*([0-9]+)', stdout, re.IGNORECASE)
            
            if success_match:
                result = {
                    "success_rate": float(success_match.group(1)),
                    "avg_time": float(time_match.group(1)) if time_match else 0.0,
                    "error_count": int(error_match.group(1)) if error_match else 0,
                    "total_tasks": int(task_match.group(1)) if task_match else 0
                }
                logger.info("Successfully parsed text output")
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse text output: {e}")
            return None

    def parse_result_file(self):
        """Parse result from JSON file"""
        try:
            import json
            
            possible_paths = ["result.json", "output.json", "results.json"]
            
            for file_name in possible_paths:
                file_path = Path(file_name)
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8')
                    result = json.loads(content)
                    
                    if self._validate_result_format(result):
                        logger.info(f"Successfully parsed result file: {file_path}")
                        return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse result file: {e}")
            return None

    def validate_result_format(self, result: dict) -> bool:
        """Validate result format"""
        required_fields = ["success_rate", "avg_time", "error_count"]
        
        for field in required_fields:
            if field not in result:
                return False
            if not isinstance(result[field], (int, float)):
                return False
        
        # Check value ranges
        if not (0.0 <= result["success_rate"] <= 1.0):
            return False
        if result["avg_time"] < 0:
            return False
        if result["error_count"] < 0:
            return False
        
        return True

    def create_default_result(self, success: bool = False, reason: str = "") -> dict:  
        """Create default result"""
        return {
            "success_rate": 1.0 if success else 0.0,
            "avg_time": 0.0 if success else float('inf'),
            "error_count": 0 if success else 1,
            "total_tasks": 0,
            "error_reason": reason
        }

    def create_error_result(self, error_message: str) -> dict:  
        """Create error result"""
        return {
            "success_rate": 0.0,
            "avg_time": float('inf'),
            "error_count": 1,
            "total_tasks": 0,
            "error_reason": error_message
        }

    def log_execution_results(self, exp: Experiment, result: dict): 
        """Log execution results"""
        logger.info("=" * 50)
        logger.info("EXECUTION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Success Rate: {result.get('success_rate', 0):.2%}")
        logger.info(f"Average Time: {result.get('avg_time', 0):.4f}s")
        logger.info(f"Error Count: {result.get('error_count', 0)}")
        logger.info(f"Total Tasks: {result.get('total_tasks', 0)}")
        
        if 'error_reason' in result:
            logger.warning(f"Error: {result['error_reason']}")
        
        logger.info("=" * 50)



            






        
