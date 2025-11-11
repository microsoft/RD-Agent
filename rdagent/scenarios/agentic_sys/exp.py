


from anyio import Path
from rdagent.core.experiment import Experiment
from rdagent.components.runner import LocalRunner
from rdagent.core.experiment import ExperimentResult


class AgenticSysExperiment(Experiment):
    def __init__(self,workspace:Path):
        self.workspace = workspace
        self.runner = LocalRunner(workspace)

    def run(self, code:str) -> ExperimentResult:
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
        run_script = f""""
        import sys
        sys.path.insert(0, '{self.workspace}')
        from agent import AgenticSystem

        #initialize agent
        agent = AgenticSystem()

        #run task
        results = agent.run_tasks()

        #output results
        print(f"Success Rate: {{results['success_rate']}}")
        print(f"Average Time: {{results['avg_time']}}")
        """

        # use runner to execute
        result = self.runner.run(
            script = run_script,
            timeout = 300,  # 5 minutes timeout
            capture_output = True
        )

        # parse output for metrics
        metrics = self._parse_output(result.stdout)
        return ExperimentResult(
            success = result.returncode == 0,
            metrics = metrics,
            logs = result.stdout,
            errors = result.stderr
        )
    
    def _parse_output(self, stdout:str):
        metrics = {}
        for line in stdout.splitlines():
            if "Success Rate:" in line:
                metrics["success_rate"] = float(line.split(":")[1].strip())
            elif "Average Time:" in line:
                metrics["avg_time"] = float(line.split(":")[1].strip())
        return metrics
