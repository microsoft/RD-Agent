import sys
from matplotlib.style import context
from prefect import task
from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.log import rdagent_logger as logger
from pathlib import Path
import subprocess
import sys
import json
import re
import os
from typing import Dict, Any, List, Optional

from rdagent.scenarios.agentic_sys.env import get_agent_sys_env

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
            # 1. Initialize workspace with FBWorkspace 
            exp.experiment_workspace = FBWorkspace()
            ws_path = Path(exp.experiment_workspace.workspace_path)
            ws_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized workspace at {ws_path}")


            #2. Generate code files using CoSTEER approach
            code_artifacts = self.generate_code_with_costeer(exp)
            exp.experiment_workspace.inject_files(**code_artifacts)
            logger.info(f"Injected {len(code_artifacts)} files into workspace")

            #prepare execution environment following conf.py pattern
            timeout = self.calculate_timeout(exp)
            env = get_agent_sys_env(
                extra_volumes = {str(ws_path): "/workspace"},
                running_timeout_period = timeout,
                enable_cache=True
            )
            logger.info(f"Prepared execution environment")

            # 3) Optinal pre-run validation
            try: 
                if self.should_validate_generation(exp):
                    validation_result = self.validate_generated_code(env, ws_path)
                    if not getattr(validation_result, 'success', False):
                        logger.warning(f"Pre-run validation failed: {validation_result.message}")
            except Exception as e_val:
                logger.error(f"Validation step raised: {e_val} continuing...")

            #4. run the entrypoint inside environment (use train.py as entry)
            try: 
                logger.info("Running generated code inside validation")
                run_res = env.run(
                    entry = "bash",
                    cmd = "cd /workspace && python train.py", timeout = timeout
                )
                #collect run outputs
                exp.run_returncode = getattr(run_res, 'returncode', None)
                exp.run_stdout = getattr(run_res, 'stdout', getattr(run_res, 'logs', None))
                exp.run_stderr = getattr(run_res, 'stderr', None)
                logger.info(f"Run finished")
            except Exception as e_run:
                logger.error(f"Execution inside environment failed: {e_run}")
                #keep exception and let caller decide; still return exp with workspace
                exp.run_exception = e_run

        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            exp.exception = e
            if not hasattr(exp, 'experiment_workspace') or not exp.experiment_workspace:
                try:
                    exp.experiment_workspace = self.create_fallback_workspace(exp)
                except Exception as e_fallback:
                    pass
        return exp
    
    def generate_code_with_costeer(self, exp: Experiment) -> Dict[str, str]:
        """
        Generate code artifacts using CoSTEER approach
        """
        logger.info("Generating code using CoSTEER framework")
        hypothesis = getattr(exp, 'hypothesis', 'Improve agentic system performance')
        context = {
            'hypothesis': hypothesis,
            'scenario_desc': self.scen.get_scenario_all_desc(),
            'success_criteria': self.scen.get_success_criteria(),
        }
        # generate code artifacts
        code_artifacts = {}

        #1. generate main agent implementation
        agent_code = self.generate_agent_code(context)
        code_artifacts['agent.py'] = agent_code

        #2. Generate execution script
        train_code = self.generate_train_script(context)
        code_artifacts['train.py'] = train_code

        #3. Generate requirements file
        requirements = self.generate_requirements(context)
        code_artifacts['requirements.txt'] = requirements

        #4. Generate configuration file if needed
        if self.needs_config_file(context):
            config_code = self.generate_config_file(context)
            code_artifacts['config.py'] = config_code

        logger.info(f"Generated {len(code_artifacts)} code artifacts")
        return code_artifacts
    
    def prepare_execution_environment(self, exp: Experiment, ws_path: Path):
        """
        Prepare execution environment similar to DS CoSTEER approach
        """
        try:
            # Get environment configuration
            extra_volumes = {str(ws_path): "/workspace"}
            #Set timeout based on experiment complexity
            timeout = self.calculate_timeout(exp)
            #create environment using agent_sys specific configuration
            env = get_agent_sys_env(
                extra_volumes = extra_volumes,
                running_timeout_period = timeout, 
                enable_cache=True
            )
            logger.info("Prepared execution environment successfully")
            return env
        
        except Exception as e:
            logger.error(f"Failed to prepare execution environment: {str(e)}")
            raise

    def calculate_timeout(self, exp: Experiment) -> int:
        """
        Calculate appropriate timeout based on experiment characteristics
        """
        base_timeout = 300  # default 5 minutes
        #Adjust timeout based on hypothesis comnplexity
        hypothesis = getattr(exp, 'hypothesis', '')
        if 'parallel' in hypothesis.lower() or 'concurrent' in hypothesis.lower():
            return base_timeout * 2  #parallel tasks may need more time
        elif 'optimisation' in hypothesis.lower():
            return base_timeout * 4 #learning/optimization may need more time
        elif 'simple' in hypothesis.lower() or 'basic' in hypothesis.lower():
            return base_timeout  #simple tasks
        return base_timeout
    
    def should_validate_generation(self,exp: Experiment) -> bool:
        """
        Determine if we should validate generated code before proceeding
        """
        pass

    def validate_generated_code(self, env, ws_path: Path):
        """
        Validate generated code by running basic checks
        """
        class ValidationResult:
            def __init__(self,success, message):
                self.success = success
                self.message = message

        try:
            #Run basic syntax check
            check_cmd = "python -m py_compile agent.py && python -m py_compile train.py"
            result = env.run(
                entry_point = "bash",
                cmd = f'cd/workspace && {check_cmd}',
                timeout = 30
            )
            if result.returncode == 0: 
                return ValidationResult(True, "syntax validation passed")
            else:
                return ValidationResult(False, f"Syntax validation failed: {result.stderr}")
        
        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")
        
    def generate_agent_code(self,context):
        """
        Generate agent code based on context
        """
        hypothesis = context.get('hypothesis', 'Improve agentic system performance')
        #enhanced agent template with CoSTEER improvement
        return f'''
        """
        Agentic System Implementation - CoSTEER enhanced
        Hypothesis: {hypothesis}
        Generated with intelligent code generation
        """
        import time
        import logging
        import threading
        from typing import Dict, List, Any, Optional
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from dataclasses import dataclass
        from enum import Enum
        import json

        #Configurable logging
        logging.basicConfig(level = logging.INFO)
        logger = logging.getLogger("AgenticSystem")

        class TaskStatus(Enum):
            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"

        @dataclass
        class TaskResult:
            task_id: int
            success: bool
            execution_time: float
            error: Optional[str] = None
            data: Optional[Dict[str, Any]] = None
        
        class AgenticSystem:
            """
            Ehanced Agentic System with CoSTEER optimizations
            """
            def __init__(self, config[Dict] = None):
                self.name = "CoSTEER_AgenticSystem"
                self.task_count = 0
                self.config = config if config else self.get_default_config()

                #Performance Tracking
                self.performance_metrics = {{"total_tasks": 0,"successful_tasks": 0,"failed_tasks": 0,"total_execution_time": 0}}

                #thread safety
                self.lock = threading.Lock()

                logger.info(f"Initialized {{self.name}} with config: {{self.config}}")
            
            def get_default_config(self):
                """Get default configuration optimized for hypothesis"""
                return {{
                    "max_workers": 4,
                    "task_timeout": 60,
                    "enable_parallel": {'parallel' in hypothesis.lower()},
                    "enable_optimization": {'optimization' in hypothesis.lower()}
                }}

            def run_task(self, task: Dict[str, Any]):
                """Execute single task with enhanced error handling and monitoring"""
                start_time = time.time()
                task_id = task.get('id', self.get_next_task_id())
                try:
                    logger.info(f"Starting task {{task_id}}")
                    #Simulate intelligent task processing
                    self.process_task_logic(task)
                    execution_time = time.time() - start_time
                    #update metrics
                    with self.lock:
                        self.metrics['total_tasks'] += 1
                        self.metrics['successful_tasks'] += 1
                        self.metrics['total_execution_time'] += execution_time
                    result = TaskResult(
                        task_id = task_id,
                        status = TaskStatus.COMPLETED,
                        execution_time = execution_time,
                        success = True,
                        data = {{'processed': True, 'task_type': task.get('type', 'unknown')}}
                    )

                    logger.info(f"Task {{task_id}} completed successfully in {{execution_time:.4f}}s")
                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    with self.lock:
                        self.metrics['total_tasks'] += 1
                        self.metrics['failed_tasks'] += 1
                        self.metrics['total_execution_time'] += execution_time
                    result = TaskResult(
                        task_id = task_id,
                        status = TaskStatus.FAILED,
                        execution_time = execution_time,
                        success = False,
                        error = str(e)
                    )
                    logger.error(f"Task {{task_id}} failed: {{str(e)}}")
                    return result
            
            def get_next_task_id(self):
                "thread-safe task id generation"
                with self.lock:
                    self.task_count += 1
                    return self.task_count

            def process_task_logic(self, task):
                """Intelligent task processing based on hypothesis"""
                task_type = task.get('type', 'default')
                complexity = task.get('complexity', 1)

                #Simulate processing time based on complexity
                base_time = 0.01
                processing_time = base_time * complexity

                #Add hypothesis-specific optimisation
                if complexity > 5 and not self.config.get('enable_optimization', False):
                    # 10% error rate for high complexity tasks
                    if time.time() % 10 < 1: 
                        raise RuntimeError(f"Simulated error for complex task {{task.get('id')}}")

            def run_tasks(self, tasks):
                """
                Execute multiple tasks with intelligent scheduling
                """
                if tasks is None:
                    tasks = self.generate_default_tasks()
                logger.info(f"Starting execution of {{len(tasks)}} tasks")
                batch_start_time = time.time()

                if self.config.get('enable_parallel', True) and len(tasks) > 1:
                    results = self.run_tasks_in_parallel(tasks)
                else:
                    results = self.run_tasks_sequential(tasks)
                
                #Calculate comprehensive metrics
                total_time = time.time() - batch_start_time
                success_count = sum(1 for r in results if r.success)
                avg_task_time = sum(r.execution_time for r in results) / len(results) if results else 0

                metrics = {{
                    "success_rate": success_count / len(results) if results else 0,
                    "avg_task_time": avg_task_time,
                    "error_count": len(results) - success_count,
                    "total_tasks": len(results),
                    "total_execution_time": total_time,
                    "system_metrics": self.metrics.copy()
                }}
                logger.info(f"Batch execution completed: {{metrics}}")
                return metrics

            def run_tasks_sequential(self, tasks): 
                """Execute task sequentially"""
                results = []
                for task in tasks:
                    result = self.run_task(task)
                    results.append(result)
                return results

            def run_tasks_parallel(self, tasks):
                """Execute tasks in parallel using ThreadPoolExecutor"""
                results = []
                max_workers = min(self.config.get('max_workers', 4), len(tasks))
                with ThreadPoolExecutor(max_workers = max_workers) as executor:
                    future_to_task = {{executor.submit(self.run_task, task): task for task in tasks}}
                    for future in as_completed(future_to_task):
                        try:
                            result = future.result(timeout = self.config.get('task_timeout', 30))
                            results.append(result)
                        except Exception as e:
                            #Create error result for failed failure
                            task = future_to_task[future]
                            error_result = TaskResult(
                                task_id = task.get('id', 0),
                                status = TaskStatus.FAILED,
                                execution_time = 0,
                                success = False,
                                error = f"Future execution failed: {{str(e)}}"
                            )
                            results.append(error_result)
                return results

            def generate_default_tasks(self):
                """Generate default tasks for testing"""
                return [
                    {{
                        "id": i,
                        "type": "test",
                        "data": f"sample_{{i}}",
                        "complexity": (i % 5 ) + 1
                    }} for i in range(10)
                ]

            def get_system_status(self):
                """Get current system status and metrics"""
                with self.lock:
                    status = {{
                        'name': self.name,
                        'config': self.config,
                        'metrics': self.metrics.copy(),
                        'success_rate': (
                            self.metrics['successful_tasks'] / self.metrics['total_tasks']
                            if self.metrics['total_tasks'] > 0 else 0
                        )
                    }}
                return status
        '''
    
    def generate_train_script(self, context):
        """
        Generate enhanced training/execution script
        """
        #acquire information from context
        hypothesis = context.get('hypothesis','Improve agentic system performance')
        enable_parallel = 'parallel' in hypothesis.lower() or 'concurrent' in hypothesis.lower()
        enable_optimisation = 'optimization' in hypothesis.lower() or 'optimize' in hypothesis.lower()
        max_workers = 8 if enable_parallel else 4
        task_timeout = 60 if enable_optimisation else 30

        #tune the configuration dynamically based on hypothesis
        return f'''
        """
        CoSTEER-Ehanced Training/Execution Script for Agentic System
        hypothesis: {hypothesis}
        """
        import json
        import sys
        import time
        import traceback
        from pathlib import Path
        from agent import AgenticSystem

        def main():
            """
            Main execution function with comprehensive error handling and reporting
            """
            try:
                print("CoSTEER Agentic System Execution Started")
                execution_start = time.time()
                #Initialize agent with configuration
                config = {{
                    'max_workers': {max_workers},
                    'enable_parallel': {enable_parallel}, 
                    'enable_optimisation': {enable_optimisation},
                    'task_timeout': {task_timeout}
                }}
                
                print(f"Configuration: {{config}}")
                agent = AgenticSystem(config)
                print(f"Initialized: {{agent.name}}")

                #Run tasks and collect results
                results = agent.run_tasks()

                #Save detailed result to file
                detailed_results = {{
                    'execution_results': results,
                    'system_status': agent.get_system_status(),
                    'execution_time': time.time() - execution_start,
                    'timestamp': time.time(),
                    'hypothesis': "{hypothesis}"
                }}

                result_file = Path("result.json")
                result_file.write_text(json.dumps(detailed_results, indent = 2))
                #Print structured output for parsing
                print("Execution Result")
                print(f"Success Rate: {{results['Success_rate']}}")
                print(f"Average Task Time: {{results['avg_task_time']}}")
                print(f"Error Count: {{results['error_count']}}")
                print(f"Total Tasks: {{results['total_tasks']}}")
                print(f"Total Execution Time: {{detailed_results['execution_time']}}s")
                #JSON output for automated parsing
                print("JSON RESULTS:")
                print(json.dumps(results))

            except Exception as e:
                print(f"Error: Execution failed - {str(e)}",file = sys.stderr)
                print("Error Details")
                traceback.print_exc()
                error_result = {{
                    "success_rate": 0,
                    "average time", float('inf'),
                    "error_count": 1,
                    "total_tasks": 0,
                    "error_reason": str(e)
                }}
                # Save error result
                try:
                    error_file = Path("error_result.json")
                    error_fi;e.write_text(json.dumps(error_result, indent = 2))
                except:
                    pass
                return 1
        if __name__ == "__main__":
            exit_code = main()
            sys.exit(exit_code)
        '''
    
    def generate_requirements(self, context):
        """
        Generate requirements file based on context
        """
        hypothesis = context.get('hypothesis', '')
        requirements = [
            "# CoSTEER Generated Requirements",
            "# Basic dependencies",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
        ]
        
        # Add context-specific requirements
        if 'parallel' in hypothesis.lower() or 'concurrent' in hypothesis.lower():
            requirements.extend([
                "# Parallel processing",
                "joblib>=1.0.0",
                "concurrent-futures>=3.1.1"
            ])
        
        if 'monitoring' in hypothesis.lower() or 'metrics' in hypothesis.lower():
            requirements.extend([
                "# Monitoring and metrics", 
                "psutil>=5.8.0",
                "prometheus-client>=0.11.0"
            ])
        
        if 'optimization' in hypothesis.lower():
            requirements.extend([
                "# Optimization libraries",
                "scipy>=1.7.0",
                "scikit-learn>=1.0.0"
            ])
        
        return "\\n".join(requirements) + "\\n"
    
    def needs_config_file(self, context):
        """
        Determine if a configuration file is needed
        """
        hypothesis = context.get('hypothesis', '')
        return any(keyword in hypothesis.lower() for keyword in ['config', 'parameter', 'setting', 'tune'])
    
    def generate_config_file(self, context):
        """
        Generate configuration file 
        """
        #acquire hypothesis from context and tune config accordingly
        hypothesis = context.get('hypothesis', 'Improve agentic system performance')

        #decide default config values based on hypothesis
        enable_parallel = 'parallel' in hypothesis.lower() or 'concurrent' in hypothesis.lower()
        enable_optimization = 'optimization' in hypothesis.lower() or 'optimize' in hypothesis.lower()
        max_workers = 8 if enable_parallel else 4
        task_timeout = 60 if enable_optimization else 30
        batch_size = 20 if enable_optimization else 10
        retry_attempts = 5 if enable_optimization else 3

        return '''
        """
        CoSTEER Generated Configuration
        """
        import os
        from dataclasses import dataclass
        from typing import Dict, Any

        @dataclass
        class AgentSystemConfig:
            """Configuration for agentic system"""
            #Execution settings
            max_workers: int = {max_workers}
            task_timeout: float = {task_timeout}
            enable_parallel: bool = {enable_parallel}
            enable_optimization: bool = {enable_optimization}
    
            # Performance settings
            retry_attempts: int = {retry_attempts}
            batch_size: int = {batch_size}
            
            # Logging settings
            log_level: str = "INFO"
            enable_detailed_logging: bool = True
            
            @classmethod
            def from_env(cls) -> 'AgenticSystemConfig':
                """Create config from environment variables"""
                return cls(
                    max_workers = int(os.getenv('AGENT_MAX_WORKERS', '{max_workers}')),
                    task_timeout = float(os.getenv('AGENT_TASK_TIMEOUT', '{task_timeout}')),
                    enable_parallel = os.getenv('AGENT_ENABLE_PARALLEL', '{str(enable_parallel).lower()}').lower() == 'true',
                    enable_optimization = os.getenv('AGENT_ENABLE_OPTIMIZATION', '{str(enable_optimization).lower()}').lower() == 'true',
                    retry_attempts = int(os.getenv('AGENT_RETRY_ATTEMPTS', '{retry_attempts}')),
                    batch_size = int(os.getenv('AGENT_BATCH_SIZE', '{batch_size}')),
                )
            
            def to_dict(self) -> Dict[str, Any]:
                """Convert config to dictionary"""
                return {{
                    'max_workers': self.max_workers,
                    'task_timeout': self.task_timeout,
                    'enable_parallel': self.enable_parallel,
                    'enable_optimization': self.enable_optimization,
                    'retry_attempts': self.retry_attempts,
                    'batch_size': self.batch_size,
                    'log_level': self.log_level,
                    'enable_detailed_logging': self.enable_detailed_logging
                }}

        # Default configuration instance
        DEFAULT_CONFIG = AgenticSystemConfig()

        #Example Usage
        #config = AgenticSystemConfig.from_hypothesis("{hypothesis}")
        #config = AgenticSystemConfig.from_env()
        '''

    def create_fallback_workspace(self, exp: Experiment) -> FBWorkspace:
        """Create a fallback worksapce in case of errors"""
        logger.warning("create fallback workspace due to previous errors")
        try:
            workspace = FBWorkspace()
            hypothesis = getattr(exp, 'hypothesis', 'Improve agentic system performance')
            exp_id = getattr(exp, 'id', 'unknown')
            
            # Create minimal working files
            minimal_files = {
                "agent.py": self.get_minimal_agent_code(hypothesis),
                "train.py": self.get_minimal_train_code(hypothesis,exp_id),
                "requirements.txt": "# Minimal requirements\\n",
                "README.md": f"# Fallback Workspace\nExperiment :{exp_id}\n Hypothesis: {hypothesis}\nThis is a fallback workspace with minimal working code."
            }
            
            workspace.inject_files(**minimal_files)
            logger.info(f"Created fallback workspace for experiment {exp_id}")
            return workspace
            
        except Exception as e:
            logger.error(f"Failed to create fallback workspace: {e}")
            raise

    def get_minimal_agent_code(self,hypothesis):
        """Get minimal working agent code"""
        return f'''
        class AgenticSystem:
            def __init__(self):
                self.name = "MinimalFallbackAgent"
                self.hypothesis = "{hypothesis}"
            def run_tasks(self):
                return {{
                    "success_rate": 0.5,
                    "avg_time": 0.01,
                    "error_count": 0,
                    "total_tasks": 1,
                    "note": "Fallback implementation"
                }}
        '''
    
    def get_minimal_train_code(self,hypothesis, exp_id):
        """Get minimal working train code"""
        return f'''
        import json
        from pathlib import Path
        from agent import AgenticSystem
        def main():
            print("Running fallback Implementation")
            print(f"Experiment: {exp_id}")
            print(f"Hypothesis: {hypothesis}")
            agent = AgenticSystem()
            results = agent.run_tasks()

            #Save results
            result_file = Path("result.json")
            result_file.write_text(json.dumps(results, indent=2))

            #Print results
            print(f"Success Rate: {{results['success_rate']}}")
            print(f"Average Time: {{results['avg_time']}}")
            print(f"Error Count: {{results['error_count']}}")
            print(f"Total Tasks: {{results['total_tasks']}}")
            print("=== Fallback Execution Completed ===")
            return 0
        if __name__ == "__main__":
            exit_code = main()
            import sys
            sys.exit(exit_code)
        ''' 

        # # begin drafting
        # # NOTE:
        # # We should implement CoSTEER here to improve high quality coding ability
        # # 1) generate code
        # # prompting
        # exp.experiment_workspace = FBWorkspace()
        # # exp.experiment_workspace.inject_files(**{"<filename>": <file content>})

        # # 2) run code
        # # prepare environment.
        # env = get_agent_sys_env(
        #     extra_volumes={exp.experiment_workspace.workspace_path: "/....."},
        #     # .....
        # )

        # env.run(entry="<entrypoint>", ...)

        # # Please refer to the following code for details.
        # # [[rdagent/components/coder/data_science/conf.py:41]]
        

        # # end drafting
        # try:
        #     #acquire workspace 
        #     ws_path = self.get_workspace_path(exp)
        #     #create workspace directory
        #     ws_path.mkdir(parents=True, exist_ok=True)
        #     #generate code
        #     self.generate_files(ws_path, exp)
        #     logger.info(f"Code generation as workspace at {ws_path}")

        # except Exception as e:
        #     logger.error(f"Code generation failed: {str(e)}")
        #     exp.exception = e
        
        # return exp

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
        Generate necessary files for the agentic system experiment and write file to disk
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
        return f'''
        """
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

            def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
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
        """generate execution template"""
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

                # Save results to file (for backup parsing)
                result_file = Path("result.json")
                result_file.write_text(json.dumps(results, indent = 2))

                #Print for logging
                print("execution completed")
                print(f"Success Rate: {results['success_rate']}")
                print(f"Average Time: {results['avg_time']}")
                print(f"Error Count: {results['error_count']}")
                print(f"Total Tasks: {results['total_tasks']}")

                return 0

            except Exception as e:
                print(f"Execution failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return 1
        
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
        # try: 
        #     # acquire workspace
        #     ws_path = self.get_workspace_path(exp)
        #     logger.info(f"Using workspace at {ws_path}")
        #     # validate necessary files
        #     self.validate_workspace(ws_path)
        #     #execute experiment
        #     stdout, stderr = self.execute_experiment(ws_path)
        #     #parse result
        #     result = self.parse_execution_output(stdout, stderr)
        #     exp.result = result
        #     # record execution logs
        #     self._log_execution_results(exp, result)
        #     logger.info("Experiment completed successfully")
        # except Exception as e:
        #     logger.error(f"Experiment execution failed: {str(e)}")
        #     exp.exception = e
        #     exp.result = self.create_error_result(str(e))
        # return exp
        try:
            if not self.has_valid_workspace(exp):
                logger.info("Workspace is not ready, calling coder to generate code")
                coder = AgenticSysCoder(self.scen)
                exp = coder.develop(exp)
                #check if coder succeeded
                if not self.has_valid_workspace(exp):
                    raise RuntimeError("Coder failed to generate valid workspace")
            #1. acquire workspace
            ws_path = self.get_workspace_path(exp)
            logger.info(f"Using workspace at {ws_path}")

            #2. validate necessary files
            self.validate_workspace(ws_path)

            #3. execute experiment
            stdout, stderr = self.execute_experiment(ws_path)

            #4. parse result
            result = self.parse_execution_output(stdout, stderr)
            exp.result = result

            #5. record execution logs
            self.log_execution_results(exp, result)
            logger.info("Experiment completed successfully")

        except Exception as e:
            logger.error(f"Experiment execution failed: {str(e)}")
            exp.exception = e
            exp.result = self.create_error_result(str(e))
        return exp
    
    def has_valid_workspace(self, exp: Experiment):
        """check if experiment has valid workspace with required files"""
        try:
            if not hasattr(exp, 'experiment_workspace') or not exp.experiment_workspace:
                return False
            ws_path = Path(exp.experiment_workspace.workspace_path)
            if not ws_path.exists():
                return False
            #check for required files
            required_files = ["train.py", "agent.py"]
            for file_name in required_files:
                if not (ws_path / file_name).exists():
                    return False
            return True
        except Exception as e:
            logger.warning(f"Error checking workspace validity : {(e)}")
            return False
    
    def get_workspace_path(self, exp):
        '''
        Get workspace path for the experiment
        '''
        if hasattr(exp, 'experiment_workspace') and exp.experiment_workspace:
            return Path(exp.experiment_workspace.workspace_path)
        # Default workspace path
        base = Path("./workspace")
        return base / f"exp_{exp.id}"
    
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
        cmd = [sys.executable, "train.py"]
        # use environment variables if necessary
        env = self._prepare_environment() 
        
        logger.info(f"Executing: {' '.join(cmd)} in {ws_path}")
        
        try:
        # pass in environment variables if necessary
            result = subprocess.run(
                cmd,
                cwd=str(ws_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env 
            )
            
            logger.info(f"Process completed with return code: {result.returncode}")
            
            if result.returncode != 0:
                logger.warning(f"Process exited with non-zero code: {result.returncode}")
            
            return result.stdout, result.stderr
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Execution timed out after {timeout} seconds")
            raise RuntimeError(f"Execution timeout: {timeout}s") from e
        
        except Exception as e:
            logger.error(f"Execution failed with exception: {str(e)}")
            raise RuntimeError(f"Execution error: {str(e)}") from e
        
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



            






        
