from asyncio.log import logger
from pathlib import Path
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, ExperimentFeedback, Trace
import re
import json


class AgenticSysExp2Feedback(Experiment2Feedback):
    def generate_feedback(self, experiment: Experiment, trace: Trace) -> ExperimentFeedback:

        # BEGIN drafting
        # read content from `expriment.workspace_path`
        # END drafting
        try:
            if hasattr(experiment, 'experiment_workspace') and experiment.experiment_workspace:
                ws_path = Path(experiment.experiment_workspace.workspace_path)
                if ws_path.exists() and ws_path.is_dir():
                    logger.info(f"Reading results from workspace: {ws_path}")
                    #Try to read result files in order of preference
                    result_files = [
                        "result.json",
                        "detailed_result.json",
                        "output.json",
                        "error_result.json"
                    ]
                    for result_file in result_files:
                        result_path = ws_path / result_file
                        if result_path.exists():
                            try:
                                content = result_path.read_text()
                                data = json.loads(content)
                                #Extract execution results if nested 
                                if isinstance(data, dict):
                                    if "execution_result" in data:
                                        experiment.result = data["execution_result"]
                                    else:
                                        experiment.result = data
                                else:
                                    experiment.result = data
                                break
                            except Exception as e:
                                logger.warning(f"Failed to parse {result_file}: {e}")
                                continue
                    #if no result file found, try parsing stdout/stderr from workspace
                    if not hasattr(experiment, 'result') or experiment.result is None:
                        self.try_parse_logs(experiment, ws_path)
        except Exception as e:
            logger.warning(f"Failed to read workspace contents: {e}")



        # 1. check whether experiment ran successfully
        if not hasattr(experiment, 'result') or experiment.result is None:
            return ExperimentFeedback(
                reason = "Experiment did not complete execution.",
                decision = False,
                exception = getattr(experiment, 'exception', None)
            )
        
        #2. extract important metrics from experiment result
        result = experiment.result

        #evaluation metrics
        success_rate = result.get('success_rate', 0)
        avg_time = result.get('avg_time', float('inf'))
        error_count = result.get('error_count', 0)

        #3. formulate success criteria
        MIN_SUCCESS_RATE = 0.7
        MAX_AVG_TIME = 30
        MAX_ERROR_COUNT = 2

        is_successful = (
            success_rate >= MIN_SUCCESS_RATE and
            avg_time <= MAX_AVG_TIME and
            error_count <= MAX_ERROR_COUNT
        )

        #4. Compare with past experiments in the trace
        historical_best = self._get_best_from_trace(trace)
        is_improvement = False

        if historical_best:
            best_success_rate = historical_best.get('success_rate', 0)
            best_avg_time = historical_best.get('avg_time', float('inf'))

            is_improvement = (
                success_rate > best_success_rate or 
                (success_rate == best_success_rate and avg_time < best_avg_time)
            )

        else:
            #first-time experiment, we should still accept it even if it is fail.
            is_improvement = True

        #5. Generate detailed feedback
        reason_parts = []
        reason_parts.append(f"Success Rate: {success_rate:.2f}")
        reason_parts.append(f"Average Time: {avg_time:.2f}s")
        if error_count > 0:
            reason_parts.append(f"Errors Encountered: {error_count}")
        if is_improvement:
            reason_parts.append("This experiment shows improvement over past results.")
        elif historical_best:
            reason_parts.append(
                f"No improvement (best: {historical_best.get('success_rate', 0)})"
            )
        reason = "|".join(reason_parts)
        return ExperimentFeedback(
            reason = reason,
            decision = is_improvement,
        )
    
    def try_parse_logs(self, experiment, ws_path):
        """Try to parse result from workspace log files"""
        try:
            # look for commonn log files patterns
            log_pattern = ["*.log", "*.out", "train_output.txt","execution.log"]
            for pattern in log_pattern:
                for log_file in ws_path.glob(pattern):
                    try:
                        content = log_file.read_text()
                        parsed = self.parse_stdout_for_metrics(content)
                        if parsed:
                            experiment.result = parsed
                            return
                    except Exception as e:
                        logger.warning(f"Failed to parse log file {log_file}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Failed to read workspace contents: {e}")


    def parse_stdout_for_metrics(self, stdout):
        """Parse metrics from stdout text"""
        if not stdout:
            return None
        try:
            # Method 1: Try to extract JSON block
            json_pattern = r'=== JSON RESULTS ===\s*\n(.*?)\n=== END JSON ==='
            match = re.search(json_pattern, stdout, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass
            # Method 2: Try to find any JSON object
            json_obj_pattern = r'\{[^{}]*"success_rate"[^{}]*\}'
            match = re.search(json_obj_pattern, stdout)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Method 3: Parse text patterns
            metrics = {}
            
            success_match = re.search(r'Success Rate:\s*([0-9.]+)', stdout)
            time_match = re.search(r'Average Time:\s*([0-9.]+)', stdout)
            error_match = re.search(r'Error Count:\s*([0-9]+)', stdout)
            tasks_match = re.search(r'Total Tasks:\s*([0-9]+)', stdout)
            
            if success_match:
                metrics['success_rate'] = float(success_match.group(1))
                metrics['avg_time'] = float(time_match.group(1)) if time_match else 0.0
                metrics['error_count'] = int(error_match.group(1)) if error_match else 0
                metrics['total_tasks'] = int(tasks_match.group(1)) if tasks_match else 0
                return metrics
                
        except Exception as e:
            logger.debug(f"Failed to parse stdout: {e}")
        
        return None

                    


    
    def get_best_from_trace(self, trace:Trace):
        # Extract the best experiment result from the trace
        if not hasattr(trace, 'hist') or not trace.hist:
            return None
        best_result = None
        best_success_rate = -1
        for exp, feedback in trace.hist:
            if hasattr(exp, 'result') and exp.result:
                success_rate = exp.result.get('success_rate', 0)
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_result = exp.result
        return best_result
    
    def analyze_performance_issues(self, result):
        # analyze performance issues based on result metrics
        issues = []
        success_rate = result.get('success_rate', 0)
        avg_time = result.get('avg_time', float('inf'))
        error_count = result.get('error_count', 0)
        
        if success_rate < 0.3:
            issues.append("Critical: Very low success rate - review core algorithm")
        elif success_rate < 0.7:
            issues.append("Warning: Success rate below target - optimize task handling")
        
        if avg_time > 10:
            issues.append("Performance: High execution time - consider optimization")
        
        if error_count > 5:
            issues.append("Stability: High error count - improve error handling")
        
        return issues
    
    def get_evaluation_summary(self, trace):
        """Get summary of all experiments in trace"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return {"total": 0, "successful": 0, "average_success_rate": 0.0}
        
        total = len(trace.hist)
        successful = 0
        success_rates = []
        
        for exp, feedback in trace.hist:
            if hasattr(exp, 'result') and exp.result:
                success_rate = exp.result.get('success_rate', 0)
                success_rates.append(success_rate)
                if feedback and feedback.decision:
                    successful += 1
        
        return {
            "total": total,
            "successful": successful,
            "success_ratio": successful / total if total > 0 else 0,
            "average_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
            "best_success_rate": max(success_rates) if success_rates else 0
        }

    
    


    
    
