

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, ExperimentFeedback, Trace


class AgenticSysExp2Feedback(Experiment2Feedback):
    def generate_feedback(self, experiment: Experiment, trace: Trace) -> ExperimentFeedback:
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

    
    


    
    
