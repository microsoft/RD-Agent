
from rdagent.core.experiment import Task
from rdagent.core.proposal import ExpGen, Trace
from rdagent.scenarios.agentic_sys.exp import AgenticSysExperiment
from rdagent.core.proposal import (
    ExpGen,
    Hypothesis,
    HypothesisGen,
    Trace,
    Experiment2Feedback
)
from rdagent.scenarios.agentic_sys.scen import AgenticSysScen
from rdagent.log import rdagent_logger as logger

#define experiment generator and generate agentic system experiment
class AgenticSysExpGen(ExpGen):
    """
    1. follow RDAgent Framework design module, inherit ExpGen class
    2. ensure compatibility with Trace and Experiment classes
    3. make full use of fundemantal functionality and agreement of RDAgent Framework
    """

    def __init__(self, scen: AgenticSysScen):
        """insert scenario context"""
        self.scen = scen
        
    def gen(self, trace: Trace) -> AgenticSysExperiment:
        # 1. analyze whether has historical record
        is_first_experiment = not (hasattr(trace, 'hist') and trace.hist)

        # 2. generate hypothesis
        if is_first_experiment: 
            task_desc = self.get_baseline_task()
        else:
            task_desc = self.get_improvement_task(trace)

        # 3. construct experiment
        main_task = Task(task_desc)
        experiment = AgenticSysExperiment(
            sub_tasks = [main_task]
        )

        #4. Add assumption 
        experiment.hypothesis = self.generate_simple_hypothesis(trace)
        logger.info(f"Generated experiment with task: ")
        return experiment
    
    def get_baseline_task(self):
        """first time experiment task"""
        competition = getattr(self.scen, "competition", 'general') if self.scen else 'general'
        return f"""Design and implement a baseline agentic system for the {competition}.
        Requirements:
        1. Create an AgenticSystem class that can execute multiple tasks autonomously.
        2. Implement task execution with performance monitoring
        3. Include metrics collection
        4. Add proper error handling and logging
5. Output results in JSON format

The system should demonstrate:
- Task planning and execution
- Basic error recovery
- Performance measurement
- Clean code structure
"""
    def get_improvement_task(self, trace: Trace) -> str:
        """generate improvement task based on trace history"""
        # analyze last experiment feedback
        last_exp, last_feedback = trace.hist[-1]
        
        if last_feedback.decision:
            # last experiment succeeded, continue improvement
            base_desc = "Enhance the successful agentic system from the previous experiment."
        else:
            # last experiment failed, fix issues
            base_desc = "Fix the issues in the previous agentic system implementation."
        # extract specific improvement directions from feedback
        feedback_hint = ""
        if hasattr(last_feedback, 'reason') and last_feedback.reason:
            if 'success_rate' in last_feedback.reason.lower():
                feedback_hint = "\n- Focus on improving task completion success rate"
            elif 'time' in last_feedback.reason.lower():
                feedback_hint = "\n- Focus on optimizing execution time and efficiency"
            elif 'error' in last_feedback.reason.lower():
                feedback_hint = "\n- Focus on reducing errors and improving robustness"
        
        competition = getattr(self.scen, 'competition', 'general') if self.scen else 'general'
        
        return f"""{base_desc}

Competition: {competition}
Previous feedback: {getattr(last_feedback, 'reason', 'No specific feedback')[:200]}

Improvement requirements:
1. Analyze the previous implementation and identify bottlenecks
2. Implement specific optimizations based on the feedback
3. Maintain or improve the current performance metrics
4. Add new features or capabilities as needed{feedback_hint}

Ensure backwards compatibility while introducing improvements.
"""
    
    def generate_simple_hypothesis(self, trace: Trace) -> str:
        """Generate a simple hypothesis based on trace history"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "Establish a baseline agentic system with core functionality"
        
        # Analyze historical performance
        success_count = 0
        total_experiments = len(trace.hist)
        
        for exp, feedback in trace.hist:
            if feedback and feedback.decision:
                success_count += 1
        
        success_rate = success_count / total_experiments if total_experiments > 0 else 0
        
        if success_rate > 0.7:
            return f"Build upon {success_count}/{total_experiments} successful experiments to further optimize the agentic system"
        else:
            return f"Address the challenges from {total_experiments - success_count}/{total_experiments} failed attempts and establish a stable implementation"
    


    
    
    


        
