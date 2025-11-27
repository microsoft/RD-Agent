from typing import Any, Dict, Optional
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.agentic_sys.evaluator import DeepResearchEvaluator, EvaluationResult

#define experiment scenario
#scenario abstraction for agentic system development
#support different competition contexts
class AgenticSysScen(Scenario):
    def __init__(self, competition: str,evaluation_weights: Optional[Dict[str, float]] = None) -> None:
        self.competition = competition

        #Initialize DeepResearch Bench evaluator
        self.evaluator = DeepResearchEvaluator(dimension_weights=evaluation_weights)

        # Set competition-specific evaluation weights
        self.evaluation_weights = evaluation_weights or {
            'comprehensiveness': 0.25,
            'insight': 0.25,
            'instruction_following': 0.25,
            'readability': 0.25
        }


    # Implement dummy functions for the abstract methods in Scenario
    @property
    def background(self) -> str:
        """Background information"""
        background_template = {
            "deepresearch": "Advanced AI agent research focusing on autonomous reasoning and complex problem solving",
            "tool_usage": "Development of agents with sophisticated tool usage and API integration capabilities", 
            "multi_agent": "Multi-agent systems with coordination, communication, and collaborative task execution",
            "planning": "Agent planning systems with strategic thinking and multi-step task decomposition",
            "general": "General-purpose agentic system development with broad task handling capabilities"
        }
        base_desc = background_template.get(self.competition,  f"Agentic system development for {self.competition}")

        evaluation_info = f"""
        
        Evaluation Framework: DeepResearch Bench Standards
        - Comprehensiveness (weight: {self.evaluator.weights['comprehensiveness']:.2f}): Breadth and depth of coverage
        - Insight (weight: {self.evaluator.weights['insight']:.2f}): Causal reasoning and originality
        - Instruction Following (weight: {self.evaluator.weights['instruction_following']:.2f}): Task compliance
        - Readability (weight: {self.evaluator.weights['readability']:.2f}): Clarity and presentation
        """

        return f"""Competition: {self.competition},Objective: {base_desc}, Focus: Create autonomous AI agents that can execute complex tasks with minimal human intervention. 
Key requirements include task planning, execution monitoring, error handling, and performance optimization. {evaluation_info}"""
    

    #running environment description and standards
    def get_runtime_environment(self) -> str:
        """Get the runtime environment information"""
        return f"""Runtime Environment for competition {self.competition}: 
        Base Requirements: 
        - Python 3.8+ execution environment
        - JSON serialization support for results
        - File I/O capabilities for workspace management
        - Standard Library access

        Agent Framework: 
        - Task execution and monitoring system
        - Performance metrics collection module (success rate, average time, error count)
        - Error handling and logging mechanisms
        - Structured output format (JSON)
        - DeepResearch Bench evaluation integration
        
        Execution Context:
        - Isolated workspace directory
        - Configurable timeout settings
        - Resource monitoring (CPU, Memory usage) and cleanup
        - Result validation and reporting
        - Multi-dimensional quality assessment (Comprehensiveness, Insight, Instruction Following, Readability)
        """

    #task content analyze
    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        """Combine all descriptions together"""
        parts = []

        #1. basic information processing
        if simple_background:
            parts.append(f"Competition: {self.competition}. Develop an autonomous agentic system.")
        else:
            parts.append(self.background)
            parts.append(self.get_runtime_environment())

        #2. task specific processing
        if task:
            parts.append(f"\n--- Current Task ---")
            parts.append(task.description)
            task_desc = task.description.lower()
            if 'memory' in task_desc:
                parts.append("Additional Focus: Memory management and state persistence.")
            elif 'parallel' in task_desc:
                parts.append("Additional Focus: Parallel execution and concurrency handling.")
            elif 'planning' in task_desc:
                parts.append("Additional Focus: Advanced planning and multi-step task decomposition.")
            
            # Add evaluation criteria for this task
            parts.append(self.get_task_evaluation_criteria(task))

        if filtered_tag:
            parts.append(f"\n--- Filtered Tags: {filtered_tag} ---")
            tag_guidance = self.get_tag_guidance(filtered_tag)
            if tag_guidance:
                parts.append(tag_guidance)

        if not simple_background:
            parts.append(self.get_success_criteria())

        return "\n".join(parts)

    def get_task_evaluation_criteria(self, task: Task) -> str:
        """Get evaluation criteria specific to the task"""

        #extract task-specific information
        task_desc = task.description.lower() if task and task.description else ""
        task_domain = getattr(task, 'domain', 'general') if task else 'general'

        focus_areas = []
        emphasis_dimensions = {}

        #Analyze task description to adjust criteria emphasis
        if 'comprehensive' in task_desc:
            focus_areas.append("comprehensive coverage")
            emphasis_dimensions['comprehensiveness'] = 'emphasized'

        if 'analyze' in task_desc or 'explain' in task_desc or 'reason' in task_desc:
            focus_areas.append("analytical reasoning")
            emphasis_dimensions['insight'] = 'emphasized'

        if 'follow' in task_desc or 'present' in task_desc or 'format' in task_desc:
            focus_areas.append("strict instruction adherence")
            emphasis_dimensions['instruction_following'] = 'emphasized'

        if 'report' in task_desc or 'present' in task_desc or 'clarity' in task_desc:
            focus_areas.append("clear presentation")
            emphasis_dimensions['readability'] = 'emphasized'
        
        #build focus statement
        focus_statement = ""
        if focus_areas:
            focus_statement = f"\n**Task Focus**: This task particularly emphasizes {', '.join(focus_areas)}.\n"
        else:
            focus_statement = "\n**Task Focus**: Standard evaluation across all dimensions, including comprehensiveness, Insight, Instruction following and readability\n"
        
        #domain specific guidance
        domain_guidance = self.get_domain_specific_guidance(task_domain)

        #build criteria with emphasis markers
        comp_marker = emphasis_dimensions.get('comprehensiveness', '')
        insight_marker = emphasis_dimensions.get('insight', '')
        instruction_marker = emphasis_dimensions.get('instruction_following', '')
        readability_marker = emphasis_dimensions.get('readability', '')

        return f"""
--- Evaluation Criteria (DeepResearch Bench) ---
{focus_statement}

Your solution will be evaluated on four dimensions (0-10 scale each):

1. Comprehensiveness ({self.evaluator.weights['comprehensiveness']:.0%} weight):
   - Coverage of all required subtopics
   - Depth of analysis with evidence
   - Multiple perspectives considered
   - No major omissions
   
2. Insight ({self.evaluator.weights['insight']:.0%} weight):
   - Causal reasoning and why-think
   - Quantified analysis with data
   - Non-obvious implications identified
   - Novel synthesis or frameworks
   
3. Instruction Following ({self.evaluator.weights['instruction_following']:.0%} weight):
   - Answers all sub-questions
   - Respects scope and constraints
   - Required deliverables present
   - Avoids out-of-scope content
   
4. Readability ({self.evaluator.weights['readability']:.0%} weight):
   - Clear structure and organization
   - Fluent, precise language
   - Effective data presentation
   - Proper formatting

Overall Score: Weighted sum of four dimensions
Target: >= 7.0/10.0 overall for success
"""

    def get_tag_guidance(self, tag):
        """acquire specific guidance based on tag"""
        tag_guidance = {
            "performance": "Optimize for speed and resource efficiency. Evaluation: Focus on insight (efficiency analysis) and comprehensiveness (performance metrics).",
            "robustness": "Focus on error handling and system stability. Evaluation: Emphasize comprehensiveness (edge cases) and instruction following (requirements).",
            "scalability": "Design for handling larger and more complex tasks. Evaluation: Highlight insight (scalability analysis) and comprehensiveness (architectural depth).",
            "planning": "Emphasize strategic thinking and multi-step execution. Evaluation: Prioritize insight (causal reasoning) and comprehensiveness (planning depth).",
            "coordination": "Multi-agent communication and collaboration. Evaluation: Focus on comprehensiveness (interaction coverage) and readability (clear protocols)."
        }
        return tag_guidance.get(tag.lower(), f"Focus on {tag} aspects.")
    

    
    def get_success_criteria(self):
        '''acquire success criteria with DeepResearch Bench standards'''
        return f"""
--- Success Criteria ---

Primary Metrics (Execution):
- Task Success Rate: >= 70%
- Average Execution Time: Within reasonable limits 
- Error Rate: < 10%

Quality Metrics (DeepResearch Bench):
- Comprehensiveness: >= 6.0/10.0 (adequate coverage)
- Insight: >= 6.0/10.0 (clear reasoning)
- Instruction Following: >= 7.0/10.0 (compliant)
- Readability: >= 6.0/10.0 (clear presentation)
- Overall Score: >= 7.0/10.0

Implementation Requirements:
- Clean, maintainable code structure
- Proper error handling and logging
- JSON-formatted result output with evaluation scores
- Autonomous task execution capability
- Documented reasoning and decision-making process

Scoring Guidance:
- 0-2: Poor/Missing - Major issues
- 4-6: Basic/Adequate - Meets minimum requirements
- 6-8: Good/Complete - Solid implementation
- 8-10: Excellent/Exhaustive - Outstanding quality
"""

    def evaluate_output(
        self,
        output: Any,
        task: Optional[Task] = None,
        reference_output: Optional[Any] = None
    ) -> EvaluationResult:
        """
        Evaluate output using DeepResearch Bench standards
        
        Args:
            output: The agent's output to evaluate
            task: Optional task for context
            reference_output: Optional reference for normalization
            
        Returns:
            EvaluationResult with scores for all dimensions
        """
        # Prepare task requirements
        task_requirements = {}
        task_context = {}
        
        if task:
            task_context = {
                'task_description': task.description,
                'competition': self.competition
            }
            
            # Extract requirements from task description
            task_desc_lower = task.description.lower()
            task_requirements['required_sections'] = []
            
            if 'results' in task_desc_lower or 'output' in task_desc_lower:
                task_requirements['required_sections'].append('results')
            if 'analysis' in task_desc_lower or 'evaluate' in task_desc_lower:
                task_requirements['required_sections'].append('analysis')
            if 'metrics' in task_desc_lower or 'performance' in task_desc_lower:
                task_requirements['required_sections'].append('metrics')
        
        # Evaluate using the evaluator
        reference_result = None
        if reference_output:
            reference_result = self.evaluator.evaluate(
                reference_output,
                task_requirements,
                task_context
            )
        
        result = self.evaluator.evaluate(
            output,
            task_requirements,
            task_context,
            reference_result
        )
        
        return result

    @property
    def rich_style_description(self) -> str:
        """Rich style description to present"""
        return f"<b>AgenticSysScen</b> for competition: <i>{self.competition}</i> with <b>DeepResearch Bench</b> evaluation"