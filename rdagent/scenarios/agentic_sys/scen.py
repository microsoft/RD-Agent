
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario


class AgenticSysScen(Scenario):
    def __init__(self, competition: str) -> None:
        self.competition = competition

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
        return f"""Competition: {self.competition}, Objective: {base_desc}. Focus: Create autonomous AI agents that can execute complex tasks with minimal human intervention. Key requirements include task planning, execution monitoring, error handling, and performance optimization."""

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
        - Structured output format(JSON)
        
        Execution Context:
        - Isolated workspace directory
        - Configurable timeout settings
        - Resource monitoring (CPU, Memory usage) and cleanup
        - Result validation and reporting
        """

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        """Combine all descriptions together"""
        parts = []
        if simple_background:
            parts.append(f"Competition: {self.competition}. Develop an autonomous agentic system.")
        else:
            parts.append(self.background)
            parts.append(self.get_runtime_environment())
        if task:
            parts.append(f"\n --- Current Tasks ---")
            parts.append(task.description)
            task_desc = task.description.lower()
            if 'memory' in task_desc:
                parts.append("Additional Focus: Memory management and state persistence.")
            elif 'parallel' in task_desc:
                parts.append("Additional Focus: Parallel execution and concurrency handling.")
            elif 'planning' in task_desc:
                parts.append("Additional Focus: Advanced planning and multi-step task decomposition.")
        if filtered_tag:
            parts.append(f"\n--- Filtered Tags: {filtered_tag} ---")
            tag_guidance = self.get_tag_guidance(filtered_tag)
            if tag_guidance:
                parts.append(tag_guidance)

        if not simple_background:
            parts.append(self.get_success_criteria())

        return "\n".join(parts)

    def get_tag_guidance(self, tag):
        """acquire specific guidance based on tag"""
        tag_guidance = {
            "performance": "Optimize for speed and resource efficiency",
            "robustness": "Focus on error handling and system stability",
            "scalability": "Design for handling larger and more complex tasks",
            "planning": "Emphasize strategic thinking and multi-step execution",
            "coordination": "Multi-agent communication and collaboration"
        }
        return tag_guidance.get(tag.lower(),f"focus on {tag} aspects.")
    
    def get_success_criteria(self):
        '''acuqire success criteria'''
        return """
        Success Criteria:
        Primary metrics:
        - Task Success Rate: >= 70%
        - Average Execution Time: Within reasonable limits 
        - Error rate: < 10%
        Implementation Requirements:
        - Clean, maintainable code structure
        - Proper error handling and logging
        - JSON-formatted result output
        - Autonomous task execution capability
        """                    

    @property
    def rich_style_description(self) -> str:
        """Rich style description to present"""
        return f"<b>AgenticSysScen</b> for competition: <i>{self.competition}</i>"
