from rdagent.core.experiment import Task
from rdagent.core.proposal import ExpGen, Trace
from pathlib import Path
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
from rdagent.core.proposal import HypothesisGen, Hypothesis
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T  # 使用 T 模板系统
import json
from typing import Any, Dict, List, Optional, Tuple
from rdagent.scenarios.agentic_sys.tools.web_search import create_web_search_tool


class AgenticSysHypothesisGen(HypothesisGen):
    """
    Generate hypothesis for agentic system improvements based on DeepResearch evaluation dimensions.
    Uses T() template system to render prompts from prompts.yaml.
    """

    def __init__(self, scen: AgenticSysScen):
        super().__init__(scen=scen)
        self.scen = scen
        
        # Initialize LLM backend
        self.api_backend = APIBackend()

        #Initialize web search tool
        search_config_path = Path(__file__).parent /"tools"/ "search_config.yaml"
        self.web_search = create_web_search_tool(config_path=search_config_path)
        
        logger.info("AgenticSysHypothesisGen initialized with T() template system")


    @property
    def web_search_tool(self):
        """Lazy load web search tool when needed"""
        if self._web_search_tool is None:
            try:
                search_config_path = Path(__file__).parent / "tools" / "search_config.yaml"
                if search_config_path.exists():
                    self._web_search_tool = create_web_search_tool(search_config_path)
                    logger.info("✓ Web search tool initialized in HypothesisGen")
                else:
                    logger.warning(f"Search config not found: {search_config_path}")
                    self._web_search_tool = False
            except Exception as e:
                logger.warning(f"Failed to initialize web search tool: {e}")
                self._web_search_tool = False
        return self._web_search_tool if self._web_search_tool is not False else None


    def gen(self, trace: Trace) -> Hypothesis:
        """
        Generate hypothesis based on trace history and evaluation dimensions.
        
        Args:
            trace: Experiment trace containing history
            
        Returns:
            Hypothesis object with structured hypothesis data
        """
        logger.info("Generating hypothesis...")
        
        # Prepare base context
        scenario_desc = trace.scen.get_scenario_all_desc()
        previous_trials = self._extract_previous_trials(trace)
        
        # Optionally enhance with web search
        external_knowledge = []
        if self._should_use_web_search(trace):
            external_knowledge = self._retrieve_external_knowledge(trace)
        
        # Generate hypothesis using LLM
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            scenario_desc=scenario_desc,
            previous_trials=previous_trials,
            external_knowledge=external_knowledge
        )
        
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True
        )
        
        # Parse and return hypothesis
        hypothesis = self._parse_hypothesis(response, trace)
        
        logger.info(f"Generated hypothesis: {hypothesis.hypothesis[:100]}...")
        return hypothesis


    def _should_use_web_search(self, trace: Trace) -> bool:
        """Determine if web search should be used"""
        # Check if tool is available
        if self.web_search_tool is None:
            return False
        
        # Check if service is healthy
        if not self.web_search_tool.client.health_check():
            logger.warning("Web search service not healthy")
            return False
        
        # Use for early iterations
        iteration = len(trace.hist)
        if iteration < 3:
            logger.info(f"Early iteration ({iteration}/3), enabling web search")
            return True
        
        # Use if previous performance is low
        if trace.hist and hasattr(trace.hist[-1][1], 'overall_score'):
            last_score = trace.hist[-1][1].overall_score
            if last_score < 6.0:  # Threshold for low performance
                logger.info(f"Low previous score ({last_score}), enabling web search")
                return True
        
        return False
    
    def _retrieve_external_knowledge(self, trace: Trace) -> list:
        """
        Retrieve external knowledge using web search tool
        
        Args:
            trace: Execution trace
            
        Returns:
            List of external sources
        """
        try:
            scenario_desc = trace.scen.get_scenario_all_desc()
            
            # Identify knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(trace)
            
            # Prepare search context
            search_context = {
                'iteration': len(trace.hist),
                'domain': getattr(trace.scen, 'domain', 'general')
            }
            
            # Call web search tool
            logger.info("Retrieving external knowledge via web search...")
            external_sources = self.web_search_tool.search_for_hypothesis(
                task_description=scenario_desc,
                current_gaps=knowledge_gaps,
                context=search_context
            )
            
            logger.info(f"Retrieved {len(external_sources)} external sources")
            return external_sources
            
        except Exception as e:
            logger.error(f"Failed to retrieve external knowledge: {e}")
            return []
    
    def _identify_knowledge_gaps(self, trace: Trace) -> list:
        """Identify knowledge gaps from trace history"""
        gaps = []
        
        if trace.hist:
            last_feedback = trace.hist[-1][1]
            
            # Check which dimensions performed poorly
            if hasattr(last_feedback, 'dimension_feedback'):
                for dim, feedback in last_feedback.dimension_feedback.items():
                    if hasattr(feedback, 'score') and feedback.score < 6.0:
                        gaps.append(f"improve {dim}")
        
        # Default gaps if none identified
        if not gaps:
            gaps = [
                "agentic system best practices",
                "system design patterns",
                "performance optimization"
            ]
        
        return gaps[:5]
    
    def _extract_previous_trials(self, trace: Trace) -> str:
        """Extract previous trials from trace"""
        if not trace.hist:
            return "No previous trials"
        
        trials = []
        for exp, feedback in trace.hist[-3:]:  # Last 3 trials
            trial_summary = {
                'hypothesis': getattr(exp, 'hypothesis', 'N/A'),
                'result': getattr(feedback, 'decision', 'N/A'),
                'score': getattr(feedback, 'overall_score', 0.0)
            }
            trials.append(trial_summary)
        
        return str(trials)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for hypothesis generation"""
        return """You are an expert AI researcher specializing in agentic systems.
Your task is to generate innovative hypotheses for improving agentic system performance.

Consider:
1. Previous experimental results
2. External knowledge from research papers and best practices
3. Novel approaches and methodologies
4. Feasibility and implementability

Generate a clear, specific, and testable hypothesis."""
    
    def _build_user_prompt(
        self,
        scenario_desc: str,
        previous_trials: str,
        external_knowledge: list
    ) -> str:
        """Build user prompt with all context"""
        prompt = f"""# Scenario
{scenario_desc}

# Previous Trials
{previous_trials}
"""
        
        if external_knowledge:
            prompt += "\n# External Knowledge\n"
            for idx, source in enumerate(external_knowledge[:5], 1):
                prompt += f"\n{idx}. [{source['credibility_level']}] {source['title']}\n"
                prompt += f"   Summary: {source['summary'][:150]}...\n"
                prompt += f"   URL: {source['url']}\n"
        
        prompt += "\n# Task\nGenerate a hypothesis to improve the agentic system."
        
        return prompt
    
    def _parse_hypothesis(self, response: str, trace: Trace) -> Hypothesis:
        """Parse LLM response into Hypothesis object"""
        # Simplified parsing - in real implementation, use structured output
        hypothesis_text = response.strip()
        
        hypothesis = Hypothesis(
            hypothesis=hypothesis_text,
            reason="Generated based on scenario and previous results",
            concise_reason="Improve system performance",
            concise_observation="",
            concise_justification="",
            concise_knowledge=""
        )
        
        return hypothesis

    def prepare_context(self, trace: Trace):
        """
        Prepare context for hypothesis generation from trace history.
        
        KEY METHOD: Uses T() template system like Kaggle scenario
        
        Args:
            trace: Experiment trace
            
        Returns:
            Tuple of (context dictionary, is_first_experiment flag)
        """
        is_first_experiment = not (hasattr(trace, 'hist') and trace.hist)
        
        # Use T() to render hypothesis_and_feedback prompt
        hypothesis_and_feedback = (
            T("scenarios.agentic_sys.prompts:hypothesis_and_feedback").r(
                trace=trace,
                history_window=10,
                most_successful_action=self._get_most_successful_action(trace),
                most_improved_dimension=self._get_most_improved_dimension(trace),
                persistent_weaknesses=self._get_persistent_weaknesses(trace),
                effective_strategies=self._get_effective_strategies(trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        
        context = {
            "is_first_experiment": is_first_experiment,
            "current_system_description": self._get_system_description(trace),
            "experiment_history": hypothesis_and_feedback,  # 使用渲染后的提示词
            "performance_gaps": self._identify_performance_gaps(trace),
            "current_scores": self._extract_current_scores(trace),
        }
        
        return context, is_first_experiment

    def prepare_rag_context(self, trace: Trace):
        """
        Prepare RAG (Retrieval-Augmented Generation) context.
        
        Uses T() template system for RAG prompt rendering.
        
        Args:
            trace: Experiment trace
            
        Returns:
            Dictionary with RAG context
        """
        # Retrieve knowledge sources
        insights = self._retrieve_cross_task_insights()
        experiences = self._retrieve_current_task_experiences(trace)
        external_sources = self._retrieve_external_sources(trace)
        
        # Render RAG prompt if sources available
        rag_prompt = ""
        if insights or experiences or external_sources:
            try:
                rag_prompt = T("scenarios.agentic_sys.prompts:KG_hypothesis_gen_RAG").r(
                    insights=insights,
                    experiences=experiences,
                    external_sources=external_sources
                )
            except Exception as e:
                logger.warning(f"Failed to render KG_hypothesis_gen_RAG: {e}")
        
        return {
            "insights": insights,
            "experiences": experiences,
            "external_sources": external_sources,
            "rag_prompt": rag_prompt  # 渲染后的 RAG 提示词
        }

    def generate_hypothesis_with_llm(
        self, 
        context: Dict[str, Any], 
        rag_context: Dict[str, Any],
        trace: Trace
    ) -> Dict[str, Any]:
        """
        Generate hypothesis using LLM with prompts from prompts.yaml.
        
        Uses T() template system to render all prompts.
        
        Args:
            context: Context dictionary
            rag_context: RAG context dictionary
            trace: Experiment trace
            
        Returns:
            Parsed hypothesis data dictionary
        """
        # Step 1: Build system prompt using T()
        try: 
            system_prompt = T("scenarios.agentic_sys.prompts:hypothesis_generation").s()
            logger.info("Rendered hypothesis_generation system prompt")
        except Exception as e:
            logger.warning(f"Failed to render hypothesis_generation system prompt: {e}")
            system_prompt = """You are an expert in agentic system optimization and research automation.Your task is to propose hypotheses to improve the system's performance on DeepResearch evaluation dimensions."""
        
        # Step 2: Build user prompt using T()
        user_prompt = self._build_user_prompt_with_t(context, rag_context, trace)
        
        # Step 3: Call LLM
        logger.info("Calling LLM for hypothesis generation...")
        response = self.api_backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True
        )
        
        # Step 4: Parse JSON response
        try:
            hypothesis_data = json.loads(response)
            logger.info("Successfully parsed hypothesis JSON")
            return hypothesis_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse hypothesis JSON: {e}")
            logger.error(f"Response: {response}")
            return self.get_fallback_hypothesis(context)

    def build_user_prompt_with_t(
        self, 
        context: Dict[str, Any], 
        rag_context: Dict[str, Any],
        trace: Trace
    ) -> str:
        """
        Build user prompt using T() template system.
        
        KEY METHOD: Shows how to use T() to render and combine multiple prompts.
        
        Pattern:
        1. T("path:prompt_name").r(**variables) - Render user part
        2. T("path:prompt_name").s(**variables) - Render system part (if needed)
        3. Combine multiple rendered prompts with "\n\n"
        
        Args:
            context: Context dictionary
            rag_context: RAG context dictionary
            trace: Experiment trace
            
        Returns:
            Complete user prompt string
        """
        prompt_parts = []
        
        # Part 1: Task background (user part)
        try:
            task_bg_user = T("scenarios.agentic_sys.prompts:task_background").r(
                task_type=getattr(self.scen, 'task_type', 'Research Automation'),
                domain=getattr(self.scen, 'domain', 'Agentic Systems'),
                brief_description=getattr(self.scen, 'description', 'Automated research system'),
                scope_requirements=getattr(self.scen, 'scope', 'N/A'),
                required_deliverables=getattr(self.scen, 'deliverables', 'N/A'),
                comprehensiveness_focus=getattr(self.scen, 'comprehensiveness_focus', 'Complete coverage'),
                insight_focus=getattr(self.scen, 'insight_focus', 'Deep analysis'),
                instruction_focus=getattr(self.scen, 'instruction_focus', 'Strict adherence'),
                readability_focus=getattr(self.scen, 'readability_focus', 'Clear presentation')
            )
            prompt_parts.append(task_bg_user)
        except Exception as e:
            logger.warning(f"Failed to render task_background: {e}")
            prompt_parts.append(f"""Task Type: {getattr(self.scen, 'task_type', 'Research Automation')}
Domain: {getattr(self.scen, 'domain', 'Agentic Systems')}
Brief Description: {getattr(self.scen, 'description', 'Automated research system')}""")
        
        # Part 2: RAG context (if available)
        if rag_context.get("rag_prompt"):
            prompt_parts.append(rag_context["rag_prompt"])
            logger.info("Added RAG context")
        
        # Part 3: Main hypothesis generation instruction
        try:
            hypothesis_gen = T("scenarios.agentic_sys.prompts:hypothesis_generation").r(
                current_system_description=context["current_system_description"],
                current_comprehensiveness=context["current_scores"]["comprehensiveness"],
                current_insight=context["current_scores"]["insight"],
                current_instruction_following=context["current_scores"]["instruction_following"],
                current_readability=context["current_scores"]["readability"],
                experiment_history=context["experiment_history"],
                performance_gaps=context["performance_gaps"]
            )
            prompt_parts.append(hypothesis_gen)
            logger.info("Rendered hypothesis_generation user prompt")
        except Exception as e:
            logger.error(f"Failed to render hypothesis_generation: {e}")
            raise
        
        # Part 4: Output format specification
        try:
            output_format = T("scenarios.agentic_sys.prompts:hypothesis_output_format").r()
            prompt_parts.append(output_format)
        except Exception as e:
            logger.warning(f"Failed to render hypothesis_output_format: {e}")
        
        # Combine all parts
        full_prompt = "\n\n".join(prompt_parts)
        
        return full_prompt

    # ==================== Helper Methods for Context Preparation ====================
    
    def get_most_successful_action(self, trace: Trace) -> str:
        """Get most successful action type from trace history"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "N/A"
        
        action_success = {}
        for exp, feedback in trace.hist:
            action_type = getattr(exp, 'action_type', 'Unknown')
            if getattr(feedback, 'decision', False):
                action_success[action_type] = action_success.get(action_type, 0) + 1
        
        return max(action_success, key=action_success.get) if action_success else "N/A"
    
    def get_most_improved_dimension(self, trace: Trace) -> str:
        """Get most improved dimension from trace history"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "N/A"
        
        dimension_improvements = {
            "comprehensiveness": 0,
            "insight": 0,
            "instruction_following": 0,
            "readability": 0
        }
        
        for exp, feedback in trace.hist:
            for dim in dimension_improvements.keys():
                delta_attr = f"{dim}_delta"
                if hasattr(feedback, delta_attr):
                    delta = getattr(feedback, delta_attr, 0)
                    if delta > 0:
                        dimension_improvements[dim] += delta
        
        return max(dimension_improvements, key=dimension_improvements.get)
    
    def get_persistent_weaknesses(self, trace: Trace) -> str:
        """Identify persistent weaknesses from trace history"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "N/A"
        
        weaknesses = []
        if trace.hist:
            _, last_feedback = trace.hist[-1]
            for dim in ["comprehensiveness", "insight", "instruction_following", "readability"]:
                score_attr = f"{dim}_score"
                if hasattr(last_feedback, score_attr):
                    score = getattr(last_feedback, score_attr, 0)
                    if score < 6.0:
                        weaknesses.append(f"{dim} (score: {score:.1f})")
        
        return ", ".join(weaknesses) if weaknesses else "None identified"
    
    def get_effective_strategies(self, trace: Trace) -> str:
        """Get effective strategies from trace history"""
        most_successful = self._get_most_successful_action(trace)
        if most_successful != "N/A":
            return f"{most_successful} action type has been most successful"
        return "No clear pattern yet"

    def get_system_description(self, trace: Trace) -> str:
        """Get current system description from trace"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "No previous system implementation. Starting from baseline."
        
        last_exp, last_feedback = trace.hist[-1]
        
        description = f"Current system status:\n"
        description += f"- Last hypothesis: {getattr(last_exp, 'hypothesis', 'N/A')}\n"
        description += f"- Last feedback: {getattr(last_feedback, 'reason', 'N/A')[:200]}\n"
        description += f"- Success rate: {self._calculate_success_rate(trace):.1%}\n"
        
        return description

    def identify_performance_gaps(self, trace: Trace) -> str:
        """Identify performance gaps from trace history"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "Initial baseline establishment needed. Focus on core functionality."
        
        gaps = []
        
        # Analyze recent failures
        failed_experiments = [
            (exp, fb) for exp, fb in trace.hist[-5:]
            if not getattr(fb, 'decision', False)
        ]
        
        if failed_experiments:
            gaps.append(f"- {len(failed_experiments)} recent failures indicate instability")
        
        # Check success rate
        success_rate = self._calculate_success_rate(trace)
        if success_rate < 0.5:
            gaps.append(f"- Low success rate ({success_rate:.1%}) requires fundamental improvements")
        elif success_rate < 0.8:
            gaps.append(f"- Moderate success rate ({success_rate:.1%}) suggests refinement opportunities")
        
        return "\n".join(gaps) if gaps else "System performing well. Focus on advanced optimizations."

    def extract_current_scores(self, trace: Trace) -> Dict[str, Optional[float]]:
        """Extract current dimension scores from latest feedback"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return {
                "comprehensiveness": None,
                "insight": None,
                "instruction_following": None,
                "readability": None
            }
        
        _, last_feedback = trace.hist[-1]
        
        return {
            "comprehensiveness": getattr(last_feedback, 'comprehensiveness_score', None),
            "insight": getattr(last_feedback, 'insight_score', None),
            "instruction_following": getattr(last_feedback, 'instruction_score', None),
            "readability": getattr(last_feedback, 'readability_score', None)
        }

    def calculate_success_rate(self, trace: Trace) -> float:
        """Calculate success rate from trace history"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return 0.0
        
        success_count = sum(
            1 for _, fb in trace.hist
            if getattr(fb, 'decision', False)
        )
        
        return success_count / len(trace.hist)

    def extract_concise_observation(self, trace: Trace) -> str:
        """Extract concise observation from trace"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "Starting baseline implementation"
        
        _, last_feedback = trace.hist[-1]
        observations = getattr(last_feedback, 'observations', '')
        
        if observations:
            first_sentence = observations.split('.')[0]
            return first_sentence[:100] + "..." if len(first_sentence) > 100 else first_sentence
        
        return "Previous experiment completed"

    # ==================== RAG Methods ====================
    
    def retrieve_cross_task_insights(self) -> List[Dict[str, Any]]:
        """Retrieve insights from other similar tasks"""
        # TODO: Implement actual knowledge base retrieval
        return []

    def retrieve_current_task_experiences(self, trace: Trace) -> List[Dict[str, Any]]:
        """Retrieve relevant experiences from current task's trace history"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return []
        
        experiences = []
        for exp, fb in trace.hist[-5:]:
            experiences.append({
                "hypothesis": getattr(exp, 'hypothesis', 'N/A'),
                "approach": getattr(exp, 'action_type', 'N/A') if hasattr(exp, 'action_type') else 'N/A',
                "improved_dims": self._extract_improved_dimensions(fb),
                "lessons": getattr(fb, 'reason', 'N/A')[:200]
            })
        
        return experiences

    def retrieve_external_sources(self, trace: Trace) -> List[Dict[str, Any]]:
        """Retrieve external sources
        Args:
            trace: Experiment trace
        Returns:
            List of external source dictionaries
        """

        #check if web search is available
        if not self.web_search.client_health_check():
            logger.warning("SearxNG service unavailable. Skipping external search")
            return []
        #prepare search content
        task_description = getattr(self.scen, 'description', 'Automated research system')
        knowledge_gaps = self._identify_performance_gaps(trace)
        context = {
            "weak_dimension": self._get_most_improved_dimension(trace),
            "methodology": getattr(self.scen, 'task_type', '')
        }
        try:
            #perform web search
            external_sources = self.web_search.search_for_hypothesis(
                task_description = task_description,
                current_gaps = knowledge_gaps,
                context = context,
            )

            logger.info(f"Retrieved {len(external_sources)} external sources")
            return external_sources
        except Exception as e:
            logger.error(f"Failed to retrieve external sources: {e}")
            return []

    def identify_knowledge_gaps(self, trace):
        """
        Identify knowledge gaps from trace history for external search
        Args:
            trace: Experiment trace
        Returns:
            List of knowledge gap descriptions
        """
        gaps = []
        if not hasattr(trace, 'hist') or not trace.hist:
            gaps.append("baseline system design")
            gaps.append("evaluation metrics implementation")
            return gaps
        #analyze recent failures
        for exp, feedback in trace.hist[-3:]:
            if not getattr(feedback, 'decision', False):
                reason = getattr(feedback, 'reason', '')
                if 'error' in reason.lower():
                    gaps.append("error handling strategies")
                if 'coverage' in reason.lower():
                    gaps.append("comprehensive task coverage techniques")
                if 'insight' in reason.lower():
                    gaps.append("methods to enhance insight generation")

        #check dimension scores
        if hasattr(trace, 'hist') and trace.hist:
            _, last_feedback = trace.hist[-1]
            dimensions = {
                'comprehensiveness': getattr(last_feedback, 'comprehensiveness_score', 0),
                'insight': getattr(last_feedback, 'insight_score', 0),
                'instruction_following': getattr(last_feedback, 'instruction_score', 0),
                'readability': getattr(last_feedback, 'readability_score', 0)
            }
            #identify low scoring dimensions
            for dim, score in dimensions.items():
                if score and score < 6.0:
                    gaps.append(f"improving {dim} techniques")

        return gaps if gaps else ["general agentic system optimization"]


    def get_weak_dimension(self, trace):
        """
        get the weakest evaluation dimension from trace history
        """
        if not hasattr(trace, 'hist') or not trace.hist:
            return None
        _, last_feedback = trace.hist[-1]
        dimensions = {
            "comprehensiveness": getattr(last_feedback, 'comprehensiveness_score', 10),
            "insight": getattr(last_feedback, 'insight_score', 10),
            "instruction_following": getattr(last_feedback, 'instruction_score', 10),
            "readability": getattr(last_feedback, 'readability_score', 10)
        }

        if dimensions:
            weakest = min(dimensions, key = lambda x: x[1])
            return weakest[0]
        
        return None




    def extract_improved_dimensions(self, feedback) -> List[str]:
        """Extract which dimensions improved from feedback"""
        improved = []
        
        for dim in ["comprehensiveness", "insight", "instruction_following", "readability"]:
            delta_attr = f"{dim}_delta"
            if hasattr(feedback, delta_attr) and getattr(feedback, delta_attr, 0) > 0:
                improved.append(dim.replace("_", " ").title())
        
        return improved if improved else ["None"]

    def get_fallback_hypothesis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback hypothesis when LLM parsing fails"""
        return {
            "action": "Information_Gathering",
            "hypothesis": "Improve system based on previous feedback",
            "target_dimensions": [
                {
                    "name": "Comprehensiveness",
                    "current_score": context["current_scores"]["comprehensiveness"] or 0.0,
                    "target_score": (context["current_scores"]["comprehensiveness"] or 0.0) + 1.0,
                    "expected_improvement": 1.0,
                    "confidence": "Low"
                }
            ],
            "current_gap": "Unable to generate structured hypothesis",
            "rationale": "LLM response parsing failed. Using fallback hypothesis.",
            "implementation_plan": {
                "step_1": "Review previous feedback",
                "step_2": "Implement basic improvements",
                "step_3": "Validate changes"
            },
            "risk_assessment": {
                "potential_negative_impacts": [],
                "mitigation_strategies": ["Incremental changes", "Thorough testing"]
            },
            "success_criteria": {
                "primary": "System runs without errors",
                "secondary": ["Performance maintained or improved"],
                "validation_method": "Manual verification"
            },
            "concise_knowledge": "When LLM parsing fails, use incremental improvements"
        }


class AgenticSysExpGen(ExpGen):
    """Generate experiment based on hypothesis"""

    def __init__(self, scen: AgenticSysScen):
        self.scen = scen
        self.api_backend = APIBackend()
        logger.info("AgenticSysExpGen initialized with T() template system")
        
    def gen(self, trace: Trace) -> AgenticSysExperiment:
        """
        Generate experiment based on trace and hypothesis.
        
        Uses T() template system for task description generation.
        
        Args:
            trace: Experiment trace
            
        Returns:
            AgenticSysExperiment object
        """
        logger.info("Generating experiment from hypothesis...")
        
        # Step 1: Get hypothesis from trace
        hypothesis = self.get_latest_hypothesis(trace)
        
        # Step 2: Generate task description using T()
        task_desc = self.generate_task_description_with_t(hypothesis, trace)
        
        # Step 3: Create experiment
        main_task = Task(task_desc)
        experiment = AgenticSysExperiment(
            sub_tasks=[main_task]
        )
        
        # Step 4: Attach hypothesis and metadata
        if hypothesis:
            experiment.hypothesis = hypothesis.hypothesis
            experiment.action_type = getattr(hypothesis, 'action_type', 'Information_Gathering')
            experiment.target_dimensions = getattr(hypothesis, 'target_dimensions', [])
            experiment.implementation_plan = getattr(hypothesis, 'implementation_plan', {})
            experiment.hypothesis_obj = hypothesis
        else:
            experiment.hypothesis = "Baseline implementation"
            experiment.action_type = "Information_Gathering"
        
        logger.info(f"Generated experiment with action type: {experiment.action_type}")
        
        return experiment

    def get_latest_hypothesis(self, trace: Trace) -> Optional[Hypothesis]:
        """Get the latest hypothesis from trace"""
        if hasattr(trace, 'hypothesis') and trace.hypothesis:
            return trace.hypothesis
        
        if hasattr(trace, 'hist') and trace.hist:
            last_exp, _ = trace.hist[-1]
            if hasattr(last_exp, 'hypothesis_obj'):
                return last_exp.hypothesis_obj
        
        return None

    def generate_task_description_with_t(
        self, 
        hypothesis: Optional[Hypothesis], 
        trace: Trace
    ) -> str:
        """
        Generate task description using T() template system.
        
        KEY METHOD: Shows how to use action-specific prompts with T().
        
        Args:
            hypothesis: Hypothesis object
            trace: Experiment trace
            
        Returns:
            Task description string
        """
        is_first_experiment = not (hasattr(trace, 'hist') and trace.hist)
        
        # First experiment: baseline task
        if is_first_experiment:
            return self.get_baseline_task()
        
        # No hypothesis: fallback
        if not hypothesis:
            return self.get_improvement_task_fallback(trace)
        
        # Generate task based on action type using T()
        action_type = getattr(hypothesis, 'action_type', 'Information_Gathering')
        
        try:
            # Use T() to render action-specific specification
            action_spec = T(f"scenarios.agentic_sys.prompts:hypothesis_specification.{action_type}").r()
            
            # Build complete task description
            task_desc = f"""Action: {action_type}

Hypothesis: {hypothesis.hypothesis}

Target Dimensions:
{self.format_target_dimensions(getattr(hypothesis, 'target_dimensions', []))}

Implementation Plan:
{self.format_implementation_plan(getattr(hypothesis, 'implementation_plan', {}))}

====== Action-Specific Guidelines ======
{action_spec}

====== Success Criteria ======
{self.format_success_criteria(getattr(hypothesis, 'success_criteria', {}))}

====== Risk Assessment ======
{self.format_risk_assessment(getattr(hypothesis, 'risk_assessment', {}))}
"""
            return task_desc
            
        except Exception as e:
            logger.warning(f"Failed to use T() for action specification: {e}")
            return self._get_improvement_task_fallback(trace)

    # ==================== Formatting Helper Methods ====================
    
    def format_target_dimensions(self, target_dimensions: List[Dict]) -> str:
        """Format target dimensions"""
        if not target_dimensions:
            return "- No specific dimension targets"
        
        lines = []
        for dim in target_dimensions:
            name = dim.get('name', 'Unknown')
            current = dim.get('current_score', 'N/A')
            target = dim.get('target_score', 'N/A')
            improvement = dim.get('expected_improvement', 'N/A')
            confidence = dim.get('confidence', 'N/A')
            
            lines.append(f"- {name}: {current} → {target} (Δ{improvement}, confidence: {confidence})")
        
        return "\n".join(lines)

    def format_implementation_plan(self, plan: Dict) -> str:
        """Format implementation plan"""
        if not plan:
            return "- No specific implementation plan"
        
        lines = []
        for key, value in plan.items():
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)

    def format_success_criteria(self, criteria: Dict) -> str:
        """Format success criteria"""
        if not criteria:
            return "- Complete implementation without errors"
        
        lines = []
        
        primary = criteria.get('primary', None)
        if primary:
            lines.append(f"- Primary: {primary}")
        
        secondary = criteria.get('secondary', [])
        if secondary:
            lines.append("- Secondary:")
            for criterion in secondary:
                lines.append(f"  * {criterion}")
        
        validation = criteria.get('validation_method', None)
        if validation:
            lines.append(f"- Validation: {validation}")
        
        return "\n".join(lines) if lines else "- Complete implementation without errors"

    def format_risk_assessment(self, risk_assessment: Dict) -> str:
        """Format risk assessment"""
        if not risk_assessment:
            return "- No specific risks identified"
        
        lines = []
        
        negative_impacts = risk_assessment.get('potential_negative_impacts', [])
        if negative_impacts:
            lines.append("Potential Negative Impacts:")
            for impact in negative_impacts:
                if isinstance(impact, dict):
                    dimension = impact.get('dimension', 'Unknown')
                    reason = impact.get('reason', 'N/A')
                    severity = impact.get('severity', 'N/A')
                    lines.append(f"  - {dimension}: {reason} (Severity: {severity})")
                else:
                    lines.append(f"  - {impact}")
        
        mitigations = risk_assessment.get('mitigation_strategies', [])
        if mitigations:
            lines.append("\nMitigation Strategies:")
            for strategy in mitigations:
                lines.append(f"  - {strategy}")
        
        return "\n".join(lines) if lines else "- No specific risks identified"

    def get_baseline_task(self) -> str:
        """Get baseline task description for first experiment"""
        competition = getattr(self.scen, "competition", 'general') if self.scen else 'general'
        
        return f"""Design and implement a baseline agentic system for {competition}.

Requirements:
1. Create an AgenticSystem class for autonomous research task execution
2. Implement task execution with performance monitoring
3. Include metrics collection for DeepResearch dimensions:
   - Comprehensiveness, Insight, Instruction Following, Readability
4. Add error handling and logging
5. Output results in structured JSON format

Target Scores: Comprehensiveness ≥6.0, Insight ≥5.0, Instruction Following ≥7.0, Readability ≥6.0
"""

    def get_improvement_task_fallback(self, trace: Trace) -> str:
        """Fallback task generation when hypothesis unavailable"""
        if not hasattr(trace, 'hist') or not trace.hist:
            return self._get_baseline_task()
        
        last_exp, last_feedback = trace.hist[-1]
        
        decision = getattr(last_feedback, 'decision', None)
        base_desc = "Enhance successful system" if decision else "Fix issues in previous implementation"
        
        feedback_reason = getattr(last_feedback, 'reason', 'No feedback')[:200]
        
        return f"""{base_desc}

Previous feedback: {feedback_reason}

Focus on improving lowest-scoring dimension.

Current Scores:
{self.format_current_scores(last_feedback)}
"""

    def format_current_scores(self, feedback) -> str:
        """Format current dimension scores"""
        scores = {
            "Comprehensiveness": getattr(feedback, 'comprehensiveness_score', 'N/A'),
            "Insight": getattr(feedback, 'insight_score', 'N/A'),
            "Instruction Following": getattr(feedback, 'instruction_score', 'N/A'),
            "Readability": getattr(feedback, 'readability_score', 'N/A')
        }
        
        return "\n".join(f"- {dim}: {score}" for dim, score in scores.items())