
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
from rdagent.utils.agent.tpl import T



class AgenticSysHypothesisGen(HypothesisGen):
    """
    Generate hypothesis for agentic system improvements based on DeepResearch evaluation dimensions.
    Integrates with prompts.yaml for structured hypothesis generation.
    """

    def __init__(self, scen: AgenticSysScen):
        super().__init__(scen=scen)
        self.scen = scen
        
        # Load prompts from prompts.yaml
        prompt_path = Path(__file__).parent / "prompts.yaml"
        self.prompt_dict = Prompts(file_path=prompt_path)
        
        # Initialize LLM backend
        self.api_backend = APIBackend()
        
        logger.info("AgenticSysHypothesisGen initialized with prompts from prompts.yaml")

    def gen(self, trace: Trace) -> Hypothesis:
        """
        Generate hypothesis based on trace history and evaluation dimensions.
        
        Args:
            trace: Experiment trace containing history
            
        Returns:
            Hypothesis object with structured hypothesis data
        """
        logger.info("Generating hypothesis based on trace history...")
        
        # 1. Prepare context from trace
        context = self.prepare_context(trace)
        
        # 2. Generate hypothesis using LLM with prompts
        hypothesis_data = self.generate_hypothesis_with_llm(context, trace)
        
        # 3. Create and return Hypothesis object
        hypothesis = Hypothesis(
            hypothesis=hypothesis_data.get("hypothesis", ""),
            reason=hypothesis_data.get("rationale", ""),
            concise_reason=hypothesis_data.get("current_gap", ""),
            concise_observation=self._extract_concise_observation(trace),
            concise_justification=hypothesis_data.get("concise_knowledge", ""),
            concise_knowledge=hypothesis_data.get("concise_knowledge", "")
        )
        
        # Store additional structured data
        hypothesis.action_type = hypothesis_data.get("action", "Information_Gathering")
        hypothesis.target_dimensions = hypothesis_data.get("target_dimensions", [])
        hypothesis.implementation_plan = hypothesis_data.get("implementation_plan", {})
        hypothesis.risk_assessment = hypothesis_data.get("risk_assessment", {})
        hypothesis.success_criteria = hypothesis_data.get("success_criteria", {})
        
        logger.info(f"Generated hypothesis: {hypothesis.hypothesis}")
        logger.info(f"Action type: {hypothesis.action_type}")
        logger.info(f"Target dimensions: {[d['name'] for d in hypothesis.target_dimensions]}")
        
        return hypothesis


    def _prepare_context(self, trace: Trace) -> Dict[str, Any]:
        """
        Prepare context for hypothesis generation from trace history.
        
        Similar to data_science scenario's context preparation.
        
        Args:
            trace: Experiment trace
            
        Returns:
            Dictionary with context information
        """
        context = {
            "is_first_experiment": not (hasattr(trace, 'hist') and trace.hist),
            "current_system_description": self._get_system_description(trace),
            "experiment_history": self._format_experiment_history(trace),
            "performance_gaps": self._identify_performance_gaps(trace),
            "current_scores": self._extract_current_scores(trace),
        }
        
        return context

    def _prepare_rag_context(self, trace: Trace) -> Dict[str, Any]:
        """
        Prepare RAG (Retrieval-Augmented Generation) context.
        
        Similar to: data_science/proposal/exp_gen/base.py knowledge retrieval
        
        Args:
            trace: Experiment trace
            
        Returns:
            Dictionary with RAG context
        """
        rag_context = {
            "insights": self._retrieve_cross_task_insights(),
            "experiences": self._retrieve_current_task_experiences(trace),
            "external_sources": self._retrieve_external_sources(trace)
        }
        
        return rag_context

    def _get_system_description(self, trace: Trace) -> str:
        """Get current system description from trace."""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "No previous system implementation. Starting from baseline."
        
        last_exp, last_feedback = trace.hist[-1]
        
        description = f"Current system status:\n"
        description += f"- Last hypothesis: {getattr(last_exp, 'hypothesis', 'N/A')}\n"
        description += f"- Last feedback: {getattr(last_feedback, 'reason', 'N/A')[:200]}\n"
        description += f"- Success rate: {self._calculate_success_rate(trace):.1%}\n"
        
        return description

    def _format_experiment_history(self, trace: Trace) -> str:
        """
        Format experiment history for prompt.
        
        Uses: prompts.yaml::hypothesis_and_feedback template
        """
        if not hasattr(trace, 'hist') or not trace.hist:
            return "No previous experiments."
        
        # Try to use the hypothesis_and_feedback template from prompts.yaml
        try:
            # Calculate pattern analysis
            pattern_analysis = self._analyze_patterns(trace)
            
            history_prompt = self.prompt_dict["hypothesis_and_feedback"]["user"].render(
                trace=trace,
                history_window=10,
                most_successful_action=pattern_analysis.get("most_successful_action", "N/A"),
                most_improved_dimension=pattern_analysis.get("most_improved_dimension", "N/A"),
                persistent_weaknesses=pattern_analysis.get("persistent_weaknesses", "N/A"),
                effective_strategies=pattern_analysis.get("effective_strategies", "N/A")
            )
            return history_prompt
        except Exception as e:
            logger.warning(f"Failed to render hypothesis_and_feedback template: {e}")
            
            # Fallback: manual formatting
            history = []
            for idx, (exp, feedback) in enumerate(trace.hist[-5:], 1):
                entry = f"Experiment {idx}:\n"
                entry += f"  Hypothesis: {getattr(exp, 'hypothesis', 'N/A')}\n"
                entry += f"  Decision: {'Success' if getattr(feedback, 'decision', False) else 'Failed'}\n"
                entry += f"  Reason: {getattr(feedback, 'reason', 'N/A')[:100]}\n"
                history.append(entry)
            
            return "\n".join(history)

    def _analyze_patterns(self, trace: Trace) -> Dict[str, str]:
        """
        Analyze patterns from trace history.
        
        Returns:
            Dictionary with pattern analysis
        """
        if not hasattr(trace, 'hist') or not trace.hist:
            return {
                "most_successful_action": "N/A",
                "most_improved_dimension": "N/A",
                "persistent_weaknesses": "N/A",
                "effective_strategies": "N/A"
            }
        
        # Count action type success
        action_success = {}
        dimension_improvements = {
            "comprehensiveness": 0,
            "insight": 0,
            "instruction_following": 0,
            "readability": 0
        }
        
        for exp, feedback in trace.hist:
            action_type = getattr(exp, 'action_type', 'Unknown')
            if getattr(feedback, 'decision', False):
                action_success[action_type] = action_success.get(action_type, 0) + 1
            
            # Track dimension improvements
            for dim in dimension_improvements.keys():
                delta_attr = f"{dim}_delta"
                if hasattr(feedback, delta_attr):
                    delta = getattr(feedback, delta_attr, 0)
                    if delta > 0:
                        dimension_improvements[dim] += delta
        
        # Find most successful action
        most_successful_action = max(action_success, key=action_success.get) if action_success else "N/A"
        
        # Find most improved dimension
        most_improved_dimension = max(dimension_improvements, key=dimension_improvements.get)
        
        # Identify persistent weaknesses
        weaknesses = []
        if hasattr(trace, 'hist') and trace.hist:
            _, last_feedback = trace.hist[-1]
            for dim, improvement in dimension_improvements.items():
                if improvement <= 0:
                    score_attr = f"{dim}_score"
                    if hasattr(last_feedback, score_attr):
                        score = getattr(last_feedback, score_attr, 0)
                        if score < 6.0:
                            weaknesses.append(f"{dim} (score: {score})")
        
        persistent_weaknesses = ", ".join(weaknesses) if weaknesses else "None identified"
        
        # Effective strategies
        effective_strategies = f"{most_successful_action} action type has been most successful"
        
        return {
            "most_successful_action": most_successful_action,
            "most_improved_dimension": most_improved_dimension,
            "persistent_weaknesses": persistent_weaknesses,
            "effective_strategies": effective_strategies
        }

    def _identify_performance_gaps(self, trace: Trace) -> str:
        """Identify performance gaps from trace history."""
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
            
            # Extract common failure patterns
            failure_reasons = [
                getattr(fb, 'reason', '') for _, fb in failed_experiments
            ]
            if any('error' in reason.lower() for reason in failure_reasons):
                gaps.append("- Error handling needs improvement")
            if any('time' in reason.lower() for reason in failure_reasons):
                gaps.append("- Performance optimization needed")
        
        # Check success rate
        success_rate = self._calculate_success_rate(trace)
        if success_rate < 0.5:
            gaps.append(f"- Low success rate ({success_rate:.1%}) requires fundamental improvements")
        elif success_rate < 0.8:
            gaps.append(f"- Moderate success rate ({success_rate:.1%}) suggests refinement opportunities")
        
        return "\n".join(gaps) if gaps else "System performing well. Focus on advanced optimizations."

    def _extract_current_scores(self, trace: Trace) -> Dict[str, Optional[float]]:
        """Extract current dimension scores from latest feedback."""
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

    def _calculate_success_rate(self, trace: Trace) -> float:
        """Calculate success rate from trace history."""
        if not hasattr(trace, 'hist') or not trace.hist:
            return 0.0
        
        success_count = sum(
            1 for _, fb in trace.hist
            if getattr(fb, 'decision', False)
        )
        
        return success_count / len(trace.hist)

    def _extract_concise_observation(self, trace: Trace) -> str:
        """Extract concise observation from trace."""
        if not hasattr(trace, 'hist') or not trace.hist:
            return "Starting baseline implementation"
        
        _, last_feedback = trace.hist[-1]
        observations = getattr(last_feedback, 'observations', '')
        
        # Extract first sentence or first 100 chars
        if observations:
            first_sentence = observations.split('.')[0]
            return first_sentence[:100] + "..." if len(first_sentence) > 100 else first_sentence
        
        return "Previous experiment completed"

    def _retrieve_cross_task_insights(self) -> List[Dict[str, Any]]:
        """
        Retrieve insights from other similar tasks (cross-task knowledge).
        
        TODO: Implement actual knowledge base retrieval
        Similar to: data_science scenario's knowledge base
        """
        # Placeholder for RAG implementation
        return []

    def _retrieve_current_task_experiences(self, trace: Trace) -> List[Dict[str, Any]]:
        """
        Retrieve relevant experiences from current task's trace history.
        
        Args:
            trace: Experiment trace
            
        Returns:
            List of experience dictionaries
        """
        if not hasattr(trace, 'hist') or not trace.hist:
            return []
        
        experiences = []
        for exp, fb in trace.hist[-5:]:  # Last 5 experiences
            experiences.append({
                "hypothesis": getattr(exp, 'hypothesis', 'N/A'),
                "approach": getattr(exp, 'action_type', 'N/A') if hasattr(exp, 'action_type') else 'N/A',
                "improved_dims": self._extract_improved_dimensions(fb),
                "lessons": getattr(fb, 'reason', 'N/A')[:200]
            })
        
        return experiences

    def _retrieve_external_sources(self, trace: Trace) -> List[Dict[str, Any]]:
        """
        Retrieve external sources (papers, documentation, etc.).
        
        TODO: Implement actual external source retrieval
        """
        # Placeholder for external source retrieval
        return []

    def _extract_improved_dimensions(self, feedback) -> List[str]:
        """Extract which dimensions improved from feedback."""
        improved = []
        
        if hasattr(feedback, 'comprehensiveness_delta') and getattr(feedback, 'comprehensiveness_delta', 0) > 0:
            improved.append("Comprehensiveness")
        if hasattr(feedback, 'insight_delta') and getattr(feedback, 'insight_delta', 0) > 0:
            improved.append("Insight")
        if hasattr(feedback, 'instruction_delta') and getattr(feedback, 'instruction_delta', 0) > 0:
            improved.append("Instruction Following")
        if hasattr(feedback, 'readability_delta') and getattr(feedback, 'readability_delta', 0) > 0:
            improved.append("Readability")
        
        return improved if improved else ["None"]

    def _generate_hypothesis_with_llm(
        self, 
        context: Dict[str, Any], 
        rag_context: Dict[str, Any],
        trace: Trace
    ) -> Dict[str, Any]:
        """
        Generate hypothesis using LLM with prompts from prompts.yaml.
        
        Similar to: data_science/proposal/exp_gen/base.py::_generate_hypothesis
        
        This is the CORE method that calls prompts.yaml templates.
        
        Args:
            context: Context dictionary
            rag_context: RAG context dictionary
            trace: Experiment trace
            
        Returns:
            Parsed hypothesis data dictionary
        """
        # Step 1: Build system prompt with task background
        system_prompt = self._build_system_prompt()
        
        # Step 2: Build user prompt with all components
        user_prompt = self._build_user_prompt(context, rag_context, trace)
        
        # Step 3: Call LLM (same as data_science scenario)
        logger.info("Calling LLM for hypothesis generation...")
        response = self.api_backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True  # Force JSON output
        )
        
        # Step 4: Parse JSON response
        try:
            hypothesis_data = json.loads(response)
            logger.info("Successfully parsed hypothesis JSON")
            return hypothesis_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse hypothesis JSON: {e}")
            logger.error(f"Response: {response}")
            
            # Fallback to simple hypothesis
            return self._get_fallback_hypothesis(context)

    def _build_system_prompt(self) -> str:
        """
        Build system prompt from prompts.yaml.
        
        Uses: prompts.yaml::task_background::system
        
        This method shows how to access nested prompts with system/user structure.
        """
        try:
            # Access the 'system' part of 'task_background'
            system_prompt = self.prompt_dict["task_background"]["system"]
            return system_prompt
        except Exception as e:
            logger.warning(f"Failed to get task_background system prompt: {e}")
            # Fallback system prompt
            return "You are an expert AI researcher specializing in agentic systems."

    def _build_user_prompt(
        self, 
        context: Dict[str, Any], 
        rag_context: Dict[str, Any],
        trace: Trace
    ) -> str:
        """
        Build user prompt combining multiple components from prompts.yaml.
        
        This method demonstrates the KEY pattern for using prompts.yaml:
        1. Access prompts using self.prompt_dict[prompt_name]
        2. For system/user structure: self.prompt_dict[prompt_name]["user"]
        3. Use .render() to substitute Jinja2 variables
        4. Combine multiple prompts into one user_prompt
        
        Components:
        1. task_background::user (task context)
        2. KG_hypothesis_gen_RAG::user (if RAG context available)
        3. hypothesis_generation::user (main generation instruction)
        4. hypothesis_output_format::user (output specification)
        
        Args:
            context: Context dictionary
            rag_context: RAG context dictionary
            trace: Experiment trace
            
        Returns:
            Complete user prompt string
        """
        prompt_parts = []
        
        # Part 0: Add task background (user part)
        try:
            task_bg = self.prompt_dict["task_background"]["user"].render(
                task_type=getattr(self.scen, 'task_type', 'Research Automation'),
                domain=getattr(self.scen, 'domain', 'Agentic Systems'),
                brief_description=getattr(self.scen, 'description', 'Automated research system'),
                scope_requirements=getattr(self.scen, 'scope', 'N/A'),
                required_deliverables=getattr(self.scen, 'deliverables', 'N/A'),
                comprehensiveness_focus=getattr(self.scen, 'comprehensiveness_focus', 'Complete task coverage'),
                insight_focus=getattr(self.scen, 'insight_focus', 'Deep causal analysis'),
                instruction_focus=getattr(self.scen, 'instruction_focus', 'Strict requirement adherence'),
                readability_focus=getattr(self.scen, 'readability_focus', 'Clear presentation')
            )
            prompt_parts.append(task_bg)
        except Exception as e:
            logger.warning(f"Failed to render task_background user prompt: {e}")
        
        # Part 1: Add RAG context if available
        if rag_context.get("insights") or rag_context.get("experiences") or rag_context.get("external_sources"):
            try:
                # Access the 'user' part and render with variables
                rag_prompt = self.prompt_dict["KG_hypothesis_gen_RAG"]["user"].render(
                    insights=rag_context.get("insights", []),
                    experiences=rag_context.get("experiences", []),
                    external_sources=rag_context.get("external_sources", [])
                )
                prompt_parts.append(rag_prompt)
            except Exception as e:
                logger.warning(f"Failed to render KG_hypothesis_gen_RAG: {e}")
        
        # Part 2: Add main hypothesis generation instruction
        try:
            # This is the CORE prompt - access 'user' part and render
            hypothesis_prompt = self.prompt_dict["hypothesis_generation"]["user"].render(
                current_system_description=context["current_system_description"],
                current_comprehensiveness=context["current_scores"]["comprehensiveness"],
                current_insight=context["current_scores"]["insight"],
                current_instruction_following=context["current_scores"]["instruction_following"],
                current_readability=context["current_scores"]["readability"],
                experiment_history=context["experiment_history"],
                performance_gaps=context["performance_gaps"]
            )
            prompt_parts.append(hypothesis_prompt)
        except Exception as e:
            logger.error(f"Failed to render hypothesis_generation: {e}")
            raise
        
        # Part 3: Add output format specification
        try:
            # Access the 'user' part for output format
            output_format = self.prompt_dict["hypothesis_output_format"]["user"]
            prompt_parts.append(output_format)
        except Exception as e:
            logger.warning(f"Failed to get hypothesis_output_format: {e}")
        
        # Part 4: Combine all parts with double newlines
        full_prompt = "\n\n".join(prompt_parts)
        
        return full_prompt

    def _get_fallback_hypothesis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get fallback hypothesis when LLM parsing fails.
        
        Args:
            context: Context dictionary
            
        Returns:
            Simple fallback hypothesis dictionary
        """
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
            "current_gap": "Unable to generate structured hypothesis due to LLM response parsing failure",
            "rationale": "LLM response parsing failed. Using fallback hypothesis for system improvement.",
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
            "concise_knowledge": "When LLM parsing fails, use fallback hypothesis with incremental improvements"
        }


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
        prompt_path = Path(__file__).parent / "prompts.yaml"
        self.prompt_dict = Prompts(file_path=prompt_path)

        #initialize LLM backend
        self.api_backend = APIBackend()
        logger.info("AgenticSysExpGen initialized.")
        
    def gen(self, trace: Trace) -> AgenticSysExperiment:
        """
        Generate experiment based on trace and hypothesis.
        
        Similar to: data_science/proposal/exp_gen/base.py::ModelExpGen.gen()
        
        Args:
            trace: Experiment trace
            
        Returns:
            AgenticSysExperiment object
        """
        logger.info("Generating experiment from hypothesis...")
        
        # Step 1: Get hypothesis from trace (should be generated by HypothesisGen)
        hypothesis = self._get_latest_hypothesis(trace)
        
        # Step 2: Generate task description based on hypothesis
        task_desc = self._generate_task_description(hypothesis, trace)
        
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
            experiment.hypothesis_obj = hypothesis  # Store full hypothesis object
        else:
            experiment.hypothesis = "Baseline implementation"
            experiment.action_type = "Information_Gathering"
        
        logger.info(f"Generated experiment with action type: {experiment.action_type}")
        
        return experiment

    def _get_latest_hypothesis(self, trace: Trace) -> Optional[Hypothesis]:
        """
        Get the latest hypothesis from trace.
        
        In RDAgent framework, hypothesis is typically stored in trace or passed separately.
        """
        # Option 1: Check if hypothesis is stored in trace
        if hasattr(trace, 'hypothesis') and trace.hypothesis:
            return trace.hypothesis
        
        # Option 2: Check last experiment in history
        if hasattr(trace, 'hist') and trace.hist:
            last_exp, _ = trace.hist[-1]
            if hasattr(last_exp, 'hypothesis_obj'):
                return last_exp.hypothesis_obj
        
        return None

    def _generate_task_description(self, hypothesis: Optional[Hypothesis], trace: Trace) -> str:
        """
        Generate task description based on hypothesis and action type.
        
        KEY METHOD: Shows how to use action-specific prompts from prompts.yaml
        
        Uses: prompts.yaml::hypothesis_specification::{action_type}::user
        
        Args:
            hypothesis: Hypothesis object (can be None for first experiment)
            trace: Experiment trace
            
        Returns:
            Task description string
        """
        is_first_experiment = not (hasattr(trace, 'hist') and trace.hist)
        
        # First experiment: baseline task
        if is_first_experiment:
            return self._get_baseline_task()
        
        # No hypothesis available: fallback
        if not hypothesis:
            return self._get_improvement_task_fallback(trace)
        
        # Generate task based on action type using prompts.yaml
        action_type = getattr(hypothesis, 'action_type', 'Information_Gathering')
        
        # Try to get action-specific specification from prompts
        try:
            # KEY PATTERN: Access nested prompts with action type
            # prompts.yaml structure:
            # hypothesis_specification:
            #   Information_Gathering:
            #     system: ...
            #     user: ...
            
            if "hypothesis_specification" in self.prompt_dict:
                if action_type in self.prompt_dict["hypothesis_specification"]:
                    # Get the 'user' part of the action specification
                    action_spec_user = self.prompt_dict["hypothesis_specification"][action_type]["user"]
                    
                    # Build complete task description
                    task_desc = f"""Action: {action_type}

Hypothesis: {hypothesis.hypothesis}

Target Dimensions:
{self._format_target_dimensions(getattr(hypothesis, 'target_dimensions', []))}

Implementation Plan:
{self._format_implementation_plan(getattr(hypothesis, 'implementation_plan', {}))}

====== Action-Specific Guidelines ======
{action_spec_user}

====== Success Criteria ======
{self._format_success_criteria(getattr(hypothesis, 'success_criteria', {}))}

====== Risk Assessment ======
{self._format_risk_assessment(getattr(hypothesis, 'risk_assessment', {}))}
"""
                    return task_desc
        except Exception as e:
            logger.warning(f"Failed to use action specification from prompts.yaml: {e}")
        
        # Fallback to simple task description
        return self._get_improvement_task_fallback(trace)

    def _format_target_dimensions(self, target_dimensions: List[Dict]) -> str:
        """Format target dimensions for task description."""
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

    def _format_implementation_plan(self, plan: Dict) -> str:
        """Format implementation plan for task description."""
        if not plan:
            return "- No specific implementation plan"
        
        lines = []
        for key, value in plan.items():
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)

    def _format_success_criteria(self, criteria: Dict) -> str:
        """Format success criteria for task description."""
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

    def _format_risk_assessment(self, risk_assessment: Dict) -> str:
        """Format risk assessment for task description."""
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

    def _get_baseline_task(self) -> str:
        """
        Get baseline task description for first experiment.
        
        Could also be loaded from prompts.yaml if needed.
        
        Returns:
            Baseline task description string
        """
        competition = getattr(self.scen, "competition", 'general') if self.scen else 'general'
        
        return f"""Design and implement a baseline agentic system for {competition}.

Requirements:
1. Create an AgenticSystem class that can execute research tasks autonomously
2. Implement task execution with performance monitoring
3. Include metrics collection for all four DeepResearch dimensions:
   - Comprehensiveness: Coverage breadth and depth
   - Insight: Causal reasoning and originality
   - Instruction Following: Requirement adherence
   - Readability: Clarity and presentation
4. Add proper error handling and logging
5. Output results in structured JSON format

The system should demonstrate:
- Information gathering and synthesis
- Task planning and execution
- Basic error recovery
- Performance measurement
- Clean code structure

Evaluation Focus:
- Comprehensiveness: Complete task coverage
- Insight: Basic causal reasoning
- Instruction Following: Meet all requirements
- Readability: Clear code and documentation

Target Scores (0-10 scale):
- Comprehensiveness: ≥ 6.0
- Insight: ≥ 5.0
- Instruction Following: ≥ 7.0
- Readability: ≥ 6.0
"""

    def _get_improvement_task_fallback(self, trace: Trace) -> str:
        """
        Fallback task generation when hypothesis is not available.
        
        Args:
            trace: Experiment trace
            
        Returns:
            Improvement task description
        """
        if not hasattr(trace, 'hist') or not trace.hist:
            return self._get_baseline_task()
        
        last_exp, last_feedback = trace.hist[-1]
        
        decision = getattr(last_feedback, 'decision', None)
        if decision is True:
            base_desc = "Enhance the successful agentic system from the previous experiment."
        elif decision is False:
            base_desc = "Fix the issues in the previous agentic system implementation."
        else:
            base_desc = "Review the previous experiment and address any uncovered issues."
        
        feedback_reason = getattr(last_feedback, 'reason', 'No specific feedback')[:200]
        
        competition = getattr(self.scen, 'competition', 'general') if self.scen else 'general'
        
        return f"""{base_desc}

Competition: {competition}
Previous feedback: {feedback_reason}

Improvement requirements:
1. Analyze the previous implementation and identify bottlenecks
2. Implement specific optimizations based on the feedback
3. Maintain or improve the current performance metrics on all four dimensions
4. Add new features or capabilities as needed
5. Focus on the weakest evaluation dimension

Ensure backwards compatibility while introducing improvements.

Current Dimension Scores:
{self._format_current_scores(last_feedback)}

Prioritize improvements in the lowest-scoring dimension.
"""

    def _format_current_scores(self, feedback) -> str:
        """Format current dimension scores from feedback."""
        scores = {
            "Comprehensiveness": getattr(feedback, 'comprehensiveness_score', 'N/A'),
            "Insight": getattr(feedback, 'insight_score', 'N/A'),
            "Instruction Following": getattr(feedback, 'instruction_score', 'N/A'),
            "Readability": getattr(feedback, 'readability_score', 'N/A')
        }
        
        lines = []
        for dim, score in scores.items():
            lines.append(f"- {dim}: {score}")
        
        return "\n".join(lines)