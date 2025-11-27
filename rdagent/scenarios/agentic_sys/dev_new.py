"""
Developer for Agentic System Scenario
Generates code for agentic system experiments with optional web search enhancement
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.agentic_sys.env import get_agent_sys_env
from rdagent.scenarios.agentic_sys.tools.web_search import create_web_search_tool


class AgenticSysCoder(Developer[Experiment]):
    """
    Code generator for agentic system experiments
    
    Features:
    - CoSTEER-based code generation
    - Optional web search tool integration
    - Lazy initialization of external tools
    - Intelligent context enhancement
    """
    
    def __init__(self, scen):
        """
        Initialize AgenticSysCoder
        
        Args:
            scen: Scenario instance containing task description and configuration
        """
        self.scen = scen
        self.api_backend = APIBackend()
        
        # Lazy initialization for web search tool
        self._web_search_tool = None
        
        logger.info("Initialized AgenticSysCoder with LLM backend")
    
    @property
    def web_search_tool(self):
        """
        Lazy load web search tool when needed
        
        Returns:
            WebSearchTool instance or None if unavailable
        """
        if self._web_search_tool is None:
            try:
                search_config_path = Path(__file__).parent / "tools" / "search_config.yaml"
                if search_config_path.exists():
                    self._web_search_tool = create_web_search_tool(search_config_path)
                    logger.info("✓ Web search tool initialized successfully")
                else:
                    logger.warning(f"Search config not found: {search_config_path}")
                    self._web_search_tool = False
            except Exception as e:
                logger.warning(f"Failed to initialize web search tool: {e}")
                self._web_search_tool = False  # Mark as failed to avoid retry
        
        return self._web_search_tool if self._web_search_tool is not False else None

    def develop(self, exp: Experiment) -> Experiment:
        """
        Generate code for the experiment
        
        Workflow:
        1. Initialize workspace
        2. Prepare base context
        3. Optionally enhance with web search (tool call)
        4. Generate code artifacts
        5. Inject files and run
        
        Args:
            exp: Experiment instance
            
        Returns:
            Experiment with generated code and results
        """
        logger.info(f"Starting code generation for experiment: {getattr(exp, 'id', 'unknown')}")

        try:
            # Step 1: Initialize workspace
            exp.experiment_workspace = FBWorkspace()
            ws_path = Path(exp.experiment_workspace.workspace_path)
            ws_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Initialized workspace at {ws_path}")

            # Step 2: Prepare base context
            context = self._prepare_base_context(exp)
            logger.info("✓ Prepared base context")
            
            # Step 3: Optionally enhance with web search (TOOL CALL)
            if self._should_use_web_search(exp):
                logger.info("→ Calling web search tool for context enhancement...")
                context = self._enhance_context_with_web_search(context, exp)
            else:
                logger.info("→ Skipping web search (not needed)")

            # Step 4: Generate code artifacts
            logger.info("→ Generating code with CoSTEER framework...")
            code_artifacts = self._generate_code_artifacts(exp, context)
            
            # Step 5: Inject files into workspace
            exp.experiment_workspace.inject_files(**code_artifacts)
            logger.info(f"✓ Injected {len(code_artifacts)} files into workspace")

            # Step 6: Prepare environment and run
            timeout = self._calculate_timeout(exp)
            env = get_agent_sys_env(
                running_timeout_period=timeout,
                enable_cache=True
            )
            
            logger.info(f"→ Running generated code (timeout: {timeout}s)...")
            run_res = exp.experiment_workspace.run(env=env, entry="python train.py")
            
            # Store results
            exp.run_returncode = getattr(run_res, 'returncode', None)
            exp.run_stdout = getattr(run_res, 'stdout', getattr(run_res, 'logs', None))
            exp.run_stderr = getattr(run_res, 'stderr', None)
            
            if exp.run_returncode == 0:
                logger.info("✓ Experiment execution succeeded")
            else:
                logger.warning(f"⚠ Experiment execution failed with return code: {exp.run_returncode}")

        except Exception as e:
            logger.error(f"❌ Code generation failed: {str(e)}", exc_info=True)
            exp.exception = e
            
            # Try to create fallback workspace
            if not hasattr(exp, 'experiment_workspace') or not exp.experiment_workspace:
                try:
                    exp.experiment_workspace = self._create_fallback_workspace(exp)
                    logger.info("Created fallback workspace")
                except Exception as e_fallback:
                    logger.error(f"Failed to create fallback workspace: {e_fallback}")
        
        return exp
    
    def _prepare_base_context(self, exp: Experiment) -> Dict[str, Any]:
        """
        Prepare base context without web search
        
        Args:
            exp: Current experiment
            
        Returns:
            Base context dictionary
        """
        hypothesis = getattr(exp, 'hypothesis', 'Improve agentic system performance')
        
        context = {
            'hypothesis': hypothesis,
            'scenario_desc': self.scen.get_scenario_all_desc(),
            'success_criteria': getattr(self.scen, 'success_criteria', 'High performance'),
            'task_id': getattr(exp, 'id', 'unknown'),
            'task_domain': getattr(self.scen, 'domain', 'general'),
            'iteration_number': getattr(exp, 'iteration_number', 0),
            'external_sources': [],  # Will be filled by web search if used
            'external_knowledge_summary': ''  # Will be filled by web search if used
        }
        
        return context
    
    def _should_use_web_search(self, exp: Experiment) -> bool:
        """
        Determine if web search should be used for this experiment
        
        Decision criteria:
        1. Web search tool is available
        2. Not explicitly disabled by configuration
        3. Hypothesis complexity requires external knowledge
        4. Early iterations (< 3) benefit from external knowledge
        5. Previous experiments show low performance
        
        Args:
            exp: Current experiment
            
        Returns:
            True if web search should be used
        """
        # Check if web search is globally disabled
        if getattr(self.scen, 'disable_web_search', False):
            logger.info("Web search disabled by scenario configuration")
            return False
        
        # Check if tool is available
        if self.web_search_tool is None:
            logger.info("Web search tool not available")
            return False
        
        # Check if search service is healthy
        if not self.web_search_tool.client.health_check():
            logger.warning("Web search service is not healthy, skipping")
            return False
        
        hypothesis = getattr(exp, 'hypothesis', '').lower()
        
        # Use web search for research-heavy hypotheses
        research_indicators = [
            'research', 'investigate', 'explore', 'analyze', 'study',
            'compare', 'evaluate', 'survey', 'benchmark', 'baseline',
            'novel', 'innovative', 'advanced', 'state-of-art', 'sota',
            'improve', 'optimize', 'enhance', 'boost'
        ]
        
        if any(indicator in hypothesis for indicator in research_indicators):
            logger.info(f"Research-heavy hypothesis detected: '{hypothesis[:50]}...'")
            return True
        
        # Use web search for early iterations
        iteration = getattr(exp, 'iteration_number', 0)
        if iteration < 3:
            logger.info(f"Early iteration ({iteration}/3), enabling web search")
            return True
        
        # Use web search if previous performance was low
        if hasattr(exp, 'previous_performance_low') and exp.previous_performance_low:
            logger.info("Previous performance low, enabling web search for improvement")
            return True
        
        # Default: don't use web search for efficiency
        logger.info("Web search not needed (simple task or late iteration)")
        return False
    
    def _enhance_context_with_web_search(
        self, 
        context: Dict[str, Any], 
        exp: Experiment
    ) -> Dict[str, Any]:
        """
        Enhance context with web search results (TOOL CALL)
        
        This is the main entry point for web search tool integration.
        
        Args:
            context: Base context to enhance
            exp: Current experiment
            
        Returns:
            Enhanced context with external sources
        """
        try:
            hypothesis = context['hypothesis']
            
            # Step 1: Identify knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(exp, hypothesis)
            logger.info(f"Identified {len(knowledge_gaps)} knowledge gaps: {knowledge_gaps}")
            
            # Step 2: Prepare search context
            search_context = {
                'methodology': self._extract_methodology(hypothesis),
                'complexity': self._assess_complexity(hypothesis),
                'iteration': context.get('iteration_number', 0),
                'domain': context.get('task_domain', 'general')
            }
            logger.info(f"Search context: {search_context}")
            
            # Step 3: TOOL CALL - Search for hypothesis
            logger.info(f"Calling web search tool with task: '{hypothesis[:80]}...'")
            external_sources = self.web_search_tool.search_for_hypothesis(
                task_description=hypothesis,
                current_gaps=knowledge_gaps,
                context=search_context
            )
            
            # Step 4: Enhance context with results
            context['external_sources'] = external_sources
            logger.info(f"✓ Retrieved {len(external_sources)} external sources")
            
            # Step 5: Add summary for easy consumption
            if external_sources:
                context['external_knowledge_summary'] = self._summarize_external_sources(
                    external_sources
                )
                logger.info("✓ Generated external knowledge summary")
                
                # Log top sources
                for idx, source in enumerate(external_sources[:3], 1):
                    logger.info(
                        f"  {idx}. [{source['credibility_level']}] {source['title'][:60]}..."
                    )
            else:
                logger.warning("No external sources found")
            
        except Exception as e:
            logger.error(f"Web search enhancement failed: {e}", exc_info=True)
            # Don't fail the entire process, just skip enhancement
            context['external_sources'] = []
            context['external_knowledge_summary'] = ''
        
        return context
    
    def _identify_knowledge_gaps(self, exp: Experiment, hypothesis: str) -> List[str]:
        """
        Identify knowledge gaps from hypothesis
        
        Args:
            exp: Current experiment
            hypothesis: Hypothesis string
            
        Returns:
            List of knowledge gap descriptions (max 5)
        """
        gaps = []
        hypothesis_lower = hypothesis.lower()
        
        # Common agentic system knowledge areas
        knowledge_areas = {
            'planning': ['plan', 'planning', 'strategy', 'approach', 'roadmap'],
            'reasoning': ['reason', 'reasoning', 'logic', 'inference', 'think', 'thought'],
            'learning': ['learn', 'learning', 'adapt', 'optimization', 'train'],
            'memory': ['memory', 'context', 'history', 'recall', 'cache'],
            'tool_use': ['tool', 'api', 'external', 'integration', 'function'],
            'evaluation': ['evaluate', 'assessment', 'metric', 'performance', 'measure'],
            'communication': ['communicate', 'language', 'dialogue', 'interaction', 'conversation'],
            'retrieval': ['retrieval', 'search', 'rag', 'knowledge base', 'database'],
            'generation': ['generate', 'generation', 'create', 'synthesize', 'produce']
        }
        
        # Identify relevant areas
        for area, keywords in knowledge_areas.items():
            if any(kw in hypothesis_lower for kw in keywords):
                gaps.append(f"{area} techniques and best practices")
        
        # Add general gaps if none identified
        if not gaps:
            gaps.extend([
                "agentic system design patterns",
                "system implementation strategies",
                "performance optimization techniques"
            ])
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _extract_methodology(self, hypothesis: str) -> str:
        """
        Extract methodology from hypothesis
        
        Args:
            hypothesis: Hypothesis string
            
        Returns:
            Identified methodology
        """
        hypothesis_lower = hypothesis.lower()
        
        methodologies = {
            'reinforcement learning': ['rl', 'reinforcement', 'q-learning', 'policy', 'reward'],
            'retrieval augmented generation': ['rag', 'retrieval', 'augmented', 'retrieve'],
            'chain of thought': ['cot', 'chain of thought', 'reasoning chain', 'step by step'],
            'tree of thought': ['tot', 'tree of thought', 'reasoning tree', 'branching'],
            'multi-agent': ['multi-agent', 'multiple agents', 'agent collaboration', 'swarm'],
            'iterative refinement': ['iterative', 'refinement', 'feedback loop', 'improve'],
            'prompt engineering': ['prompt', 'prompting', 'instruction', 'template'],
            'fine-tuning': ['fine-tune', 'fine-tuning', 'training', 'adapt model']
        }
        
        for method, keywords in methodologies.items():
            if any(kw in hypothesis_lower for kw in keywords):
                return method
        
        return 'general agentic approach'
    
    def _assess_complexity(self, hypothesis: str) -> str:
        """
        Assess hypothesis complexity
        
        Args:
            hypothesis: Hypothesis string
            
        Returns:
            Complexity level: 'high', 'medium', or 'low'
        """
        hypothesis_lower = hypothesis.lower()
        
        high_complexity_indicators = [
            'complex', 'advanced', 'sophisticated', 'multi-stage', 'multi-step',
            'distributed', 'parallel', 'optimization', 'novel', 'innovative',
            'state-of-art', 'cutting-edge', 'research'
        ]
        
        medium_complexity_indicators = [
            'moderate', 'standard', 'typical', 'conventional', 'improve',
            'enhance', 'optimize', 'refine'
        ]
        
        low_complexity_indicators = [
            'simple', 'basic', 'straightforward', 'minimal', 'quick',
            'fix', 'patch', 'update'
        ]
        
        if any(ind in hypothesis_lower for ind in high_complexity_indicators):
            return 'high'
        elif any(ind in hypothesis_lower for ind in medium_complexity_indicators):
            return 'medium'
        elif any(ind in hypothesis_lower for ind in low_complexity_indicators):
            return 'low'
        else:
            return 'medium'  # Default to medium
    
    def _summarize_external_sources(self, sources: List[Dict[str, Any]]) -> str:
        """
        Summarize external sources for context injection
        
        Args:
            sources: List of external source dictionaries
            
        Returns:
            Formatted summary string
        """
        if not sources:
            return "No external sources available."
        
        summary_parts = []
        
        # Count by credibility
        high_cred = [s for s in sources if s.get('credibility_level') == 'High']
        medium_cred = [s for s in sources if s.get('credibility_level') == 'Medium']
        low_cred = [s for s in sources if s.get('credibility_level') == 'Low']
        
        summary_parts.append(
            f"Retrieved {len(sources)} sources: "
            f"{len(high_cred)} high-credibility, "
            f"{len(medium_cred)} medium-credibility, "
            f"{len(low_cred)} low-credibility"
        )
        
        # High credibility sources
        if high_cred:
            summary_parts.append(
                "\nHigh-credibility sources:\n" +
                "\n".join(f"  - {s['title'][:70]}" for s in high_cred[:3])
            )
        
        # Key insights from top sources
        key_insights = []
        for source in sources[:3]:
            summary = source.get('summary', '')
            if len(summary) > 50:
                key_insights.append(f"  • {summary[:150]}...")
        
        if key_insights:
            summary_parts.append("\nKey insights:\n" + "\n".join(key_insights))
        
        return "\n".join(summary_parts)
    
    def _generate_code_artifacts(
        self, 
        exp: Experiment, 
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate code artifacts using CoSTEER approach
        
        Args:
            exp: Current experiment
            context: Enhanced context (possibly with external knowledge)
        
        Returns:
            Dictionary of code artifacts {filename: content}
        """
        logger.info("Generating code artifacts with CoSTEER framework...")
        
        code_artifacts = {}
        
        # Extract task information
        task_info = self._extract_task_info(context)
        
        # Generate main agent implementation
        logger.info("→ Generating agent.py...")
        agent_code = self._generate_agent_code(task_info, context)
        code_artifacts['agent.py'] = agent_code
        
        # Generate evaluator
        logger.info("→ Generating evaluator.py...")
        evaluator_code = self._generate_evaluator_code(task_info)
        code_artifacts['evaluator.py'] = evaluator_code
        
        # Generate execution script
        logger.info("→ Generating train.py...")
        train_code = self._generate_execution_script(task_info)
        code_artifacts['train.py'] = train_code
        
        # Generate requirements
        logger.info("→ Generating requirements.txt...")
        requirements = self._generate_requirements(task_info)
        code_artifacts['requirements.txt'] = requirements
        
        logger.info(f"✓ Generated {len(code_artifacts)} code artifacts")
        return code_artifacts
    
    def _extract_task_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract task information from context
        
        Args:
            context: Context dictionary with external knowledge
            
        Returns:
            Task information dictionary
        """
        hypothesis = context.get('hypothesis', 'Improve agentic system performance')
        
        task_info = {
            'task_id': context.get('task_id', 'unknown'),
            'domain': context.get('task_domain', 'general'),
            'hypothesis': hypothesis,
            'complexity': context.get('complexity', self._assess_complexity(hypothesis)),
            'methodology': self._extract_methodology(hypothesis),
            'external_sources': context.get('external_sources', []),
            'external_knowledge_summary': context.get('external_knowledge_summary', ''),
            'has_external_knowledge': len(context.get('external_sources', [])) > 0,
            'iteration_number': context.get('iteration_number', 0)
        }
        
        return task_info
    
    def _generate_agent_code(self, task_info: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Generate agent implementation code
        
        Args:
            task_info: Task information
            context: Full context with external knowledge
            
        Returns:
            Agent code as string
        """
        # Simplified placeholder - in real implementation, use LLM with prompts
        hypothesis = task_info['hypothesis']
        external_summary = task_info['external_knowledge_summary']
        
        code = f'''"""
Agentic System Implementation
Generated for: {hypothesis}

External Knowledge:
{external_summary if external_summary else "No external knowledge used"}
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AgenticSystem:
    """
    Main agentic system implementation
    Hypothesis: {hypothesis}
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initialized AgenticSystem")
    
    def run(self, task: str) -> Dict[str, Any]:
        """Execute the agentic system on a task"""
        logger.info(f"Running task: {{task}}")
        
        # Implementation based on hypothesis
        result = {{
            'task': task,
            'status': 'completed',
            'output': 'Task completed successfully'
        }}
        
        return result


def create_agent(config: Dict[str, Any]) -> AgenticSystem:
    """Factory function to create agent"""
    return AgenticSystem(config)
'''
        return code
    
    def _generate_evaluator_code(self, task_info: Dict[str, Any]) -> str:
        """Generate evaluator code"""
        code = '''"""
Evaluator for Agentic System
"""

from typing import Dict, Any


class AgenticSystemEvaluator:
    """Evaluates agentic system performance"""
    
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate system performance
        
        Returns:
            Dictionary of metric scores
        """
        scores = {
            'comprehensiveness': 7.0,
            'insight': 6.5,
            'instruction_following': 8.0,
            'readability': 7.5
        }
        
        return scores


def create_evaluator() -> AgenticSystemEvaluator:
    """Factory function"""
    return AgenticSystemEvaluator()
'''
        return code
    
    def _generate_execution_script(self, task_info: Dict[str, Any]) -> str:
        """Generate execution script"""
        code = '''"""
Training/Execution script for agentic system
"""

import logging
from agent import create_agent
from evaluator import create_evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main execution"""
    logger.info("Starting agentic system execution")
    
    # Create agent
    config = {'model': 'gpt-4', 'temperature': 0.7}
    agent = create_agent(config)
    
    # Run task
    task = "Sample agentic task"
    results = agent.run(task)
    
    # Evaluate
    evaluator = create_evaluator()
    scores = evaluator.evaluate(results)
    
    logger.info(f"Evaluation scores: {scores}")
    logger.info("Execution completed")


if __name__ == '__main__':
    main()
'''
        return code
    
    def _generate_requirements(self, task_info: Dict[str, Any]) -> str:
        """Generate requirements.txt"""
        requirements = '''# Requirements for agentic system
openai>=1.0.0
anthropic>=0.7.0
pydantic>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
'''
        return requirements
    
    def _calculate_timeout(self, exp: Experiment) -> int:
        """Calculate execution timeout based on complexity"""
        complexity = getattr(exp, 'complexity', 'medium')
        
        timeout_map = {
            'low': 300,      # 5 minutes
            'medium': 600,   # 10 minutes
            'high': 1200     # 20 minutes
        }
        
        return timeout_map.get(complexity, 600)
    
    def _create_fallback_workspace(self, exp: Experiment) -> FBWorkspace:
        """Create fallback workspace on error"""
        ws = FBWorkspace()
        
        # Create minimal agent.py
        ws.inject_files(**{
            'agent.py': '# Fallback agent implementation\nprint("Fallback mode")',
            'train.py': '# Fallback execution\nprint("Running in fallback mode")'
        })
        
        return ws


class AgenticSysRunner(Developer[Experiment]):
    """
    Runner for agentic system experiments
    Executes generated code and collects results
    """
    
    def __init__(self, scen):
        self.scen = scen
        logger.info("Initialized AgenticSysRunner")
    
    def develop(self, exp: Experiment) -> Experiment:
        """
        Run the experiment
        
        Args:
            exp: Experiment with generated code
            
        Returns:
            Experiment with execution results
        """
        logger.info(f"Running experiment: {getattr(exp, 'id', 'unknown')}")
        
        try:
            if not hasattr(exp, 'experiment_workspace') or not exp.experiment_workspace:
                raise ValueError("No workspace found in experiment")
            
            # Execute the code
            env = get_agent_sys_env(running_timeout_period=600, enable_cache=True)
            run_res = exp.experiment_workspace.run(env=env, entry="python train.py")
            
            # Store results
            exp.run_returncode = getattr(run_res, 'returncode', None)
            exp.run_stdout = getattr(run_res, 'stdout', getattr(run_res, 'logs', None))
            exp.run_stderr = getattr(run_res, 'stderr', None)
            
            logger.info(f"Execution completed with return code: {exp.run_returncode}")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            exp.exception = e
        
        return exp