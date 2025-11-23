"""
DeepResearch Bench Evaluator for Agentic System
Implements 4-dimension evaluation: Comprehensiveness, Insight, Instruction Following, Readability
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Score for a single dimension"""
    score: float  # 0-10
    checks: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result with all dimensions"""
    comprehensiveness: DimensionScore
    insight: DimensionScore
    instruction_following: DimensionScore
    readability: DimensionScore
    overall: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    normalized_scores: Optional[Dict[str, float]] = None
    
    def __init__(self):
        if not self.weights:
            # Default equal weights
            self.weights = {
                'comprehensiveness': 0.25,
                'insight': 0.25,
                'instruction_following': 0.25,
                'readability': 0.25
            }
        self.calculate_overall()
    
    def calculate_overall(self) -> float:
        """Calculate weighted overall score"""
        self.overall = (
            self.comprehensiveness.score * self.weights['comprehensiveness'] +
            self.insight.score * self.weights['insight'] +
            self.instruction_following.score * self.weights['instruction_following'] +
            self.readability.score * self.weights['readability']
        )
        return self.overall
    
    def normalize_against_reference(self, reference: 'EvaluationResult') -> Dict[str, float]:
        """
        Pairwise normalization: target_normalized = target_score / (target_score + reference_score)
        """
        normalized = {}
        
        dimensions = ['comprehensiveness', 'insight', 'instruction_following', 'readability']
        for dim in dimensions:
            target_score = getattr(self, dim).score
            ref_score = getattr(reference, dim).score
            total = target_score + ref_score
            normalized[dim] = target_score / total if total > 0 else 0.5
        
        # Normalize overall score
        total_overall = self.overall + reference.overall
        normalized['overall'] = self.overall / total_overall if total_overall > 0 else 0.5
        
        self.normalized_scores = normalized
        return normalized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'scores': {
                'comprehensiveness': round(self.comprehensiveness.score, 2),
                'insight': round(self.insight.score, 2),
                'instruction_following': round(self.instruction_following.score, 2),
                'readability': round(self.readability.score, 2),
                'overall': round(self.overall, 2)
            },
            'weights': self.weights,
            'details': {
                'comprehensiveness': {
                    'score': round(self.comprehensiveness.score, 2),
                    'checks': self.comprehensiveness.checks,
                    'details': self.comprehensiveness.details
                },
                'insight': {
                    'score': round(self.insight.score, 2),
                    'checks': self.insight.checks,
                    'details': self.insight.details
                },
                'instruction_following': {
                    'score': round(self.instruction_following.score, 2),
                    'checks': self.instruction_following.checks,
                    'details': self.instruction_following.details
                },
                'readability': {
                    'score': round(self.readability.score, 2),
                    'checks': self.readability.checks,
                    'details': self.readability.details
                }
            }
        }
        
        if self.normalized_scores:
            result['normalized_scores'] = self.normalized_scores
        
        return result


class DeepResearchEvaluator:
    """
    Evaluator implementing DeepResearch Bench 4-dimension rubric
    
    Scoring Anchors:
    - 0-2: Poor/Missing core elements
    - 4-6: Basic/Adequate with gaps
    - 6-8: Good/Complete coverage
    - 8-10: Excellent/Exhaustive
    """
    
    def __init__(self, dimension_weights: Optional[Dict[str, float]] = None):
        """
        Initialize evaluator with optional custom weights
        
        Args:
            dimension_weights: Custom weights for dimensions (must sum to 1.0)
        """
        self.weights = dimension_weights or {
            'comprehensiveness': 0.25,
            'insight': 0.25,
            'instruction_following': 0.25,
            'readability': 0.25
        }
        
        # Validate weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            for k in self.weights:
                self.weights[k] /= total
    
    def evaluate(
        self,
        output: Any,
        task_requirements: Optional[Dict[str, Any]] = None,
        task_context: Optional[Dict[str, Any]] = None,
        reference_result: Optional[EvaluationResult] = None
    ) -> EvaluationResult:
        """
        Evaluate output against DeepResearch Bench criteria
        
        Args:
            output: The agent's output to evaluate
            task_requirements: Task requirements and constraints
            task_context: Additional context about the task
            reference_result: Optional reference for normalization
            
        Returns:
            EvaluationResult with scores for all dimensions
        """
        task_requirements = task_requirements or {}
        task_context = task_context or {}
        
        # Evaluate each dimension
        comp_score = self._evaluate_comprehensiveness(output, task_requirements)
        insight_score = self._evaluate_insight(output, task_context)
        following_score = self._evaluate_instruction_following(output, task_requirements)
        read_score = self._evaluate_readability(output)
        
        # Create result
        result = EvaluationResult(
            comprehensiveness=comp_score,
            insight=insight_score,
            instruction_following=following_score,
            readability=read_score,
            weights=self.weights
        )
        
        # Normalize against reference if provided
        if reference_result:
            result.normalize_against_reference(reference_result)
        
        return result
    
    def evaluate_comprehensiveness(
        self, 
        output: Any, 
        requirements: Dict[str, Any]
    ) -> DimensionScore:
        """
        Evaluate Comprehensiveness (0-10)
        - Required subtopics coverage (0-3 pts)
        - Depth of analysis (0-3 pts)
        - Evidence and sources (0-2 pts)
        - Multiple perspectives (0-2 pts)
        """
        score = 0.0
        checks = []
        details = {}
        
        output_str = str(output).lower()
        
        # 1. Required subtopics coverage (0-3 pts)
        required_topics = requirements.get('required_topics', [])
        if required_topics:
            covered = sum(1 for topic in required_topics 
                         if topic.lower() in output_str)
            coverage_ratio = covered / len(required_topics)
            coverage_score = min(3.0, coverage_ratio * 3.0)
            score += coverage_score
            checks.append(f"Topic coverage: {covered}/{len(required_topics)} ({coverage_score:.1f}/3.0)")
            details['topic_coverage'] = {
                'required': len(required_topics),
                'covered': covered,
                'ratio': coverage_ratio
            }
        else:
            score += 2.0
            checks.append("No specific topic requirements (default 2.0/3.0)")
        
        # 2. Depth of analysis (0-3 pts)
        depth_indicators = {
            'detailed_analysis': 'detailed analysis' in output_str or 'in-depth' in output_str,
            'data_evidence': 'data' in output_str or 'evidence' in output_str,
            'substantial_content': len(str(output)) > 500,
            'methodology': 'methodology' in output_str or 'approach' in output_str
        }
        depth_score = sum(depth_indicators.values()) * 0.75
        score += depth_score
        checks.append(f"Depth indicators: {sum(depth_indicators.values())}/4 ({depth_score:.1f}/3.0)")
        details['depth_indicators'] = depth_indicators
        
        # 3. Evidence and sources (0-2 pts)
        evidence_score = 0.0
        if 'references' in output_str or 'sources' in output_str or 'citation' in output_str:
            evidence_score += 1.0
        if 'data' in output_str or 'statistics' in output_str or 'figure' in output_str:
            evidence_score += 1.0
        score += evidence_score
        checks.append(f"Evidence & sources: ({evidence_score:.1f}/2.0)")
        details['evidence_score'] = evidence_score
        
        # 4. Multiple perspectives (0-2 pts)
        perspective_keywords = [
            'advantage', 'disadvantage', 'trade-off', 'alternative',
            'limitation', 'consideration', 'pros', 'cons'
        ]
        perspectives_found = sum(1 for kw in perspective_keywords if kw in output_str)
        perspective_score = min(2.0, perspectives_found * 0.4)
        score += perspective_score
        checks.append(f"Multiple perspectives: {perspectives_found} keywords ({perspective_score:.1f}/2.0)")
        details['perspectives_found'] = perspectives_found
        
        final_score = min(10.0, score)
        checks.append(f"Total Comprehensiveness Score: {final_score:.2f}/10.0")
        
        return DimensionScore(score=final_score, checks=checks, details=details)
    
    def evaluate_insight(
        self,
        output: Any,
        context: Dict[str, Any]
    ) -> DimensionScore:
        """
        Evaluate Insight (0-10)
        - Causal reasoning (0-3 pts)
        - Quantified analysis (0-2 pts)
        - Non-obvious implications (0-3 pts)
        - Novel synthesis (0-2 pts)
        """
        score = 0.0
        checks = []
        details = {}
        
        output_str = str(output).lower()
        
        # 1. Causal reasoning (0-3 pts)
        causal_indicators = [
            'because', 'therefore', 'as a result', 'leads to',
            'causes', 'impacts', 'due to', 'consequently'
        ]
        causal_found = sum(1 for indicator in causal_indicators if indicator in output_str)
        causal_score = min(3.0, causal_found * 0.5)
        score += causal_score
        checks.append(f"Causal reasoning: {causal_found} indicators ({causal_score:.1f}/3.0)")
        details['causal_indicators'] = causal_found
        
        # 2. Quantified analysis (0-2 pts)
        has_numbers = any(char.isdigit() for char in str(output))
        metric_keywords = ['percent', '%', 'rate', 'ratio', 'metric', 'measure', 'score']
        has_metrics = any(kw in output_str for kw in metric_keywords)
        quant_score = (1.0 if has_numbers else 0.0) + (1.0 if has_metrics else 0.0)
        score += quant_score
        checks.append(f"Quantified analysis: numbers={has_numbers}, metrics={has_metrics} ({quant_score:.1f}/2.0)")
        details['quantification'] = {'has_numbers': has_numbers, 'has_metrics': has_metrics}
        
        # 3. Non-obvious implications (0-3 pts)
        insight_keywords = [
            'implication', 'insight', 'suggests', 'indicates',
            'reveals', 'unexpected', 'surprisingly', 'notable', 'interesting'
        ]
        insights_found = sum(1 for kw in insight_keywords if kw in output_str)
        implication_score = min(3.0, insights_found * 0.6)
        score += implication_score
        checks.append(f"Implications: {insights_found} keywords ({implication_score:.1f}/3.0)")
        details['insights_found'] = insights_found
        
        # 4. Novel synthesis (0-2 pts)
        synthesis_indicators = [
            'framework', 'model', 'synthesis', 'integration',
            'novel', 'innovative', 'unique', 'original'
        ]
        synthesis_found = sum(1 for kw in synthesis_indicators if kw in output_str)
        synthesis_score = min(2.0, synthesis_found * 0.5)
        score += synthesis_score
        checks.append(f"Novel synthesis: {synthesis_found} indicators ({synthesis_score:.1f}/2.0)")
        details['synthesis_indicators'] = synthesis_found
        
        final_score = min(10.0, score)
        checks.append(f"Total Insight Score: {final_score:.2f}/10.0")
        
        return DimensionScore(score=final_score, checks=checks, details=details)
    
    def evaluate_instruction_following(
        self,
        output: Any,
        requirements: Dict[str, Any]
    ) -> DimensionScore:
        """
        Evaluate Instruction Following (0-10)
        - Required sections present (0-4 pts)
        - Scope compliance (0-3 pts)
        - Format compliance (0-2 pts)
        - Completeness (0-1 pt)
        """
        score = 0.0
        checks = []
        details = {}
        
        output_str = str(output).lower()
        
        # 1. Required sections present (0-4 pts)
        required_sections = requirements.get('required_sections', [])
        if required_sections:
            present = sum(1 for section in required_sections 
                         if section.lower() in output_str)
            section_score = min(4.0, (present / len(required_sections)) * 4.0)
            score += section_score
            checks.append(f"Required sections: {present}/{len(required_sections)} ({section_score:.1f}/4.0)")
            details['sections'] = {
                'required': len(required_sections),
                'present': present
            }
        else:
            score += 3.0
            checks.append("No specific section requirements (default 3.0/4.0)")
        
        # 2. Scope compliance (0-3 pts)
        scope_violations = self._check_scope_violations(output, requirements)
        scope_score = max(0.0, 3.0 - len(scope_violations) * 0.5)
        score += scope_score
        if scope_violations:
            checks.append(f"Scope violations: {len(scope_violations)} ({scope_score:.1f}/3.0)")
            details['scope_violations'] = scope_violations
        else:
            checks.append("No scope violations (3.0/3.0)")
        
        # 3. Format compliance (0-2 pts)
        format_reqs = requirements.get('format', {})
        format_score = 2.0  # Default
        if format_reqs:
            format_type = format_reqs.get('type', '').lower()
            if format_type == 'json':
                try:
                    import json
                    json.loads(str(output))
                    format_score = 2.0
                except:
                    format_score = 0.5
            checks.append(f"Format compliance: {format_type} ({format_score:.1f}/2.0)")
        else:
            checks.append("Format compliance: (2.0/2.0)")
        score += format_score
        details['format_score'] = format_score
        
        # 4. Completeness (0-1 pt)
        length = len(str(output))
        completeness_score = 1.0 if length > 200 else 0.5
        score += completeness_score
        checks.append(f"Completeness: {length} chars ({completeness_score:.1f}/1.0)")
        details['output_length'] = length
        
        final_score = min(10.0, score)
        checks.append(f"Total Instruction Following Score: {final_score:.2f}/10.0")
        
        return DimensionScore(score=final_score, checks=checks, details=details)
    
    def evaluate_readability(self, output: Any) -> DimensionScore:
        """
        Evaluate Readability (0-10)
        - Structure and organization (0-3 pts)
        - Language quality (0-3 pts)
        - Data presentation (0-2 pts)
        - Clarity (0-2 pts)
        """
        score = 0.0
        checks = []
        details = {}
        
        output_str = str(output)
        output_lower = output_str.lower()
        
        # 1. Structure and organization (0-3 pts)
        structure_indicators = {
            'has_breaks': '\n' in output_str,
            'has_sections': any(word in output_lower for word in 
                               ['summary', 'introduction', 'conclusion', 'results', 'method']),
            'multi_paragraph': output_str.count('\n') > 5
        }
        structure_score = min(3.0, sum(structure_indicators.values()) * 1.0)
        score += structure_score
        checks.append(f"Structure: {sum(structure_indicators.values())}/3 indicators ({structure_score:.1f}/3.0)")
        details['structure'] = structure_indicators
        
        # 2. Language quality (0-3 pts)
        words = output_str.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            unique_ratio = len(set(output_lower.split())) / len(words) if words else 0
            
            language_score = 0.0
            if 4 < avg_word_length < 7:  # Reasonable word length
                language_score += 1.5
            if unique_ratio > 0.5:  # Vocabulary variety
                language_score += 1.5
            
            score += language_score
            checks.append(f"Language quality: avg_len={avg_word_length:.1f}, variety={unique_ratio:.2f} ({language_score:.1f}/3.0)")
            details['language'] = {
                'avg_word_length': avg_word_length,
                'unique_ratio': unique_ratio
            }
        else:
            checks.append("Language quality: no content (0.0/3.0)")
        
        # 3. Data presentation (0-2 pts)
        has_formatting = any(marker in output_str for marker in ['|', ':', '-', '*', '•'])
        has_structure = output_str.count('\n') > 3
        presentation_score = (1.0 if has_formatting else 0.0) + (1.0 if has_structure else 0.0)
        score += presentation_score
        checks.append(f"Data presentation: formatting={has_formatting}, structure={has_structure} ({presentation_score:.1f}/2.0)")
        details['presentation'] = {
            'has_formatting': has_formatting,
            'has_structure': has_structure
        }
        
        # 4. Clarity (0-2 pts)
        length = len(output_str)
        clarity_score = 2.0
        if length < 100:
            clarity_score = 0.5
        elif length > 5000:
            clarity_score = 1.5
        
        score += clarity_score
        checks.append(f"Clarity: length={length} chars ({clarity_score:.1f}/2.0)")
        details['clarity'] = {'length': length, 'score': clarity_score}
        
        final_score = min(10.0, score)
        checks.append(f"Total Readability Score: {final_score:.2f}/10.0")
        
        return DimensionScore(score=final_score, checks=checks, details=details)
    
    def _check_scope_violations(
        self,
        output: Any,
        requirements: Dict[str, Any]
    ) -> List[str]:
        """Check for scope violations"""
        violations = []
        output_lower = str(output).lower()
        
        # Check timeframe constraints
        timeframe = requirements.get('timeframe')
        if timeframe:
            excluded_periods = requirements.get('excluded_periods', [])
            for period in excluded_periods:
                if period.lower() in output_lower:
                    violations.append(f"Out-of-scope timeframe: {period}")
        
        # Check topic constraints
        excluded_topics = requirements.get('excluded_topics', [])
        for topic in excluded_topics:
            if topic.lower() in output_lower:
                violations.append(f"Out-of-scope topic: {topic}")
        
        return violations