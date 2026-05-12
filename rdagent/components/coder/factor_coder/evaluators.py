import ast
import re

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedbackDeprecated,
)
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorCodeEvaluator,
    FactorFinalDecisionEvaluator,
    FactorValueEvaluator,
)
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Workspace

FactorSingleFeedback = CoSTEERSingleFeedbackDeprecated

# Identifier hints used to recognise an outer ``for`` loop that iterates over
# stocks / instruments / tickers / symbols / codes.
_INSTRUMENT_LOOP_HINTS = ("instrument", "stock", "ticker", "symbol", "code")

# Substrings used to recognise an ML estimator constructor or training call
# inside a nested loop body.  Matched against ``ast.unparse`` of the inner
# ``for`` body, so they catch both ``model.fit(...)`` style calls and direct
# instantiation of common estimators.
_ML_TRAINING_PATTERNS = re.compile(
    r"\b(?:"
    r"\.fit\s*\(|"
    r"\.partial_fit\s*\(|"
    r"\.train\s*\(|"
    r"train_[a-z_]+\s*\(|"
    r"LSTM\s*\(|GRU\s*\(|RNN\s*\(|Transformer\s*\(|"
    r"RandomForest\w*\s*\(|XGB\w+\s*\(|LGBM\w*\s*\(|CatBoost\w*\s*\(|"
    r"GradientBoosting\w*\s*\(|SVR\s*\(|SVC\s*\(|"
    r"MLPRegressor\s*\(|MLPClassifier\s*\(|Sequential\s*\("
    r")"
)

_PER_INSTRUMENT_TRAINING_FEEDBACK = (
    "Performance-critical anti-pattern detected: the factor code wraps an ML estimator "
    "construction or training call inside a nested loop over instruments and time. With "
    "N instruments x T trading days this produces O(N * T) training iterations and the "
    "run hangs at 100% CPU for hours on realistic A-share panels (see issue #1407). "
    "Fit the estimator exactly once on the full (datetime, instrument) panel and "
    "batch-predict for every row in a single call, or restrict ML usage to vectorized "
    "rolling closed-form estimators that can be expressed via "
    "groupby(level='instrument').rolling(...).apply(...) or pandas/numpy operations. "
    "Do not re-instantiate or re-train the model per stock or per date."
)


def _identifier_text(node: ast.AST) -> str:
    """Return a lowercased text snippet for ``node`` suitable for hint matching."""

    try:
        return ast.unparse(node).lower()
    except Exception:  # pragma: no cover - defensive, ast.unparse is stable on >=3.9
        return ""


def _body_source(nodes: list[ast.stmt]) -> str:
    parts: list[str] = []
    for node in nodes:
        try:
            parts.append(ast.unparse(node))
        except Exception:  # pragma: no cover - defensive
            continue
    return "\n".join(parts)


def detect_per_instrument_training_antipattern(code: str) -> str | None:
    """Detect the nested per-instrument / per-day ML training anti-pattern.

    Returns a critic-style feedback string if the anti-pattern is present in
    ``code``, otherwise ``None``.  Callers can use the message verbatim as
    code feedback so the LLM gets actionable repair guidance without paying
    for a multi-hour execution attempt first.
    """

    if not code:
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax issues are surfaced through the normal execution path.
        return None

    for outer in ast.walk(tree):
        if not isinstance(outer, (ast.For, ast.AsyncFor)):
            continue

        outer_target = _identifier_text(outer.target)
        outer_iter = _identifier_text(outer.iter)
        if not any(hint in outer_target or hint in outer_iter for hint in _INSTRUMENT_LOOP_HINTS):
            continue

        for inner in ast.walk(outer):
            if inner is outer or not isinstance(inner, (ast.For, ast.AsyncFor)):
                continue
            inner_body_source = _body_source(list(inner.body))
            if _ML_TRAINING_PATTERNS.search(inner_body_source):
                return _PER_INSTRUMENT_TRAINING_FEEDBACK

    return None


class FactorEvaluatorForCoder(CoSTEEREvaluator):
    """This class is the v1 version of evaluator for a single factor implementation.
    It calls several evaluators in share modules to evaluate the factor implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value_evaluator = FactorValueEvaluator(self.scen)
        self.code_evaluator = FactorCodeEvaluator(self.scen)
        self.final_decision_evaluator = FactorFinalDecisionEvaluator(self.scen)

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback:
        if implementation is None:
            return None

        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return FactorSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                value_generated_flag=False,
                code_feedback="This task has failed too many times, skip code evaluation.",
                value_feedback="This task has failed too many times, skip value evaluation.",
                final_decision=False,
                final_feedback="This task has failed too many times, skip final decision evaluation.",
                final_decision_based_on_gt=False,
            )
        else:
            factor_feedback = FactorSingleFeedback()

            # Pre-execution static check for the per-instrument / per-day ML
            # training anti-pattern (issue #1407).  On realistic stock panels
            # this anti-pattern hangs ``implementation.execute()`` for hours,
            # so short-circuit with critic-style feedback before paying that
            # cost and let CoSTEER repair using the guidance instead.
            anti_pattern_feedback = detect_per_instrument_training_antipattern(implementation.all_codes)
            if anti_pattern_feedback is not None:
                factor_feedback.execution_feedback = anti_pattern_feedback
                factor_feedback.value_generated_flag = False
                factor_feedback.value_feedback = "No factor value generated, skip value evaluation."
                factor_feedback.code_feedback = anti_pattern_feedback
                factor_feedback.final_decision = False
                factor_feedback.final_feedback = anti_pattern_feedback
                factor_feedback.final_decision_based_on_gt = gt_implementation is not None
                return factor_feedback

            # 1. Get factor execution feedback to generated implementation and remove the long list of numbers in execution feedback
            (
                execution_feedback,
                gen_df,
            ) = implementation.execute()

            execution_feedback = re.sub(r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)", ", ", execution_feedback)
            factor_feedback.execution_feedback = "\n".join(
                [line for line in execution_feedback.split("\n") if "warning" not in line.lower()]
            )

            # 2. Get factor value feedback
            if gen_df is None:
                factor_feedback.value_feedback = "No factor value generated, skip value evaluation."
                factor_feedback.value_generated_flag = False
                decision_from_value_check = None
            else:
                factor_feedback.value_generated_flag = True
                (
                    factor_feedback.value_feedback,
                    decision_from_value_check,
                ) = self.value_evaluator.evaluate(
                    implementation=implementation, gt_implementation=gt_implementation, version=target_task.version
                )

            factor_feedback.final_decision_based_on_gt = gt_implementation is not None

            if decision_from_value_check is not None and decision_from_value_check is True:
                # To avoid confusion, when same_value_or_high_correlation is True, we do not need code feedback
                factor_feedback.code_feedback = "Final decision is True and there are no code critics."
                factor_feedback.final_decision = decision_from_value_check
                factor_feedback.final_feedback = "Value evaluation passed, skip final decision evaluation."
            elif decision_from_value_check is not None and decision_from_value_check is False:
                factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                    target_task=target_task,
                    implementation=implementation,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.value_feedback,
                    gt_implementation=gt_implementation,
                )
                factor_feedback.final_decision = decision_from_value_check
                factor_feedback.final_feedback = "Value evaluation failed, skip final decision evaluation."
            else:
                factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                    target_task=target_task,
                    implementation=implementation,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.value_feedback,
                    gt_implementation=gt_implementation,
                )
                (
                    factor_feedback.final_decision,
                    factor_feedback.final_feedback,
                ) = self.final_decision_evaluator.evaluate(
                    target_task=target_task,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.value_feedback,
                    code_feedback=factor_feedback.code_feedback,
                )
            return factor_feedback


# TODO:
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """When the prompt is too long. We have to shorten it.
    But we should not truncate the prompt directly, so we should find the key we want to shorten and then shorten it.
    """
    # TODO: this should replace most of code in
    # - FactorFinalDecisionEvaluator.evaluate
    # - FactorCodeEvaluator.evaluate
