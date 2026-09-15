from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERSingleFeedback,
    _append_return_checking,
)


def test_append_return_checking_initializes_none() -> None:
    feedback = CoSTEERSingleFeedback(
        execution="execution",
        return_checking=None,
        code="code",
        final_decision=True,
    )

    _append_return_checking(feedback, "\n[Error] scores.csv is missing")

    assert feedback.return_checking == "\n[Error] scores.csv is missing"


def test_append_return_checking_preserves_existing_text() -> None:
    feedback = CoSTEERSingleFeedback(
        execution="execution",
        return_checking="model feedback",
        code="code",
        final_decision=True,
    )

    _append_return_checking(feedback, "\nsubmission check failed")

    assert feedback.return_checking == "model feedback\nsubmission check failed"
