from pathlib import Path

# Check if our submission file exists
assert Path("submission.csv").exists(), "Error: submission.csv not found"

submission_lines = Path("submission.csv").read_text().splitlines()
test_lines = Path("submission_test.csv").read_text().splitlines()

is_valid = len(submission_lines) == len(test_lines)

if is_valid:
    message = "submission.csv and submission_test.csv have the same number of lines."
else:
    message = (
        f"submission.csv has {len(submission_lines)} lines, while submission_test.csv has {len(test_lines)} lines."
    )

print(message)

if not is_valid:
    raise AssertionError("Submission is invalid")
