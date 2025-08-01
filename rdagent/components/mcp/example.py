"""Example usage of Context7 MCP integration."""

import asyncio

from context7 import query_context7


async def main():
    """Main function for testing context7 functionality."""
    error_msg = """### TRACEBACK: Traceback (most recent call last):
Traceback (most recent call last):
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/862e5d2ff8d4489b91c38c5be5001b44/main.py", line 400, in <module>
main()
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/862e5d2ff8d4489b91c38c5be5001b44/main.py", line 285, in main
sample_weight, groups, adv_auc = adversarial_validation(train_df, test_df, features, debug=DEBUG, random_state=random_state)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/862e5d2ff8d4489b91c38c5be5001b44/main.py", line 162, in adversarial_validation
adv_clf.fit(
TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'"""

    # Normal usage (verbose=False by default)
    result = await query_context7(error_msg)
    print("Result:", result)

    # Debug usage with verbose output
    # result = await query_context7(error_msg, verbose=True)
    # print("Debug Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
