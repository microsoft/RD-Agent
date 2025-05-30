import glob
import os

PROMPT_TEMPLATE = open("rdagent/scenarios/data_science/dev/prompts_v3.yaml").read()

competition_names = [
    "aerial-cactus-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "hotel-id-2021-fgvc8",
    "mlsp-2013-birds",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "text-normalization-challenge-english-language"
]

for COMPETITION_NAME in competition_names:

    STATISTICS = open(f"logs_0523/{COMPETITION_NAME}.csv").read()

    USER_PROMPT = f"""
You are an expert data scientist specializing in analyzing and improving automated machine learning systems, particularly focusing on the decision-making processes driven by Large Language Models (LLMs).

## System Overview: RD-Agent ##
RD-Agent (Research-Development Agent) is an automated tool designed to assist data scientists in Kaggle competitions. It operates in a loop:
1. Proposing new ideas for solutions.
2. Implementing and testing these ideas.
3. Deciding whether the new solution should replace the current best candidate for final submission.

A key challenge is that Kaggle competitions often have a private leaderboard with limited submission attempts (typically one). RD-Agent uses an LLM to decide in each iteration if a newly developed solution is promising enough to become the new submission candidate. This decision is guided by a specific **Decision Prompt Template** (this is the template we ultimately want to improve).

## The Problem: Suboptimal Solution Choices ##
In a recent Kaggle competition, '{COMPETITION_NAME}', RD-Agent made suboptimal choices. Promising solutions (potential medal winners based on metric 'Running Score (test)') were discarded. Subsequently, less effective solutions were sometimes chosen, leading to a final submission that wasn't the best possible. This pattern suggests potential flaws in how the LLM was guided by the **Decision Prompt Template**.

## Your Mission: Analyze Past Decisions to Refine the Decision Prompt Template ##
Your primary objective is to analyze the historical operational logs of RD-Agent for the '{COMPETITION_NAME}' competition. Based on this analysis, you will identify why the decision-making LLM made suboptimal choices and propose specific, actionable improvements to the **Decision Prompt Template** that RD-Agent uses to query the LLM for solution selection.

## Provided Data for Your Analysis ##

### 1. Performance Summary Statistics ###

The following table summarizes RD-Agent's performance across various iterations.
* "Feedback = ✅" indicates the LLM, guided by the **Decision Prompt Template**, chose to adopt the solution in that iteration as the new submission candidate.
* "Feedback = ❌" signifies the solution was rejected.
* Observe that the last solution marked with "Feedback = ✅" is the one submitted, which may not always possess the best "Running Score (test)".
* Remember the "Running Score (test)" is not visible to the LLM during decision-making, so it cannot directly influence the LLM's choices.

```csv
{STATISTICS}
```

### 2. Detailed Iteration Logs ###

You will be provided with (or have access to) detailed logs for relevant iterations from the '{COMPETITION_NAME}' competition. These logs are crucial and contain:
* The exact prompt content (generated from the **Decision Prompt Template** in use at that time) that was given to the decision-making LLM. This includes the competition scenario, current solution sketch, code, evaluation criteria, etc.
* The decision-making LLM's full response, including whether it chose to replace the solution and its articulated reasoning.
* Note: Some iterations (i.e., loops) might be absent from the logs if RD-Agent couldn't produce a valid, testable solution for a given idea.

## Required Output: Detailed Analysis and Recommendations ##

Please structure your response to include the following sections:

### 1. Identification of Key Misjudgments ###
* Pinpoint specific iterations where **high-potential solutions** (e.g., strong 'Running Score (test)' or other positive indicators evident in the logs/statistics) were **erroneously rejected (Feedback = ❌)** by the LLM.
* Identify specific iterations where **subpar or demonstrably worse solutions** were **erroneously accepted (Feedback = ✅)** by the LLM, particularly if they superseded a superior earlier candidate.
* Keep in mind that when LLM makes decisions, it does not have access to the "Running Score (test)" but only to the "Running Score (valid)", although the former is visible in the logs/statistics you have.

### 2. Root Cause Analysis of LLM Decisions ###
For the misjudgments identified above, find the corresponding iteration (by iteration/loop number) in the provided logs (LLM prompt and response) to determine the most likely reasons for the LLM's decisions. Consider factors such as:
* Ambiguity or misinterpretation of instructions within the **Decision Prompt Template**.
* Incorrect weighting or focus on specific evaluation criteria by the LLM.
* Missing or insufficient contextual information in the prompt provided to the LLM.
* Potential flaws in the reasoning framework implicitly or explicitly suggested by the **Decision Prompt Template**.
* Any discernible patterns in the LLM's reasoning that led to suboptimal outcomes.

### 3. Actionable Recommendations for Improving the Decision Prompt Template ###
* Based on your root cause analysis, propose concrete, specific, and actionable modifications to the **Decision Prompt Template** (i.e., the template RD-Agent uses to instruct the LLM on solution selection).
* For each suggestion, clearly explain how it addresses the identified issues and how it is expected to guide the LLM towards more accurate and beneficial decisions in future RD-Agent runs.
* Focus on enhancing clarity, ensuring comprehensive context, refining evaluation criteria, improving instruction precision, and optimizing the overall structure of the **Decision Prompt Template**. Your goal is to make this template more robust against future misjudgments.

Please begin your detailed analysis.
""".strip()

    LOGS_COMBINED = ""
    for file in sorted(glob.glob(f"logs_0523/feedback_{COMPETITION_NAME}_loop*.log")):
        with open(file, "r") as f:
            header = "### " + os.path.basename(file).upper() + " ###"
            LOGS_COMBINED += "#" * len(header) + "\n" + header + "\n" + "#" * len(header) + "\n\n" + f.read() + "\n\n\n"


    with open(f"logs_0523/prompt_user_{COMPETITION_NAME}.txt", "w") as f:
        f.write(USER_PROMPT)
    with open(f"logs_0523/prompt_logs_{COMPETITION_NAME}.log", "w") as f:
        f.write(LOGS_COMBINED)
