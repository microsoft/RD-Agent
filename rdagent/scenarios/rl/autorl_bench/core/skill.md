You are responsible for maintaining the cumulative run summary file `reports/summary.md` for the experiment.

After each round of experiments, you need to use file_editor to append a new section at the end of `reports/summary.md`.

If `reports/summary.md` does not exist, create the file first and write `# Run summary` in the first line.

The section format added in each round (strictly adhered to, no fields added or deleted):

```
## Iteration N (YYYY-MM-DD HH:MM, takes Xs)
- **Status**: ✅ Success / ❌ Failed (exit_code=X)
- **Score**: X | Improvement: X | Best: X (iter N)
- **Training type**: GRPO / SFT / PPO / copy_model / placeholder / unknown
- **Key configuration**: lr=X, epochs=X, batch=X, ... (extracted from code)
- **What was done**: Specific strategies (training methods, reward function design, data processing, etc.)
- **Why**: Why choose this strategy (reasoning based on the results of the previous round)
- **Problems/Progress**: What problems were discovered, or what progress was made compared to the previous round
- **Key code**: 3-5 lines of code that best reflect the strategy of this round
- **Code snippet (context)**: Copy 15-40 lines of runnable context from train.py, mark the function name/line number range; if it is the same as the previous round, write "Same as Iteration N, no changes"
```

Data source recommendations (in order of priority):
- Score/Improvement/Best: scores.json or returned by the server
- Status/Time-consuming/exit_code: run.log
- Training type/key configuration/key code: code/train.py
- Root cause of failure: agent.log + run.log

rule:
1. **Append**, do not overwrite existing content
2. Must be written in increments of Iteration: read the last iteration number in `reports/summary.md`, which must currently be N+1; if not satisfied, correct it first and then write
3. Analyze the source code under code and extract the training type and hyperparameters; if you are not sure, write unknown and leave it blank.
4. If training fails, a localizable root cause (log fragment or error type) must be given and failure_type marked (such as: code_error_runtime / rollout_logic_wrong / timeout_no_submission / copy_model_fallback / training_diverged / unknown)
5. “What was done” and “why” are the most important fields, which must be reproducible and testable, and “why” must refer to the previous round of evidence (scores.json/run.log/agent.log)
6. **Issue/Progress** must contain process indicators or failure types (eg: valid_submission_rate / first_valid_idx / time_to_first_improvement / time_used_ratio / failure_type)
7. Long code snippets can have up to 1 paragraph per round and up to 40 lines; priority will be given to posting new or changed parts of this round; if it is the same as the previous round, write "Same as Iteration N, no changes"
8. If the root cause is the same as the previous round, avoid repeating the entire paragraph and clearly write "new evidence/new attempt/no new addition"
9. Synchronously append `reports/summary.jsonl` (same directory), one line of JSON for each round, field suggestions:
   - iteration, timestamp, duration_s, status, exit_code
   - score, improvement
   - train_type, failure_type
- Only keep quantifiable fields (used in charts), do not write long text such as why/what/next_step
- metrics are uniformly calculated by benchmark post-processing (not filled in in jsonl)
- Example:
     {"iteration": 3, "timestamp": "2026-03-10 12:34", "duration_s": 812, "status": "success", "exit_code": 0, "score": 65.0, "improvement": 20.0, "train_type": "GRPO", "failure_type": "unknown"}
10. Self-check section integrity after writing: the above fields must be complete, if missing, complete them before ending
