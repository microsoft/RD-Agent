# AutoRL-Bench Agent Guidelines

## Summary Maintenance (MANDATORY)

You MUST maintain a file called `summary.md` in the workspace root. Update it **after every training attempt and every submission**, not just at the end.

### Format

```markdown
# Run Summary

## Attempt N (YYYY-MM-DD HH:MM)
- **Status**: ✅ Success / ❌ Failure
- **Score**: X.XX | Improvement: +Y.YY | Best: Z.ZZ
- **Training Type**: SFT / GRPO / PPO / DPO / ...
- **Hyperparameters**: lr=X, epochs=Y, batch_size=Z, ...
- **What Was Done**: Briefly describe this attempt's strategy and concrete actions
- **Why**: Why this method/hyperparameter setting was chosen
- **Issues/Progress**: Problems encountered and improvements over the previous attempt
- **Key Code**: Key code changes (if any)
- **Next Step**: Planned next action based on this result
```

### Rules
1. **Append only** — never overwrite previous attempts
2. Analyze `code/train.py` source to extract training type and hyperparameters
3. If training fails, extract root cause from error output
4. "What Was Done" and "Why" are the most important fields — be thorough
5. Update summary.md IMMEDIATELY after each submission result comes back
6. Include the grading server response (score, improvement, best) verbatim
