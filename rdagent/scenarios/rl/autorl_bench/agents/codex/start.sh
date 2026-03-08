#!/bin/bash
# Codex CLI Agent wrapper for AutoRL-Bench

echo "=== Codex CLI Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"
echo "Grading Server: $GRADING_SERVER_URL"

# TRAPI token (az login must be done beforehand)
export TRAPI_API_KEY=$(az account get-access-token --resource "api://trapi" --query accessToken --output tsv 2>/dev/null)
if [ -z "$TRAPI_API_KEY" ]; then
    echo "ERROR: Failed to get TRAPI token. Run 'az login' first."
    exit 1
fi
echo "TRAPI token: ${#TRAPI_API_KEY} chars"

CODEX_MODEL="${CODEX_MODEL:-gpt-5.1-codex-max_2025-12-04}"
CODEX_PROVIDER="${CODEX_PROVIDER:-trapi}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-36000}"
START_EPOCH=$(date +%s)

# Generate timer.sh in workspace so agent can query remaining time
cat > "$WORKSPACE/timer.sh" << TIMER
#!/bin/bash
DEADLINE=$((START_EPOCH + CODEX_TIMEOUT))
NOW=\$(date +%s)
REMAINING=\$((DEADLINE - NOW))
if [ \$REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    HOURS=\$((REMAINING / 3600))
    MINUTES=\$(((REMAINING % 3600) / 60))
    printf "Remaining: %d:%02d\n" \$HOURS \$MINUTES
fi
TIMER
chmod +x "$WORKSPACE/timer.sh"

# Build prompt from workspace files
INSTRUCTIONS=$(cat "$WORKSPACE/instructions.md" 2>/dev/null || echo "")
DESCRIPTION=$(cat "$WORKSPACE/description.md" 2>/dev/null || echo "")
WORKSPACE_LS=$(ls -la "$WORKSPACE" 2>/dev/null)
DATA_SAMPLE=$(head -5 "$WORKSPACE/data/"*.jsonl 2>/dev/null || head -5 "$WORKSPACE/data/"*.json 2>/dev/null || echo "No data files found")

PROMPT="You are an AI researcher doing RL post-training. Complete the entire task autonomously.

## Task: ${TASK}
## Base Model: ${BASE_MODEL}
## Model Path: ${MODEL_PATH}
## Output Dir: ${OUTPUT_DIR}
## Grading Server: ${GRADING_SERVER_URL}

## Task Description
${DESCRIPTION}

## Instructions
${INSTRUCTIONS}

## Workspace Contents
\`\`\`
${WORKSPACE_LS}
\`\`\`

## Data Sample (first 5 lines)
\`\`\`
${DATA_SAMPLE}
\`\`\`

## Your Mission
1. Read all files in the workspace to understand the task
2. Write training code in code/train.py (use GRPO, PPO, SFT, or any effective method)
3. Run the training: python code/train.py
4. Save the trained model to ${OUTPUT_DIR}/v1
5. IMPORTANT: If you use LoRA/PEFT, you MUST merge before saving:
   model = model.merge_and_unload()
   model.save_pretrained(output_path)
   tokenizer.save_pretrained(output_path)
6. Fix tokenizer_config.json if needed (remove extra_special_tokens list format)
7. Submit for evaluation:
   curl -X POST ${GRADING_SERVER_URL}/submit -H 'Content-Type: application/json' -d '{\"model_path\": \"${OUTPUT_DIR}/v1\"}'
8. Based on the score, iterate: improve your approach and submit again as v2, v3, etc.
9. Keep iterating until you achieve the best possible score or run out of time.

## Time Budget
You have ${CODEX_TIMEOUT} seconds total. Run \`bash timer.sh\` at any time to check remaining time.
- If > 2 hours remain: try ambitious methods (GRPO, PPO)
- If < 30 min remain: submit your best model immediately

IMPORTANT: Work efficiently. Start with a simple approach, get a baseline score, then iterate."

echo "Prompt length: ${#PROMPT} chars"
echo "Running Codex CLI..."

# JSON trace goes to agent.jsonl; original text log is still agent.log (via run.py stdout redirect)
JSONL_LOG="$WORKSPACE/agent.jsonl"

timeout "${CODEX_TIMEOUT}" codex exec \
    --search \
    --json \
    -m "${CODEX_MODEL}" \
    -c "model_provider=\"${CODEX_PROVIDER}\"" \
    -c "model_reasoning_summary=\"detailed\"" \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    -C "$WORKSPACE" \
    "$PROMPT" \
    > "$JSONL_LOG" 2>&1

EXIT_CODE=$?

# --- Diagnostics ---
echo ""
echo "--- DIAGNOSTICS ---"
echo "exit_code: $EXIT_CODE"
END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))
printf "elapsed: %02d:%02d:%02d\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "model_files: $(ls "$OUTPUT_DIR/" 2>/dev/null | wc -l) dirs in output/"
echo "code_files: $(ls "$WORKSPACE/code/" 2>/dev/null | wc -l) files in code/"
echo "summary_exists: $([ -f "$WORKSPACE/summary.md" ] && echo yes || echo no)"
echo "gpu_memory:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo "disk_workspace: $(du -sh "$WORKSPACE" 2>/dev/null | cut -f1)"
echo "--- END DIAGNOSTICS ---"

echo "Codex CLI exited with code: $EXIT_CODE"
exit $EXIT_CODE
