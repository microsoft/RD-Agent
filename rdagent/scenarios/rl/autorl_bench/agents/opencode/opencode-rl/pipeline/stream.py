"""流式输出：实时打印 OpenCode 每一轮的动作（rich 美化版）。"""

import time

from .ui import (
    print_agent_thought,
    print_tool_call,
    print_tool_result,
    print_turn_done,
    print_turn_header,
    print_turn_waiting,
)


def make_stream_printer(label: str):
    """返回一个 on_turn 回调，实时打印 agent 每轮的思考和工具调用。

    现在支持同一个 turn 内多次回调（每个工具调用完成后触发一次），
    只在第一次回调时打印 Turn header 和 Agent 思考。

    输出格式示例：
        [CodeGen iter1] Turn 1 (12s)
          Agent: 让我先看看数据格式...
          > bash: head -5 /data/train.jsonl
          < (rc=0) {"question": "What is 2+3?", ...
          > read: /workspace/description.md
          < ok (2345 bytes)
        [CodeGen iter1] Done (turn 1, 45s)
    """
    start = time.time()
    last_turn = [0]  # 用 list 包装以便在闭包里修改
    print_turn_waiting(label)

    def _print(event):
        elapsed = time.time() - start

        if event.finished:
            print_turn_done(label, event.turn, elapsed)
            return

        # 只在 turn 变化时打印 Turn header
        if event.turn != last_turn[0]:
            last_turn[0] = event.turn
            print_turn_header(label, event.turn, elapsed)

        # agent 思考摘要：取第一个非空行，截取 200 字符
        if event.assistant_text:
            lines = [l.strip() for l in event.assistant_text.strip().splitlines() if l.strip()]
            summary = ""
            for line in lines:
                if line.startswith("<") or line.startswith("```"):
                    continue
                summary = line[:200]
                break
            if summary:
                print_agent_thought(summary)

        # 打印每个 tool call + 对应结果
        results = list(event.results) if event.results else []
        for i, call in enumerate(event.calls or []):
            result = results[i] if i < len(results) else None
            payload = call.payload if isinstance(call.payload, dict) else {}

            if call.kind == "bash":
                cmd = str(payload.get("command", ""))
                if len(cmd) > 150:
                    cmd = cmd[:147] + "..."
                print_tool_call("bash", cmd)
                if result:
                    rc = result.detail.get("rc", "?")
                    stdout = str(result.detail.get("stdout") or "").strip()
                    stderr = str(result.detail.get("stderr") or "").strip()
                    if result.ok:
                        out = stdout[:200].replace("\n", " | ") if stdout else ""
                        print_tool_result(f"(rc={rc}) {out}", ok=True)
                    else:
                        err = stderr[:200].replace("\n", " | ") if stderr else stdout[:200].replace("\n", " | ")
                        print_tool_result(f"(rc={rc}) {err}", ok=False)

            elif call.kind == "file":
                file_path = str(payload.get("filePath", ""))
                if result is None:
                    print_tool_call("file", file_path)
                    continue

                kind = result.kind
                ok_str = "ok" if result.ok else str(result.detail.get("error", "failed"))

                if kind == "read":
                    print_tool_call("read", file_path)
                    if result.ok:
                        content = str(result.detail.get("content") or "")
                        print_tool_result(f"ok ({len(content)} chars)", ok=True)
                    else:
                        print_tool_result(ok_str, ok=False)

                elif kind in ("write", "edit"):
                    size = result.detail.get("bytes", 0)
                    mode = result.detail.get("mode", kind)
                    print_tool_call(kind, file_path)
                    if result.ok:
                        print_tool_result(f"ok ({size} bytes, {mode})", ok=True)
                    else:
                        print_tool_result(ok_str, ok=False)

                else:
                    print_tool_call(kind, file_path)
                    print_tool_result(ok_str, ok=result.ok)

    return _print
