#!/usr/bin/env python3
"""Pretty-print Codex CLI stream JSON files.

Auto-detects format: if one of the first 200 lines starts with
'{"type":"thread.started"', the structured JSON parser is used;
otherwise the file is copied verbatim.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
from pathlib import Path
from typing import Any

DETECT_LINES = 200
DETECT_PREFIX = '{"type":"thread.started"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Codex CLI --json output file into a human-readable text report. "
            "Auto-detects structured JSON vs plain text."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input JSONL file produced by codex CLI",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Destination text file. Defaults to <input>.parsed.txt in the same "
            "directory."
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the parsed output to stdout instead of writing a file.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ""
    if suffix:
        return input_path.with_suffix(f"{suffix}.parsed.txt")
    return input_path.with_name(f"{input_path.name}.parsed.txt")


def is_structured_json(input_path: Path) -> bool:
    """Check if the file is structured Codex JSON by scanning the first DETECT_LINES lines."""
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= DETECT_LINES:
                break
            if line.lstrip().startswith(DETECT_PREFIX):
                return True
    return False


def copy_file(input_path: Path, args: argparse.Namespace) -> None:
    """Copy the input file verbatim to the output."""
    if args.stdout:
        print(input_path.read_text(encoding="utf-8"))
        return

    output_path = args.output or default_output_path(input_path)
    with input_path.open("rb") as src, output_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"Wrote copied file to {output_path}")


def pretty_format_json(obj: Any, indent_level: int = 0) -> str:
    """Format JSON with actual newlines preserved in strings."""
    indent_str = "  " * indent_level
    next_indent = "  " * (indent_level + 1)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for key, value in obj.items():
            formatted_value = pretty_format_json(value, indent_level + 1)
            if (
                "\n" in formatted_value
                and not formatted_value.startswith("{")
                and not formatted_value.startswith("[")
            ):
                first_line = formatted_value.split("\n")[0]
                rest_lines = "\n".join(formatted_value.split("\n")[1:])
                items.append(f'{next_indent}"{key}": {first_line}\n{rest_lines}')
            else:
                items.append(f'{next_indent}"{key}": {formatted_value}')
        return "{\n" + ",\n".join(items) + "\n" + indent_str + "}"
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        items = []
        for item in obj:
            formatted_item = pretty_format_json(item, indent_level + 1)
            items.append(f"{next_indent}{formatted_item}")
        return "[\n" + ",\n".join(items) + "\n" + indent_str + "]"
    elif isinstance(obj, str):
        if "\n" in obj:
            return obj
        else:
            return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif obj is None:
        return "null"
    else:
        return str(obj)


def indent(text: str, level: int) -> str:
    pad = "  " * level
    return "\n".join(pad + line if line else pad for line in text.splitlines())


def format_unparsable_line(index: int, line: str, error_msg: str = "") -> str:
    return line


def format_command(command: list[str] | str) -> str:
    """Format a command for display."""
    if isinstance(command, list):
        return " ".join(shlex.quote(str(token)) for token in command)
    return str(command)


def format_event(index: int, data: dict[str, Any]) -> str:
    """Format a Codex event for display."""
    # Codex events are wrapped: {"id": "...", "msg": {...}}
    # The actual event type is in msg.type
    msg = data.get("msg", data)
    event_id = data.get("id", "")
    event_type = msg.get("type", "unknown")

    header_bits: list[str] = [f"type: {event_type}"]
    if event_id:
        header_bits.append(f"id: {event_id}")

    header_extra = " | ".join(header_bits)
    lines: list[str] = [f"=== Event {index} | {header_extra} ==="]

    handler = EVENT_HANDLERS.get(event_type, format_unknown_event)
    lines.extend(handler(msg))

    return "\n".join(lines)


def format_session_configured(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if session_id := msg.get("session_id"):
        lines.append(indent(f"Session: {session_id}", 1))
    if model := msg.get("model"):
        lines.append(indent(f"Model: {model}", 1))
    if provider := msg.get("model_provider_id"):
        lines.append(indent(f"Provider: {provider}", 1))
    if cwd := msg.get("cwd"):
        lines.append(indent(f"Working directory: {cwd}", 1))
    if approval := msg.get("approval_policy"):
        lines.append(indent(f"Approval policy: {approval}", 1))
    if sandbox := msg.get("sandbox_policy"):
        lines.append(indent(f"Sandbox policy: {sandbox}", 1))
    return lines


def format_task_started(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if ctx_window := msg.get("model_context_window"):
        lines.append(indent(f"Context window: {ctx_window}", 1))
    if collab_mode := msg.get("collaboration_mode_kind"):
        lines.append(indent(f"Collaboration mode: {collab_mode}", 1))
    return lines


def format_task_complete(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if last_msg := msg.get("last_agent_message"):
        lines.append(indent("Last message:", 1))
        lines.append(indent(last_msg.rstrip(), 2))
    return lines


def format_agent_message(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if message := msg.get("message"):
        lines.append(indent("Message:", 1))
        lines.append(indent(message.rstrip(), 2))
    return lines


def format_agent_message_delta(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if delta := msg.get("delta"):
        lines.append(indent(f"Delta: {delta}", 1))
    return lines


def format_user_message(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if message := msg.get("message"):
        lines.append(indent("Message:", 1))
        lines.append(indent(message.rstrip(), 2))
    if images := msg.get("images"):
        lines.append(indent(f"Images: {images}", 1))
    return lines


def format_exec_command_begin(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if call_id := msg.get("call_id"):
        lines.append(indent(f"Call ID: {call_id}", 1))
    if command := msg.get("command"):
        lines.append(indent(f"Command: {format_command(command)}", 1))
    if cwd := msg.get("cwd"):
        lines.append(indent(f"Working directory: {cwd}", 1))
    if source := msg.get("source"):
        lines.append(indent(f"Source: {source}", 1))
    return lines


def format_exec_command_output_delta(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if call_id := msg.get("call_id"):
        lines.append(indent(f"Call ID: {call_id}", 1))
    if chunk := msg.get("chunk"):
        lines.append(indent("Output:", 1))
        lines.append(indent(chunk.rstrip(), 2))
    return lines


def format_exec_command_end(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if call_id := msg.get("call_id"):
        lines.append(indent(f"Call ID: {call_id}", 1))
    if command := msg.get("command"):
        lines.append(indent(f"Command: {format_command(command)}", 1))
    if (exit_code := msg.get("exit_code")) is not None:
        lines.append(indent(f"Exit code: {exit_code}", 1))
    if stdout := msg.get("stdout"):
        lines.append(indent("Stdout:", 1))
        lines.append(indent(stdout.rstrip(), 2))
    if stderr := msg.get("stderr"):
        lines.append(indent("Stderr:", 1))
        lines.append(indent(stderr.rstrip(), 2))
    return lines


def format_agent_reasoning(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if text := msg.get("text"):
        lines.append(indent("Reasoning:", 1))
        lines.append(indent(text.rstrip(), 2))
    if title := msg.get("title"):
        lines.append(indent(f"Title: {title}", 1))
    return lines


def format_agent_reasoning_delta(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if delta := msg.get("delta"):
        lines.append(indent(f"Delta: {delta}", 1))
    return lines


def format_token_count(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if session := msg.get("session"):
        bits = []
        for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_output_tokens"):
            if key in session:
                bits.append(f"{key}={session[key]}")
        if bits:
            lines.append(indent(f"Session: {', '.join(bits)}", 1))
    if turn := msg.get("turn"):
        bits = []
        for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_output_tokens"):
            if key in turn:
                bits.append(f"{key}={turn[key]}")
        if bits:
            lines.append(indent(f"Turn: {', '.join(bits)}", 1))
    return lines


def format_error(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if message := msg.get("message"):
        lines.append(indent(f"Error: {message}", 1))
    if code := msg.get("code"):
        lines.append(indent(f"Code: {code}", 1))
    return lines


def format_warning(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if message := msg.get("message"):
        lines.append(indent(f"Warning: {message}", 1))
    return lines


def format_mcp_tool_call_begin(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if server := msg.get("server_name"):
        lines.append(indent(f"Server: {server}", 1))
    if tool := msg.get("tool_name"):
        lines.append(indent(f"Tool: {tool}", 1))
    if args := msg.get("arguments"):
        lines.append(indent("Arguments:", 1))
        lines.append(indent(pretty_format_json(args, 0), 2))
    return lines


def format_mcp_tool_call_end(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if server := msg.get("server_name"):
        lines.append(indent(f"Server: {server}", 1))
    if tool := msg.get("tool_name"):
        lines.append(indent(f"Tool: {tool}", 1))
    if result := msg.get("result"):
        lines.append(indent("Result:", 1))
        lines.append(indent(pretty_format_json(result, 0), 2))
    return lines


def format_patch_apply_begin(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if call_id := msg.get("call_id"):
        lines.append(indent(f"Call ID: {call_id}", 1))
    if patch := msg.get("patch"):
        lines.append(indent("Patch:", 1))
        lines.append(indent(patch.rstrip(), 2))
    return lines


def format_patch_apply_end(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if call_id := msg.get("call_id"):
        lines.append(indent(f"Call ID: {call_id}", 1))
    if success := msg.get("success"):
        lines.append(indent(f"Success: {success}", 1))
    if error := msg.get("error"):
        lines.append(indent(f"Error: {error}", 1))
    return lines


def format_turn_aborted(msg: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if reason := msg.get("reason"):
        lines.append(indent(f"Reason: {reason}", 1))
    return lines


def format_unknown_event(msg: dict[str, Any]) -> list[str]:
    # Filter out the type field for cleaner output
    filtered = {k: v for k, v in msg.items() if k != "type"}
    if filtered:
        return [indent(pretty_format_json(filtered, 0), 1)]
    return []


EVENT_HANDLERS: dict[str, Any] = {
    "session_configured": format_session_configured,
    "task_started": format_task_started,
    "turn_started": format_task_started,
    "task_complete": format_task_complete,
    "turn_complete": format_task_complete,
    "agent_message": format_agent_message,
    "agent_message_delta": format_agent_message_delta,
    "user_message": format_user_message,
    "exec_command_begin": format_exec_command_begin,
    "exec_command_output_delta": format_exec_command_output_delta,
    "exec_command_end": format_exec_command_end,
    "agent_reasoning": format_agent_reasoning,
    "agent_reasoning_delta": format_agent_reasoning_delta,
    "agent_reasoning_raw_content": format_agent_reasoning,
    "agent_reasoning_raw_content_delta": format_agent_reasoning_delta,
    "token_count": format_token_count,
    "error": format_error,
    "warning": format_warning,
    "mcp_tool_call_begin": format_mcp_tool_call_begin,
    "mcp_tool_call_end": format_mcp_tool_call_end,
    "patch_apply_begin": format_patch_apply_begin,
    "patch_apply_end": format_patch_apply_end,
    "turn_aborted": format_turn_aborted,
}


def is_delta_event(event: dict[str, Any]) -> tuple[bool, str | None]:
    """Check if this event is a streaming delta. Returns (is_delta, delta_type)."""
    msg = event.get("msg", event)
    event_type = msg.get("type", "")
    if event_type in ("agent_message_delta", "agent_reasoning_delta", "agent_reasoning_raw_content_delta"):
        return True, event_type
    return False, None


def format_consolidated_deltas(index: int, deltas: list[dict[str, Any]], delta_type: str) -> str:
    """Format a sequence of delta events as a single consolidated event."""
    if not deltas:
        return ""

    # Combine all delta content
    combined_content = ""
    for d in deltas:
        msg = d.get("msg", d)
        if chunk := msg.get("delta"):
            combined_content += chunk
        elif chunk := msg.get("text"):
            combined_content += chunk

    # Build header
    type_label = delta_type.replace("_delta", "").replace("_", " ")
    header = f"=== Event {index} | type: {delta_type} (consolidated from {len(deltas)} deltas) ==="
    lines = [header]

    if combined_content:
        lines.append(indent(f"{type_label.title()}:", 1))
        lines.append(indent(combined_content.rstrip(), 2))

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    if not is_structured_json(input_path):
        copy_file(input_path, args)
        return

    output_path = args.output or default_output_path(input_path)

    formatted_events: list[str] = []
    pending_deltas: list[dict[str, Any]] = []
    current_delta_type: str | None = None
    event_counter = 0

    def flush_deltas() -> None:
        nonlocal pending_deltas, current_delta_type, event_counter
        if pending_deltas and current_delta_type:
            event_counter += 1
            formatted_events.append(
                format_consolidated_deltas(event_counter, pending_deltas, current_delta_type)
            )
            pending_deltas = []
            current_delta_type = None

    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                flush_deltas()
                formatted_events.append(
                    format_unparsable_line(0, stripped, exc.msg)
                )
                continue

            if not isinstance(event, dict):
                flush_deltas()
                formatted_events.append(
                    format_unparsable_line(0, stripped, "Parsed JSON is not an object")
                )
                continue

            is_delta, delta_type = is_delta_event(event)
            if is_delta:
                # If delta type changes, flush previous deltas first
                if current_delta_type is not None and delta_type != current_delta_type:
                    flush_deltas()
                pending_deltas.append(event)
                current_delta_type = delta_type
            else:
                flush_deltas()
                event_counter += 1
                formatted_events.append(format_event(event_counter, event))

    flush_deltas()

    output_text = "\n\n".join(formatted_events) + "\n"

    if args.stdout:
        print(output_text)
    else:
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote parsed report to {output_path}")


if __name__ == "__main__":
    main()
