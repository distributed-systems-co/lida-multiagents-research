#!/usr/bin/env python3
"""
Interactive LLM log viewer.

Usage:
    python view_logs.py <logfile.json>
    python view_logs.py logs/deliberation_*.llm_logs.json

Controls:
    Up/Down, j/k  - Navigate entries
    Enter/Space   - View full entry details
    q/Esc         - Back / Quit
    p             - Toggle show prompts only
    r             - Toggle show responses only
    /             - Search
    n             - Next search result
    Home/g        - Go to top
    End/G         - Go to bottom
"""

import argparse
import curses
import json
import sys
import textwrap
from pathlib import Path
from typing import Optional


def load_logs(filepath: str) -> list:
    """Load LLM logs from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    # Handle both formats: {"logs": [...]} or just [...]
    if isinstance(data, dict) and "logs" in data:
        return data["logs"]
    elif isinstance(data, list):
        return data
    else:
        return []


def truncate(text: str, width: int) -> str:
    """Truncate text to fit width."""
    if not text:
        return ""
    # Replace newlines with spaces for single-line display
    text = text.replace("\n", " ").replace("\r", "")
    if len(text) > width:
        return text[:width - 3] + "..."
    return text


def format_entry_line(entry: dict, width: int, index: int) -> str:
    """Format a log entry for the list view."""
    model = entry.get("model_actual", entry.get("model", "unknown"))[:20]
    agent = entry.get("agent_name", entry.get("agent_id", "?"))[:15]

    # Extract time (HH:MM:SS) from timestamp
    timestamp = entry.get("timestamp", "")
    time_str = ""
    if timestamp:
        try:
            # Handle ISO format: 2026-01-23T01:04:04.454623+00:00
            if "T" in timestamp:
                time_part = timestamp.split("T")[1].split(".")[0]  # Get HH:MM:SS
                time_str = time_part[:8]
        except:
            pass

    # Get prompt preview - check both formats
    prompt = entry.get("prompt", "")
    if not prompt:
        messages = entry.get("messages", [])
        if messages:
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user:
                content = last_user.get("content", "")
                if isinstance(content, list):
                    content = " ".join(c.get("text", str(c)) for c in content if isinstance(c, dict))
                prompt = content

    # Skip first two lines (Topic: ... and blank line) for list view
    if prompt:
        lines = prompt.split("\n")
        if len(lines) > 2:
            prompt = "\n".join(lines[2:])

    # Get response preview
    response = entry.get("response", "")
    if isinstance(response, dict):
        response = response.get("content", str(response))

    # Format: [idx] time | agent | prompt... -> response...
    prefix = f"[{index:3d}] {time_str:8} {agent:<15} "
    remaining = width - len(prefix) - 4

    if remaining > 20:
        half = remaining // 2
        prompt_part = truncate(prompt, half)
        response_part = truncate(response, half)
        return f"{prefix}{prompt_part} -> {response_part}"
    else:
        return f"{prefix}{truncate(prompt, remaining)}"


def wrap_text(text: str, width: int) -> list:
    """Wrap text to multiple lines."""
    if not text:
        return [""]
    lines = []
    for paragraph in text.split("\n"):
        if paragraph:
            wrapped = textwrap.wrap(paragraph, width=width, break_long_words=True, break_on_hyphens=False)
            lines.extend(wrapped if wrapped else [""])
        else:
            lines.append("")
    return lines


def show_detail_view(stdscr, entry: dict, index: int):
    """Show detailed view of a single entry."""
    curses.curs_set(0)

    # Build content lines
    lines = []

    lines.append(f"{'=' * 60}")
    lines.append(f"Entry #{index}")
    lines.append(f"{'=' * 60}")
    lines.append("")

    # Metadata
    lines.append(f"Model: {entry.get('model_actual', entry.get('model', 'unknown'))}")
    lines.append(f"Agent: {entry.get('agent_name', entry.get('agent_id', 'unknown'))}")
    if "timestamp" in entry:
        lines.append(f"Time: {entry['timestamp']}")
    if "duration_ms" in entry:
        lines.append(f"Duration: {entry['duration_ms']}ms")
    if "tokens_in" in entry or "tokens_out" in entry:
        lines.append(f"Tokens: {entry.get('tokens_in', '?')} in / {entry.get('tokens_out', '?')} out")
    elif "tokens" in entry:
        tokens = entry["tokens"]
        lines.append(f"Tokens: {tokens.get('input', '?')} in / {tokens.get('output', '?')} out")
    lines.append("")

    height, width = stdscr.getmaxyx()
    content_width = width - 2

    # Prompt - check direct prompt field first
    prompt = entry.get("prompt", "")
    if prompt:
        lines.append(f"{'-' * 40}")
        lines.append("PROMPT:")
        lines.append(f"{'-' * 40}")
        lines.extend(wrap_text(prompt, content_width))
    else:
        # Fall back to messages format
        messages = entry.get("messages", [])
        if messages:
            lines.append(f"{'-' * 40}")
            lines.append("MESSAGES / PROMPTS:")
            lines.append(f"{'-' * 40}")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle content arrays (e.g., with images)
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "text":
                                parts.append(c.get("text", ""))
                            else:
                                parts.append(f"[{c.get('type', 'unknown')}]")
                        else:
                            parts.append(str(c))
                    content = "\n".join(parts)

                lines.append("")
                lines.append(f"[{role}]")
                lines.extend(wrap_text(content, content_width))

    # Response
    response = entry.get("response", "")
    if response:
        lines.append("")
        lines.append(f"{'-' * 40}")
        lines.append("RESPONSE:")
        lines.append(f"{'-' * 40}")
        if isinstance(response, dict):
            response = response.get("content", json.dumps(response, indent=2))
        lines.extend(wrap_text(response, content_width))

    # Error if any
    if "error" in entry:
        lines.append("")
        lines.append(f"{'-' * 40}")
        lines.append("ERROR:")
        lines.append(f"{'-' * 40}")
        lines.extend(wrap_text(str(entry["error"]), content_width))

    lines.append("")
    lines.append(f"{'=' * 60}")
    lines.append("Press q/Esc/Enter to go back, j/k or Up/Down to scroll")

    # Scroll state
    scroll_offset = 0
    max_scroll = max(0, len(lines) - height + 2)

    while True:
        stdscr.clear()

        # Draw lines
        for i, line in enumerate(lines[scroll_offset:scroll_offset + height - 1]):
            try:
                stdscr.addnstr(i, 0, line, width - 1)
            except curses.error:
                pass

        # Scroll indicator
        if max_scroll > 0:
            pct = int(100 * scroll_offset / max_scroll) if max_scroll else 0
            indicator = f" [{scroll_offset}/{max_scroll}] {pct}% "
            try:
                stdscr.addstr(height - 1, width - len(indicator) - 1, indicator, curses.A_REVERSE)
            except curses.error:
                pass

        stdscr.refresh()

        key = stdscr.getch()

        if key in (ord('q'), ord('\x1b'), ord('\n'), ord(' ')):  # q, Esc, Enter, Space
            break
        elif key in (curses.KEY_UP, ord('k')):
            scroll_offset = max(0, scroll_offset - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            scroll_offset = min(max_scroll, scroll_offset + 1)
        elif key == curses.KEY_PPAGE:  # Page Up
            scroll_offset = max(0, scroll_offset - (height - 2))
        elif key == curses.KEY_NPAGE:  # Page Down
            scroll_offset = min(max_scroll, scroll_offset + (height - 2))
        elif key in (curses.KEY_HOME, ord('g')):
            scroll_offset = 0
        elif key in (curses.KEY_END, ord('G')):
            scroll_offset = max_scroll


def main_loop(stdscr, logs: list, filename: str):
    """Main interactive loop."""
    curses.curs_set(0)
    curses.use_default_colors()

    # Try to set up colors
    try:
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
    except:
        pass

    selected = 0
    scroll_offset = 0
    search_term = ""
    search_results = []
    search_idx = 0

    while True:
        height, width = stdscr.getmaxyx()
        stdscr.clear()

        # Header
        header = f" LLM Log Viewer: {Path(filename).name} ({len(logs)} entries) "
        try:
            stdscr.addstr(0, 0, header[:width-1], curses.A_REVERSE)
        except curses.error:
            pass

        # Help line
        help_text = " q:quit  Enter:view  /:search  n:next "
        try:
            stdscr.addstr(0, width - len(help_text) - 1, help_text, curses.A_REVERSE)
        except curses.error:
            pass

        # Calculate visible range
        list_height = height - 3  # header + footer + status
        if selected < scroll_offset:
            scroll_offset = selected
        elif selected >= scroll_offset + list_height:
            scroll_offset = selected - list_height + 1

        # Draw entries
        for i, entry in enumerate(logs[scroll_offset:scroll_offset + list_height]):
            idx = scroll_offset + i
            line = format_entry_line(entry, width - 2, idx)

            attr = curses.A_NORMAL
            if idx == selected:
                attr = curses.A_REVERSE
            if search_term and idx in search_results:
                attr |= curses.A_BOLD

            try:
                stdscr.addnstr(i + 1, 0, line, width - 1, attr)
            except curses.error:
                pass

        # Footer / status
        status = f" Entry {selected + 1}/{len(logs)} "
        if search_term:
            status += f"| Search: '{search_term}' ({len(search_results)} matches) "
        try:
            stdscr.addstr(height - 1, 0, status, curses.A_DIM)
        except curses.error:
            pass

        stdscr.refresh()

        key = stdscr.getch()

        if key in (ord('q'), ord('\x1b')):  # q or Esc
            break
        elif key in (curses.KEY_UP, ord('k')):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = min(len(logs) - 1, selected + 1)
        elif key == curses.KEY_PPAGE:  # Page Up
            selected = max(0, selected - list_height)
        elif key == curses.KEY_NPAGE:  # Page Down
            selected = min(len(logs) - 1, selected + list_height)
        elif key in (curses.KEY_HOME, ord('g')):
            selected = 0
        elif key in (curses.KEY_END, ord('G')):
            selected = len(logs) - 1
        elif key in (ord('\n'), ord(' ')):  # Enter or Space
            if logs:
                show_detail_view(stdscr, logs[selected], selected)
        elif key == ord('/'):
            # Search mode
            curses.echo()
            curses.curs_set(1)
            try:
                stdscr.addstr(height - 1, 0, " Search: " + " " * (width - 10))
                stdscr.move(height - 1, 9)
                search_term = stdscr.getstr(height - 1, 9, 50).decode('utf-8', errors='ignore')
            except:
                search_term = ""
            curses.noecho()
            curses.curs_set(0)

            # Find matches
            if search_term:
                search_results = []
                term_lower = search_term.lower()
                for i, entry in enumerate(logs):
                    entry_str = json.dumps(entry).lower()
                    if term_lower in entry_str:
                        search_results.append(i)
                search_idx = 0
                if search_results:
                    selected = search_results[0]
        elif key == ord('n'):  # Next search result
            if search_results:
                search_idx = (search_idx + 1) % len(search_results)
                selected = search_results[search_idx]
        elif key == ord('N'):  # Previous search result
            if search_results:
                search_idx = (search_idx - 1) % len(search_results)
                selected = search_results[search_idx]


def main():
    parser = argparse.ArgumentParser(description="Interactive LLM log viewer")
    parser.add_argument("logfile", help="Path to the .llm_logs.json file")
    args = parser.parse_args()

    if not Path(args.logfile).exists():
        print(f"Error: File not found: {args.logfile}")
        sys.exit(1)

    try:
        logs = load_logs(args.logfile)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)

    if not logs:
        print("No log entries found in file.")
        sys.exit(0)

    print(f"Loaded {len(logs)} log entries. Starting viewer...")

    try:
        curses.wrapper(lambda stdscr: main_loop(stdscr, logs, args.logfile))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
