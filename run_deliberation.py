#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a deliberation scenario against a running LIDA instance.

The LIDA services must already be running (started via run.sh).

Usage:
    python run_deliberation.py --port PORT [--scenario SCENARIO] [--topic TOPIC] [--timeout SECONDS]

Arguments:
    --port PORT           API port of the running LIDA instance (required)
    --scenario SCENARIO   Scenario name (default: quick_personas3)
    --topic TOPIC         Deliberation topic (default: from scenario config)
    --timeout SECONDS     Max seconds to wait (default: 600)
    --output FILE         Output file for logs (default: deliberation_TIMESTAMP.log)
    --poll-interval SECS  Seconds between status checks (default: 5)

Example:
    # First start services:
    ./run.sh 6379 2040 start

    # Then run deliberation:
    python run_deliberation.py --port 2040 --scenario quick_personas3 --timeout 300

    # Or with a custom topic:
    python run_deliberation.py --port 2040 --topic "Should AI be regulated?"

Logs are saved to the logs/ subdirectory:
    logs/deliberation_<scenario>_<timestamp>.log           - Status log
    logs/deliberation_<scenario>_<timestamp>.llm_logs.json - LLM API logs
"""

import argparse
import json
import sys
import time
import signal
import requests
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run deliberation against a running LIDA instance"
    )
    parser.add_argument(
        "--port", type=int, required=True,
        help="API port of the running LIDA instance (required)"
    )
    parser.add_argument(
        "--scenario", default="quick_personas3",
        help="Scenario name (default: quick_personas3)"
    )
    parser.add_argument(
        "--topic", default=None,
        help="Deliberation topic (default: from scenario config)"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Max seconds to wait (default: 600)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file for logs (default: deliberation_TIMESTAMP.log)"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=5,
        help="Seconds between status checks (default: 5)"
    )
    return parser.parse_args()


def get_topic_from_scenario(scenario: str) -> str:
    """Read the auto_start_topic from scenario YAML file."""
    import yaml
    scenario_file = Path(f"scenarios/{scenario}.yaml")
    if not scenario_file.exists():
        return None
    try:
        with open(scenario_file) as f:
            config = yaml.safe_load(f)
        return config.get("simulation", {}).get("auto_start_topic")
    except Exception as e:
        print(f"Warning: Could not read scenario file: {e}")
        return None


def check_health(api_port: int) -> bool:
    """Check if the API is healthy."""
    # Try /health first, then fall back to /api/stats or root
    endpoints = ["/health", "/api/stats", "/"]
    for endpoint in endpoints:
        try:
            resp = requests.get(f"http://localhost:{api_port}{endpoint}", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
    return False


def start_deliberation(api_port: int, topic: str) -> bool:
    """Start a deliberation via API call."""
    url = f"http://localhost:{api_port}/api/deliberate"
    print(f">>> Calling POST {url}?topic={topic[:50]}...", flush=True)
    try:
        resp = requests.post(url, params={"topic": topic}, timeout=10)
        print(f">>> Response: {resp.status_code}", flush=True)
        if resp.status_code == 200:
            result = resp.json()
            print(f">>> Started deliberation: {result}", flush=True)
            return True
        else:
            print(f">>> Failed to start deliberation: {resp.status_code} {resp.text}", flush=True)
            return False
    except requests.exceptions.RequestException as e:
        print(f">>> Error starting deliberation: {e}", flush=True)
        return False


def check_status(api_port: int) -> dict:
    """Check deliberation status via API."""
    try:
        resp = requests.get(f"http://localhost:{api_port}/api/deliberation/status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def fetch_llm_logs(api_port: int) -> dict:
    """Fetch LLM logs from the API."""
    try:
        resp = requests.get(f"http://localhost:{api_port}/api/llm/logs", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch LLM logs: {e}")
    return None


def fetch_deliberation_logs(api_port: int) -> str:
    """Fetch deliberation logs from the API if available."""
    try:
        resp = requests.get(f"http://localhost:{api_port}/api/deliberation/logs", timeout=10)
        if resp.status_code == 200:
            return resp.text
    except requests.exceptions.RequestException:
        pass
    return None


def main():
    args = parse_args()

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Output file
    if args.output:
        output_file = logs_dir / Path(args.output).name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = logs_dir / f"deliberation_{args.scenario}_{timestamp}.log"

    # Get topic from args or scenario config
    topic = args.topic or get_topic_from_scenario(args.scenario)
    if not topic:
        print("ERROR: No topic specified. Use --topic or add auto_start_topic to scenario config.")
        sys.exit(1)

    print("=" * 60)
    print("Deliberation Runner (Client Mode)")
    print("=" * 60)
    print(f"API Port: {args.port}")
    print(f"Scenario: {args.scenario}")
    print(f"Topic: {topic}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output: {output_file}")
    print()

    # Check if API is running
    print(f"Checking API health on port {args.port}...")
    if not check_health(args.port):
        print()
        print("!" * 60)
        print(f"ERROR: Cannot connect to API on port {args.port}")
        print("!" * 60)
        print()
        print("Make sure the LIDA services are running:")
        print(f"  ./run.sh <redis-port> start")
        print()
        print("Then set API_PORT if using non-default:")
        print(f"  API_PORT={args.port} ./run.sh <redis-port> start")
        print()
        sys.exit(1)

    print("API is healthy")
    print()

    # Track state
    deliberation_started = False
    deliberation_completed = False
    start_time = time.time()
    status_log = []
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\nInterrupted! Saving logs...")
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the deliberation
    print(f">>> Starting deliberation: {topic}", flush=True)
    if not start_deliberation(args.port, topic):
        print("ERROR: Failed to start deliberation")
        sys.exit(1)

    print()
    print("Monitoring deliberation progress...")
    print("-" * 60)

    # Poll for status
    try:
        while not interrupted:
            time.sleep(args.poll_interval)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > args.timeout:
                print(f"\nTimeout after {args.timeout}s")
                break

            # Check status
            status = check_status(args.port)

            if status:
                status_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "elapsed": elapsed,
                    "status": status,
                }
                status_log.append(status_entry)

                phase = status.get("phase", "unknown")
                active = status.get("active", False)
                messages = status.get("total_messages", 0)

                if active and not deliberation_started:
                    deliberation_started = True
                    print(f">>> Deliberation STARTED", flush=True)

                # Also mark as started if we see messages (in case we missed the active=True)
                if messages > 0 and not deliberation_started:
                    deliberation_started = True
                    print(f">>> Deliberation STARTED (detected via messages)", flush=True)

                print(f"[{elapsed:6.1f}s] Phase: {phase:<20} Messages: {messages}", flush=True)

                # Check for completion - only if deliberation actually started
                # Require either: we saw it start, OR there are messages, OR phase indicates completion
                has_activity = deliberation_started or messages > 0
                phase_complete = phase and "complete" in phase.lower()

                is_complete = has_activity and (
                    (not active and (messages > 0 or phase_complete)) or
                    phase_complete or
                    (status.get("completed") and messages > 0)
                )

                if is_complete and not deliberation_completed:
                    deliberation_completed = True
                    print()
                    print("=" * 60)
                    print(">>> Deliberation COMPLETED", flush=True)
                    print(f">>> Phase: {phase}", flush=True)
                    print(f">>> Consensus: {status.get('consensus')}", flush=True)
                    print(f">>> Messages: {messages}", flush=True)
                    print("=" * 60)
                    # Give it a moment to finish any final logging
                    time.sleep(2)
                    break
            else:
                print(f"[{elapsed:6.1f}s] (no status available)", flush=True)

    except Exception as e:
        print(f"Error during monitoring: {e}")

    # Fetch and save logs
    print()
    print("-" * 60)
    print("Fetching logs...")

    # Fetch LLM logs
    llm_logs = fetch_llm_logs(args.port)
    llm_logs_file = output_file.with_suffix(".llm_logs.json")
    if llm_logs:
        print(f"Saving {len(llm_logs.get('logs', []))} LLM log entries to {llm_logs_file}...")
        with open(llm_logs_file, "w") as f:
            json.dump(llm_logs, f, indent=2)
    else:
        print("No LLM logs retrieved")

    # Fetch deliberation logs if available
    delib_logs = fetch_deliberation_logs(args.port)

    # Save consolidated log file
    print(f"Saving logs to {output_file}...")
    with open(output_file, "w") as f:
        f.write(f"# Deliberation Run: {args.scenario}\n")
        f.write(f"# API Port: {args.port}\n")
        f.write(f"# Topic: {topic}\n")
        f.write(f"# Started: {datetime.now().isoformat()}\n")
        f.write(f"# Completed: {deliberation_completed}\n")
        f.write(f"# Elapsed: {time.time() - start_time:.1f}s\n")
        f.write(f"# LLM Logs: {llm_logs_file}\n")
        f.write("#" + "=" * 59 + "\n\n")

        f.write("## Status Log\n\n")
        for entry in status_log:
            f.write(f"[{entry['elapsed']:6.1f}s] {json.dumps(entry['status'])}\n")

        if delib_logs:
            f.write("\n## Deliberation Logs\n\n")
            f.write(delib_logs)

    # Final summary
    print()
    print("=" * 60)
    print(f"Deliberation {'COMPLETED' if deliberation_completed else 'INCOMPLETE'}")
    print(f"Logs: {output_file}")
    if llm_logs:
        print(f"LLM logs: {llm_logs_file} ({len(llm_logs.get('logs', []))} entries)")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print()
    print("NOTE: Services are still running. To stop them:")
    print(f"  ./run.sh <redis-port> stop")
    print("=" * 60)

    sys.exit(0 if deliberation_completed else 1)


if __name__ == "__main__":
    main()
