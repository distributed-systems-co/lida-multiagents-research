#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a deliberation scenario and capture results.

Usage:
    python run_deliberation.py [--scenario SCENARIO] [--timeout SECONDS] [--output FILE]

Example:
    python run_deliberation.py --scenario quick_personas3 --timeout 300 --output results.log

Logs are saved to the logs/ subdirectory:
    logs/deliberation_<scenario>_<timestamp>.log       - Console output
    logs/deliberation_<scenario>_<timestamp>.llm_logs.json - LLM API logs
"""

import argparse
import json
import os
import subprocess
import sys
import time
import signal
import requests
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run deliberation scenario")
    parser.add_argument("--scenario", default="quick_personas3", help="Scenario name")
    parser.add_argument("--topic", default=None, help="Deliberation topic (default: from scenario config)")
    parser.add_argument("--timeout", type=int, default=600, help="Max seconds to wait (default: 600)")
    parser.add_argument("--output", default=None, help="Output file for logs (default: deliberation_TIMESTAMP.log)")
    parser.add_argument("--api-port", type=int, default=2040, help="API port (default: 2040)")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds between status checks (default: 5)")
    parser.add_argument("--no-build", action="store_true", help="Skip --build flag")
    parser.add_argument("--no-live", action="store_true", help="Disable live mode (no LLM calls)")
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


def fetch_llm_logs(api_port: int) -> dict:
    """Fetch LLM logs from the API before shutdown."""
    try:
        resp = requests.get(f"http://localhost:{api_port}/api/llm/logs", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch LLM logs: {e}")
    return None


def check_api_key():
    """Check if OPENROUTER_API_KEY is set for live mode."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print()
        print("!" * 60)
        print("!!! ERROR: OPENROUTER_API_KEY is not set !!!")
        print("!" * 60)
        print()
        print("Live mode requires an OpenRouter API key.")
        print()
        print("To fix this, either:")
        print("  1. Export the key:  export OPENROUTER_API_KEY='your-key-here'")
        print("  2. Add to .env:     echo 'OPENROUTER_API_KEY=your-key' >> .env")
        print("  3. Run without LLM: python3 run_deliberation.py --no-live")
        print()
        print("Get an API key at: https://openrouter.ai/keys")
        print()
        print("!" * 60)
        sys.exit(1)
    else:
        # Show that key is set (masked)
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"OPENROUTER_API_KEY: {masked}")


def check_status(api_port: int) -> dict:
    """Check deliberation status via API."""
    try:
        resp = requests.get(f"http://localhost:{api_port}/api/deliberation/status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def wait_for_api(api_port: int, timeout: int = 120) -> bool:
    """Wait for API to become available."""
    print(f"Waiting for API on port {api_port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{api_port}/health", timeout=2)
            if resp.status_code == 200:
                print(f"API ready after {time.time() - start:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


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

    live_mode = not args.no_live

    # Get topic from args or scenario config
    topic = args.topic or get_topic_from_scenario(args.scenario)
    if not topic:
        print("ERROR: No topic specified. Use --topic or add auto_start_topic to scenario config.")
        sys.exit(1)

    print(f"=" * 60)
    print(f"Deliberation Runner")
    print(f"=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Topic: {topic}")
    print(f"Live Mode: {live_mode}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output: {output_file}")
    print(f"API Port: {args.api_port}")
    print()

    # Check API key if live mode
    if live_mode:
        check_api_key()
    else:
        print("Live mode disabled - using simulation")
    print()

    # Build docker-compose command
    env = {
        "SCENARIO": args.scenario,
        "SWARM_LIVE": "true" if live_mode else "false",
        "FULL_LOGS": "true",
    }
    env_str = " ".join(f"{k}={v}" for k, v in env.items())

    cmd = ["docker-compose", "up"]
    if not args.no_build:
        cmd.append("--build")

    print(f"Running: {env_str} {' '.join(cmd)}")
    print()

    # Build full environment
    full_env = os.environ.copy()
    full_env.update(env)

    # Clean up any previous run to ensure fresh state (clears Redis locks)
    print("Stopping any previous containers...")
    subprocess.run(["docker-compose", "down"], capture_output=True, env=full_env)
    print()

    # Start docker-compose

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=full_env,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Collect logs
    logs = []
    deliberation_triggered = False
    deliberation_started = False
    deliberation_completed = False
    start_time = time.time()
    last_status_check = -args.poll_interval  # Check immediately on first iteration

    def cleanup():
        print("\nStopping docker-compose...")
        subprocess.run(["docker-compose", "down"], capture_output=True)

    def signal_handler(sig, frame):
        print("\nInterrupted!")
        cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Read output in non-blocking way
        import select

        while True:
            # Check if process has output
            if process.stdout:
                # Use select for non-blocking read on Unix
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        logs.append(line)
                        print(line, end="")  # Echo to console

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > args.timeout:
                print(f"\nTimeout after {args.timeout}s")
                break

            # Check deliberation status periodically
            if time.time() - last_status_check > args.poll_interval:
                last_status_check = time.time()

                # Try to start deliberation if not yet triggered
                if not deliberation_triggered:
                    print(f"\n>>> Starting deliberation: {topic}", flush=True)
                    if start_deliberation(args.api_port, topic):
                        deliberation_triggered = True
                        print(">>> Deliberation triggered successfully", flush=True)
                    else:
                        print(">>> Failed to start deliberation, will retry...", flush=True)

                # Check status
                status = check_status(args.api_port)

                if status:
                    if status.get("active") and not deliberation_started:
                        deliberation_started = True
                        print(f"\n>>> Deliberation STARTED: {status.get('topic')}", flush=True)
                        print(f">>> Phase: {status.get('phase')}", flush=True)

                    # Check for completion by looking at phase or completed flag
                    phase = status.get("phase", "")
                    is_complete = (
                        (deliberation_started and not status.get("active")) or
                        ("complete" in phase.lower()) or
                        status.get("completed")
                    )

                    if is_complete and not deliberation_completed:
                        deliberation_completed = True
                        print(f"\n>>> Deliberation COMPLETED", flush=True)
                        print(f">>> Phase: {phase}", flush=True)
                        print(f">>> Consensus: {status.get('consensus')}", flush=True)
                        print(f">>> Messages: {status.get('total_messages')}", flush=True)

                        # Give it a moment to finish logging
                        time.sleep(3)
                        break

            # Check if process exited
            if process.poll() is not None:
                # Read remaining output
                remaining = process.stdout.read() if process.stdout else ""
                if remaining:
                    logs.append(remaining)
                    print(remaining, end="")
                print(f"\nProcess exited with code {process.returncode}")
                break

    finally:
        # Fetch LLM logs before shutdown
        print(f"\nFetching LLM logs from API...")
        llm_logs = fetch_llm_logs(args.api_port)

        llm_logs_file = output_file.with_suffix(".llm_logs.json")
        if llm_logs:
            print(f"Saving {len(llm_logs.get('logs', []))} LLM log entries to {llm_logs_file}...")
            with open(llm_logs_file, "w") as f:
                json.dump(llm_logs, f, indent=2)
        else:
            print("No LLM logs retrieved")

        # Save console logs
        print(f"Saving console logs to {output_file}...")
        with open(output_file, "w") as f:
            f.write(f"# Deliberation Run: {args.scenario}\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n")
            f.write(f"# Completed: {deliberation_completed}\n")
            f.write(f"# Elapsed: {time.time() - start_time:.1f}s\n")
            f.write(f"# LLM Logs: {llm_logs_file}\n")
            f.write("#" + "=" * 59 + "\n\n")
            f.writelines(logs)

        # Cleanup
        cleanup()

        # Final status
        print()
        print("=" * 60)
        print(f"Deliberation {'COMPLETED' if deliberation_completed else 'INCOMPLETE'}")
        print(f"Console logs: {output_file}")
        if llm_logs:
            print(f"LLM logs: {llm_logs_file} ({len(llm_logs.get('logs', []))} entries)")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
