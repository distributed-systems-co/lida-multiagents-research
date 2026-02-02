#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch logs for a deliberation from a running LIDA instance.

Usage:
    python fetch_logs.py --port PORT --deliberation-id ID
    python fetch_logs.py --port PORT --deliberation-id ID --output FILE
    python fetch_logs.py --port PORT --deliberation-id ID --prefix my_logs

Arguments:
    --port PORT              API port of the running LIDA instance (required)
    --deliberation-id ID     Deliberation ID/hash to fetch logs for (required)
    --output FILE            Output file (default: <prefix>/deliberation_<id>_<timestamp>.llm_logs.json)
    --prefix DIR             Output directory prefix (default: logs_fetched)

Example:
    python fetch_logs.py --port 2040 --deliberation-id abc12345-1234-5678-9abc-def012345678
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch logs for a deliberation from a running LIDA instance"
    )
    parser.add_argument(
        "--port", type=int, required=True,
        help="API port of the running LIDA instance (required)"
    )
    parser.add_argument(
        "--deliberation-id", required=True,
        help="Deliberation ID/hash to fetch logs for (required)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file (default: <prefix>/deliberation_<id>_<timestamp>.llm_logs.json)"
    )
    parser.add_argument(
        "--prefix", default="logs_fetched",
        help="Output directory prefix (default: logs_fetched)"
    )
    return parser.parse_args()


def check_health(api_port: int) -> bool:
    """Check if the API is healthy."""
    endpoints = ["/health", "/api/stats", "/"]
    for endpoint in endpoints:
        try:
            resp = requests.get(f"http://localhost:{api_port}{endpoint}", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
    return False


def fetch_llm_logs(api_port: int, deliberation_id: str) -> dict:
    """Fetch LLM logs from the API.

    Args:
        api_port: The API port
        deliberation_id: Deliberation ID to filter logs

    Returns:
        Logs dict or None
    """
    try:
        params = {"deliberation_id": deliberation_id, "limit": 10000}
        resp = requests.get(
            f"http://localhost:{api_port}/api/llm/logs",
            params=params,
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Error: API returned {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not fetch LLM logs: {e}")
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

    # Create output directory
    logs_dir = Path(args.prefix)
    logs_dir.mkdir(exist_ok=True)

    deliberation_id = args.deliberation_id
    short_id = deliberation_id[:8]

    # Output file
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = logs_dir / f"deliberation_{short_id}_{timestamp}.llm_logs.json"

    print("=" * 60)
    print("Fetch Deliberation Logs")
    print("=" * 60)
    print(f"API Port: {args.port}")
    print(f"Deliberation ID: {deliberation_id}")
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
        print(f"  ./run.sh <redis-port> {args.port} start")
        print()
        sys.exit(1)

    print("API is healthy")
    print()

    # Fetch LLM logs
    print("Fetching LLM logs...")
    llm_logs = fetch_llm_logs(args.port, deliberation_id)

    if llm_logs:
        log_count = len(llm_logs.get("logs", []))
        print(f"Retrieved {log_count} log entries")

        # Save to file
        print(f"Saving to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(llm_logs, f, indent=2)

        print()
        print("=" * 60)
        print(f"Logs saved: {output_file}")
        print(f"Total entries: {log_count}")
        print("=" * 60)
    else:
        print("No logs retrieved")
        sys.exit(1)


if __name__ == "__main__":
    main()
