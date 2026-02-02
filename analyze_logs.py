#!/usr/bin/env python3
"""Analyze deliberation logs and generate a summary CSV."""

import json
import os
import re
import csv
from pathlib import Path
from collections import defaultdict

# Question mapping - key phrases to Q numbers (original logs/)
QUESTION_MAPPING = {
    "requiring all new AI accelerators to include": "Q1",
    "requiring 80% of existing datacenter AI chips": "Q2",
    "mandating hardware-rooted attestation to verify training run": "Q3",
    "requiring AI chips to use offline licensing": "Q4",
    "embedding location verification mechanisms in AI chips": "Q5",
    "requiring chip manufacturers to implement hardware-enabled mechanisms": "Q6",
    "chips automatically entering reduced-capability mode": "Q7",
    "requiring compute usage reporting through privacy-preserving mechanisms": "Q8",
    "automatic escalation from anonymized reporting to detailed": "Q9",
    "requiring AI training infrastructure to use only": "Q10",
    "requiring additional government approval for AI hardware": "Q11",
    "mandating tamper-evident mechanisms in AI chips": "Q12",
    "establishing an international technical body to maintain": "Q13",
    "requiring participation in this international body": "Q14",
    "hardware-level enforcement of non-compliance": "Q15",
    "requiring 72-hour incident notification": "Q16",
    "linking dangerous capability evaluations to hardware-enforced": "Q17",
    "requiring 90-day advance notification for training runs": "Q18",
    "automatic flagging for government review when actual": "Q19",
    "using hardware-enabled mechanisms as the primary enforcement": "Q20",
    "trade restrictions on AI-related exports": "Q21",
    "multilateral technical secretariat publishing anonymized aggregate": "Q22",
}

# Policy mapping - key phrases to P numbers (logs-test2/)
POLICY_MAPPING = {
    "Forced reporting of transparency on how monitorable CoT": "P1",
    "Every country agrees to buy chips which report on a blockchain": "P2",
    "Every country agrees to run software which reports": "P3",
    "All frontier AI developers must report quarterly to": "P4",
    "All signatory countries agree that all AI accelerators": "P5",
    "All countries agree to require datacenter operators hosting": "P6",
    "Frontier AI developers must implement security measures corresponding": "P7",
    "Signatory nations establish a 72-hour notification protocol": "P8",
    "All organizations developing frontier AI models must conduct annual security assessments": "P9",
    "Organizations handling frontier AI model weights must implement insider threat": "P10",
    "Organizations handling frontier AI model weights must implement": "P10",  # shorter match
    "Frontier AI developers must maintain comprehensive software bills": "P11",
    "Organizations planning training runs expected to consume more": "P12",
    "Organizations must evaluate frontier models against standardized dangerous": "P13",
}


def get_question_id(topic: str) -> str:
    """Map topic text to Q1-Q22 or P1-P13 identifier."""
    # Check P mapping first (more specific, case-sensitive)
    for key_phrase, p_id in POLICY_MAPPING.items():
        if key_phrase in topic:
            return p_id
    # Check Q mapping (case-insensitive)
    topic_lower = topic.lower()
    for key_phrase, q_id in QUESTION_MAPPING.items():
        if key_phrase.lower() in topic_lower:
            return q_id
    return "?"


def extract_topic_full(prompt: str) -> str:
    """Extract full topic from prompt."""
    # Look for "TOPIC FOR DELIBERATION:" pattern
    match = re.search(r'TOPIC FOR DELIBERATION:\s*(.+?)(?:\n\n|VOTING)', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: take first line
    return prompt.split('\n')[0].strip()


def truncate_topic(topic: str) -> str:
    """Truncate topic for display."""
    # For topics starting with common prefix, extract more words
    if topic.startswith("Do you support the following policy"):
        # Extract 30 words to get past the common prefix and capture key phrases
        words = topic.split()[:30]
    else:
        # Get first 10 words
        words = topic.split()[:10]
    return ' '.join(words)


def parse_vote_response(response: str) -> dict:
    """Parse vote from response text."""
    result = {"vote": None, "confidence": None}

    # Extract VOTE
    vote_match = re.search(r'VOTE:\s*(\w+)', response, re.IGNORECASE)
    if vote_match:
        result["vote"] = vote_match.group(1).upper()

    # Extract CONFIDENCE
    conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
    if conf_match:
        result["confidence"] = float(conf_match.group(1))

    return result


def extract_vote_history_from_prompt(prompt: str) -> dict:
    """Extract vote history from prompt for tracking changes."""
    history = defaultdict(list)  # agent_name -> [votes per round]

    # Find all round results in the prompt
    round_pattern = r'═══ ROUND (\d+) RESULTS ═══.*?(?=═══ ROUND|\Z)'
    rounds = re.findall(round_pattern, prompt, re.DOTALL)

    # Find individual votes per agent
    # Pattern: • Agent Name: VOTE (confidence: X%)
    vote_pattern = r'•\s*([^:]+):\s*(SUPPORT|OPPOSE|MODIFY|ABSTAIN)\s*\(confidence'

    for match in re.finditer(vote_pattern, prompt, re.IGNORECASE):
        agent_name = match.group(1).strip()
        vote = match.group(2).upper()
        history[agent_name].append(vote)

    return history


def analyze_log_file(filepath: str) -> dict:
    """Analyze a single log file and extract relevant data."""
    with open(filepath) as f:
        data = json.load(f)

    logs = data.get('logs', [])
    if not logs:
        return None

    # Extract deliberation ID from filename
    filename = os.path.basename(filepath)
    delib_id_match = re.search(r'_([a-f0-9]{8})_', filename)
    delib_id = delib_id_match.group(1) if delib_id_match else filename

    topic_full = None
    final_votes = {}  # agent_name -> (round_num, vote)
    vote_history = defaultdict(dict)  # agent_name -> {round_num: vote}

    # Find the final round votes by looking at entries with voting prompts
    for log in logs:
        prompt = log.get('prompt', '')
        response = log.get('response', '')
        agent_name = log.get('agent_name', 'Unknown')

        # Extract topic from first relevant prompt
        if topic_full is None and 'TOPIC FOR DELIBERATION:' in prompt:
            topic_full = extract_topic_full(prompt)

        # Extract vote history from prompt (includes all previous rounds)
        if 'VOTING ROUND' in prompt:
            round_match = re.search(r'VOTING ROUND (\d+)', prompt)
            round_num = int(round_match.group(1)) if round_match else 0

            # Parse this agent's current vote from response
            vote_data = parse_vote_response(response)
            if vote_data["vote"]:
                # Track votes per agent per round
                vote_history[agent_name][round_num] = vote_data["vote"]

                # Track final votes - only keep the HIGHEST round number
                current_round, _ = final_votes.get(agent_name, (0, None))
                if round_num >= current_round:
                    final_votes[agent_name] = (round_num, vote_data["vote"])

    # Build vote sequences per agent (sorted by round)
    vote_sequences = {}  # agent_name -> [vote1, vote2, ...]
    for agent_name, rounds_dict in vote_history.items():
        sorted_rounds = sorted(rounds_dict.keys())
        vote_sequences[agent_name] = [rounds_dict[r] for r in sorted_rounds]

    # Identify vote changers
    vote_changers = []
    for agent_name, votes in vote_sequences.items():
        if len(votes) >= 2:
            # Check if any votes changed
            unique_votes = set(votes)
            if len(unique_votes) > 1:
                vote_changers.append(f"{agent_name}: {' -> '.join(votes)}")

    # Count final votes (extract just the vote from (round_num, vote) tuple)
    vote_counts = defaultdict(int)
    actual_final_votes = {}
    for agent_name, (round_num, vote) in final_votes.items():
        vote_counts[vote] += 1
        actual_final_votes[agent_name] = vote

    topic_text = topic_full or "Unknown topic"
    return {
        "delib_id": delib_id,
        "question_id": get_question_id(topic_text),
        "topic": truncate_topic(topic_text),
        "final_votes": actual_final_votes,
        "vote_counts": dict(vote_counts),
        "vote_changers": vote_changers,
        "total_agents": len(final_votes),
    }


def main():
    import sys

    # Allow specifying logs directory and output file as arguments
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
    csv_file = sys.argv[2] if len(sys.argv) > 2 else "deliberation_results.csv"

    results = []

    # Find all .llm_logs.json files
    log_files = sorted(logs_dir.glob("*.llm_logs.json"))

    print(f"Found {len(log_files)} log files to analyze...")

    for filepath in log_files:
        try:
            result = analyze_log_file(str(filepath))
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    # Sort by Q number (extract numeric part for proper sorting)
    def q_sort_key(r):
        q_id = r.get("question_id", "Q99")
        try:
            return int(q_id[1:])  # Extract number after 'Q'
        except (ValueError, IndexError):
            return 99

    results.sort(key=q_sort_key)

    # Write CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question",
            "Deliberation ID",
            "Topic (first 10 words)",
            "Support",
            "Oppose",
            "Modify",
            "Abstain",
            "Total Agents",
            "Vote Changers"
        ])

        for r in results:
            counts = r["vote_counts"]
            changers = "; ".join(r["vote_changers"]) if r["vote_changers"] else "None"
            writer.writerow([
                r["question_id"],
                r["delib_id"],
                r["topic"],
                counts.get("SUPPORT", 0),
                counts.get("OPPOSE", 0),
                counts.get("MODIFY", 0),
                counts.get("ABSTAIN", 0),
                r["total_agents"],
                changers
            ])

    print(f"\nResults written to {csv_file}")

    # Also print to console as table
    print("\n" + "=" * 130)
    print(f"{'Q':<4} {'ID':<10} {'Topic':<50} {'S':>3} {'O':>3} {'M':>3} {'A':>3} {'N':>3} {'Vote Changers'}")
    print("=" * 130)

    for r in results:
        counts = r["vote_counts"]
        changers = "; ".join(r["vote_changers"][:2]) if r["vote_changers"] else "None"
        if len(changers) > 35:
            changers = changers[:32] + "..."
        topic_short = r["topic"][:48] + "..." if len(r["topic"]) > 48 else r["topic"]
        print(f"{r['question_id']:<4} {r['delib_id']:<10} {topic_short:<50} {counts.get('SUPPORT', 0):>3} {counts.get('OPPOSE', 0):>3} {counts.get('MODIFY', 0):>3} {counts.get('ABSTAIN', 0):>3} {r['total_agents']:>3} {changers}")

    print("=" * 120)
    print(f"\nTotal deliberations: {len(results)}")

    # Summary stats
    total_support = sum(r["vote_counts"].get("SUPPORT", 0) for r in results)
    total_oppose = sum(r["vote_counts"].get("OPPOSE", 0) for r in results)
    total_changers = sum(len(r["vote_changers"]) for r in results)

    print(f"Total Support votes: {total_support}")
    print(f"Total Oppose votes: {total_oppose}")
    print(f"Total vote changers: {total_changers}")


if __name__ == "__main__":
    main()
