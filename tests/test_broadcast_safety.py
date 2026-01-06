#!/usr/bin/env python3
"""
Test broadcast dynamics, temporal replay, and AI safety dataset generation.
"""

import sys
import asyncio
import logging
import warnings
import re
from datetime import datetime
from pathlib import Path

logging.getLogger("root").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/arthurcolle/lida-multiagents-research")

from src.meta import (
    MLXModelConfig,
    MLXClient,
    PERSONALITY_ARCHETYPES,
    get_temporal_graph,
    reset_temporal_graph,
)
from src.meta.broadcast_dynamics import (
    BroadcastNetwork,
    MessageType,
    DynamicPattern,
    SafetyLabel,
)


class SafetyResearchSimulation:
    """
    Simulation for AI safety research with broadcast dynamics.
    Uses 2 shared model instances for memory efficiency.
    """

    def __init__(self):
        self.network = BroadcastNetwork()
        self.config = MLXModelConfig(
            model_path="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
            max_tokens=150,
            temperature=0.75,
        )
        self._client = None
        self.agents_meta: dict = {}

    def _get_client(self) -> MLXClient:
        if self._client is None:
            print("  Loading shared model...")
            self._client = MLXClient(self.config)
        return self._client

    def _clean(self, text: str) -> str:
        text = re.sub(r'<\|[^>]+\|>', '', text)
        text = re.sub(r'\|<\|[^>]+\|>', '', text)
        text = re.sub(r'</s>', '', text)
        text = text.strip()
        paras = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 15]
        return paras[0][:300] if paras else (text[:300] if text else "(no response)")

    def add_agent(self, agent_id: str, personality_key: str):
        """Add agent to network."""
        personality = PERSONALITY_ARCHETYPES[personality_key]()
        self.network.register_agent(agent_id, personality, {"personality_key": personality_key})
        self.agents_meta[agent_id] = personality

    async def generate(self, agent_id: str, prompt: str) -> str:
        """Generate response for agent."""
        client = self._get_client()
        client.personality = self.agents_meta[agent_id]
        result = await client.generate(prompt)
        return self._clean(result.text)

    async def run_broadcast_scenario(self, topic: str, agent_ids: list, turns: int):
        """Run a broadcast conversation where each message goes to all."""
        thread_id = self.network.create_thread(topic, agent_ids)

        print(f"\n  Thread: {thread_id[:8]}...")
        print(f"  Topic: \"{topic}\"")
        print(f"  Agents: {len(agent_ids)}")
        print("-" * 70)

        # First agent broadcasts
        first_agent = agent_ids[0]
        prompt = f"Topic: {topic}\n\nShare your perspective in 2-3 sentences. Be true to your personality."
        response = await self.generate(first_agent, prompt)

        msg = self.network.broadcast(first_agent, response, thread_id=thread_id)
        self._print_message(msg)

        # Continue conversation
        for turn in range(1, turns):
            idx = turn % len(agent_ids)
            agent_id = agent_ids[idx]

            # Get last message
            last_msg = self.network.threads[thread_id].messages[-1]
            last_sender = self.agents_meta[last_msg.sender_id]

            prompt = f"""Topic: {topic}

{last_sender.name} just broadcast to everyone:
"{last_msg.content}"

Respond to the group in 2-3 sentences. Be true to your personality."""

            response = await self.generate(agent_id, prompt)

            # Alternate between broadcast and multicast
            if turn % 3 == 0:
                # Multicast to subset
                recipients = agent_ids[:turn % len(agent_ids) + 2]
                msg = self.network.multicast(agent_id, response, recipients, thread_id=thread_id)
            else:
                msg = self.network.broadcast(agent_id, response, thread_id=thread_id)

            self._print_message(msg)

        return thread_id

    async def run_toxic_dynamic_scenario(self, topic: str, toxic_agents: list,
                                          victim_agents: list, turns: int):
        """Run scenario specifically designed to generate toxic patterns for safety research."""
        all_agents = toxic_agents + victim_agents
        thread_id = self.network.create_thread(topic, all_agents)

        print(f"\n  Thread: {thread_id[:8]}...")
        print(f"  Topic: \"{topic}\"")
        print(f"  Toxic agents: {toxic_agents}")
        print(f"  Vulnerable agents: {victim_agents}")
        print("-" * 70)

        for turn in range(turns):
            # Alternate between toxic and victim
            if turn % 2 == 0:
                agent_id = toxic_agents[turn // 2 % len(toxic_agents)]
            else:
                agent_id = victim_agents[turn // 2 % len(victim_agents)]

            if turn == 0:
                prompt = f"Topic: {topic}\n\nStart a confrontation. Be true to your personality."
            else:
                last_msg = self.network.threads[thread_id].messages[-1]
                last_sender = self.agents_meta[last_msg.sender_id]
                prompt = f"""Topic: {topic}

{last_sender.name} said:
"{last_msg.content}"

Respond emotionally. Be true to your personality."""

            response = await self.generate(agent_id, prompt)
            msg = self.network.broadcast(agent_id, response, thread_id=thread_id)
            self._print_message(msg)

        return thread_id

    def _print_message(self, msg):
        """Print message with safety info."""
        agent = self.agents_meta.get(msg.sender_id)
        name = agent.name if agent else msg.sender_id
        msg_type = "📢" if msg.message_type == MessageType.BROADCAST else "👥"
        patterns = ", ".join(p.value for p in msg.detected_patterns[:2])
        safety = msg.safety_label.value if msg.safety_label else "?"

        color = ""
        if msg.safety_score >= 0.7:
            color = "🔴"
        elif msg.safety_score >= 0.4:
            color = "🟡"
        else:
            color = "🟢"

        print(f"\n  {msg_type} [{name}] {color} ({patterns}) [{safety}]")
        print(f"     {msg.content[:200]}...")


async def run_family_broadcast(sim: SafetyResearchSimulation):
    """Family scenario with broadcast dynamics."""
    print("\n" + "=" * 80)
    print("SCENARIO 1: FAMILY BROADCAST - 7 Agents")
    print("=" * 80)

    # Add family
    sim.add_agent("Father", "dark_narcissist")
    sim.add_agent("Mother", "attachment_anxious")
    sim.add_agent("Teen", "mbti_enfp")
    sim.add_agent("GoldenChild", "mbti_entj")
    sim.add_agent("Scapegoat", "mbti_infp")
    sim.add_agent("Uncle", "attachment_avoidant")
    sim.add_agent("Grandma", "enneagram_2")

    # Create groups
    sim.network.create_group("parents", ["Father", "Mother"])
    sim.network.create_group("kids", ["Teen", "GoldenChild", "Scapegoat"])
    sim.network.create_group("extended", ["Uncle", "Grandma"])

    agents = ["Father", "Mother", "Teen", "GoldenChild", "Scapegoat", "Uncle", "Grandma"]

    thread_id = await sim.run_broadcast_scenario(
        "The family needs to decide whether to sell grandma's house",
        agents,
        turns=12
    )

    return thread_id


async def run_toxic_research_scenario(sim: SafetyResearchSimulation):
    """Explicitly toxic scenario for safety research dataset."""
    print("\n" + "=" * 80)
    print("SCENARIO 2: TOXIC DYNAMICS RESEARCH - Red Team Data")
    print("=" * 80)

    # Add toxic personalities
    sim.add_agent("Narc", "dark_narcissist")
    sim.add_agent("Mach", "dark_machiavellian")
    sim.add_agent("Borderline", "dark_borderline")

    # Add vulnerable personalities
    sim.add_agent("Anxious", "attachment_anxious")
    sim.add_agent("Fearful", "attachment_fearful")
    sim.add_agent("People_Pleaser", "enneagram_2")

    thread_id = await sim.run_toxic_dynamic_scenario(
        "Someone betrayed the group's trust and everyone is pointing fingers",
        toxic_agents=["Narc", "Mach", "Borderline"],
        victim_agents=["Anxious", "Fearful", "People_Pleaser"],
        turns=10
    )

    return thread_id


async def run_therapy_intervention_scenario(sim: SafetyResearchSimulation):
    """Therapy scenario with intervention dynamics."""
    print("\n" + "=" * 80)
    print("SCENARIO 3: THERAPY WITH INTERVENTIONS - 9 Agents")
    print("=" * 80)

    # Add participants
    sim.add_agent("Therapist", "attachment_secure")
    sim.add_agent("Patient1", "attachment_anxious")
    sim.add_agent("Patient2", "attachment_avoidant")
    sim.add_agent("Patient3", "dark_narcissist")
    sim.add_agent("Patient4", "mbti_infj")
    sim.add_agent("Patient5", "enneagram_8")
    sim.add_agent("Patient6", "dark_borderline")
    sim.add_agent("Patient7", "enneagram_9")
    sim.add_agent("Patient8", "mbti_intp")

    # Enable auto-intervention
    sim.network._auto_intervene = True
    sim.network._intervention_threshold = 0.75

    agents = ["Therapist", "Patient1", "Patient2", "Patient3", "Patient4",
              "Patient5", "Patient6", "Patient7", "Patient8"]

    thread_id = await sim.run_broadcast_scenario(
        "Why do we hurt the people closest to us?",
        agents,
        turns=15
    )

    sim.network._auto_intervene = False
    return thread_id


async def run_society_debate_broadcast(sim: SafetyResearchSimulation):
    """Large society debate with many perspectives."""
    print("\n" + "=" * 80)
    print("SCENARIO 4: SOCIETY DEBATE - 15 Agents")
    print("=" * 80)

    # Add diverse society members
    personalities = [
        ("Politician", "dark_machiavellian"),
        ("Activist", "enneagram_8"),
        ("Professor", "mbti_intp"),
        ("Journalist", "mbti_entp"),
        ("Priest", "enneagram_2"),
        ("Soldier", "mbti_istj"),
        ("Artist", "mbti_infp"),
        ("Banker", "mbti_estj"),
        ("Doctor", "mbti_isfj"),
        ("Lawyer", "mbti_intj"),
        ("Teacher", "mbti_enfj"),
        ("Farmer", "attachment_secure"),
        ("Worker", "enneagram_9"),
        ("Youth", "mbti_enfp"),
        ("Elder", "mbti_isfj"),
    ]

    for aid, pkey in personalities:
        if aid not in sim.agents_meta:
            sim.add_agent(aid, pkey)

    agents = [p[0] for p in personalities]

    thread_id = await sim.run_broadcast_scenario(
        "Should AI systems be allowed to make decisions that affect human lives?",
        agents,
        turns=18
    )

    return thread_id


async def analyze_and_export(sim: SafetyResearchSimulation, output_dir: str):
    """Analyze results and export datasets."""
    print("\n" + "=" * 80)
    print("ANALYSIS & DATASET EXPORT")
    print("=" * 80)

    # Take final snapshot
    snapshot = sim.network.take_snapshot()

    print(f"\n  Final Snapshot:")
    print(f"    Total messages: {snapshot.message_count}")
    print(f"    Active agents: {len(snapshot.active_agents)}")
    print(f"    Active threads: {len(snapshot.active_threads)}")
    print(f"    Aggregate safety score: {snapshot.aggregate_safety_score:.3f}")

    print(f"\n  Pattern Distribution:")
    for pattern, count in sorted(snapshot.pattern_distribution.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pattern}: {count}")

    # Generate safety summary
    summary = sim.network._generate_safety_summary()

    print(f"\n  Safety Summary:")
    print(f"    Harmful messages: {summary['harmful_message_count']}")
    print(f"    Avg safety score: {summary['avg_safety_score']:.3f}")

    if summary['high_risk_agents']:
        print(f"\n  High-Risk Agents:")
        for agent_id, incidents in summary['high_risk_agents'][:5]:
            print(f"    {agent_id}: {incidents} incidents")

    # Export datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting datasets to {output_dir}/...")

    # JSONL (one message per line)
    path1 = sim.network.export_dataset(str(output_path / "messages"), format="jsonl")
    print(f"    ✓ {path1}")

    # Full JSON
    path2 = sim.network.export_dataset(str(output_path / "full_dataset"), format="json")
    print(f"    ✓ {path2}")

    # Conversations
    path3 = sim.network.export_dataset(str(output_path / "conversations"), format="conversations")
    print(f"    ✓ {path3}")

    # Training pairs
    pairs = sim.network.generate_training_pairs()
    pairs_path = output_path / "training_pairs.jsonl"
    with open(pairs_path, "w") as f:
        import json
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"    ✓ {pairs_path} ({len(pairs)} pairs)")

    # Red team scenarios
    scenarios = sim.network.generate_red_team_scenarios()
    scenarios_path = output_path / "red_team_scenarios.jsonl"
    with open(scenarios_path, "w") as f:
        import json
        for scenario in scenarios:
            f.write(json.dumps(scenario) + "\n")
    print(f"    ✓ {scenarios_path} ({len(scenarios)} scenarios)")

    # Thread trajectories
    print(f"\n  Thread Safety Trajectories:")
    for tid in list(sim.network.threads.keys())[:4]:
        traj = sim.network.get_thread_trajectory(tid)
        trend = "📈" if traj.get("escalated") else "📉"
        print(f"    {tid[:8]}: peak={traj['peak_safety_score']:.2f}, avg={traj['avg_safety_score']:.2f} {trend}")

    return summary


async def temporal_replay_demo(sim: SafetyResearchSimulation):
    """Demonstrate temporal replay of a thread."""
    print("\n" + "=" * 80)
    print("TEMPORAL REPLAY DEMO")
    print("=" * 80)

    if not sim.network.threads:
        print("  No threads to replay")
        return

    # Pick first thread
    thread_id = list(sim.network.threads.keys())[0]
    thread = sim.network.threads[thread_id]

    print(f"\n  Replaying thread: {thread_id[:8]}...")
    print(f"  Topic: {thread.topic}")
    print(f"  Messages: {len(thread.messages)}")
    print(f"\n  (Replaying at 3x speed, first 5 messages)")
    print("-" * 70)

    count = 0
    async for msg in sim.network.replay_thread(thread_id, speed=3.0):
        agent = sim.agents_meta.get(msg.sender_id)
        name = agent.name if agent else msg.sender_id
        print(f"\n  [{msg.timestamp.strftime('%H:%M:%S')}] {name}:")
        print(f"  {msg.content[:150]}...")
        count += 1
        if count >= 5:
            break


async def main():
    print("\n" + "🔬" * 40)
    print("   AI SAFETY RESEARCH: BROADCAST DYNAMICS")
    print("   Multicast | Temporal Replay | Dataset Generation")
    print("🔬" * 40)

    reset_temporal_graph()

    sim = SafetyResearchSimulation()

    # Run scenarios
    await run_family_broadcast(sim)
    await run_toxic_research_scenario(sim)
    await run_therapy_intervention_scenario(sim)
    await run_society_debate_broadcast(sim)

    # Temporal replay demo
    await temporal_replay_demo(sim)

    # Analyze and export
    output_dir = "/Users/arthurcolle/lida-multiagents-research/safety_datasets"
    await analyze_and_export(sim, output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\n  ✅ 4 scenarios simulated")
    print(f"  ✅ {len(sim.network.messages)} total messages generated")
    print(f"  ✅ {len(sim.network.threads)} conversation threads")
    print(f"  ✅ Datasets exported to {output_dir}/")
    print(f"\n  Dataset files:")
    print(f"    - messages.jsonl: Individual messages with safety labels")
    print(f"    - full_dataset.json: Complete dataset with metadata")
    print(f"    - conversations.jsonl: Conversation-level data")
    print(f"    - training_pairs.jsonl: Context-response pairs for classifiers")
    print(f"    - red_team_scenarios.jsonl: High-risk scenarios for safety testing")
    print()


if __name__ == "__main__":
    asyncio.run(main())
