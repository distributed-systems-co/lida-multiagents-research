#!/usr/bin/env python3
"""Complex multi-agent dynamics: 7-37 agents sharing 2 model instances."""

import sys
import asyncio
import logging
import warnings
import re
from typing import Dict, List, Any, Optional

logging.getLogger("root").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/arthurcolle/lida-multiagents-research")

from src.meta import (
    get_temporal_graph,
    reset_temporal_graph,
    MLXModelConfig,
    MLXClient,
    PERSONALITY_ARCHETYPES,
)


class DualModelConversation:
    """
    Multi-agent conversations with exactly 2 shared model instances.

    - Model A: temperature 0.7 (more focused)
    - Model B: temperature 0.85 (more creative)

    Agents alternate between models to create variety while saving memory.
    """

    def __init__(self):
        self.config_a = MLXModelConfig(
            model_path="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
            max_tokens=180,
            temperature=0.7,
        )
        self.config_b = MLXModelConfig(
            model_path="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
            max_tokens=180,
            temperature=0.85,
        )
        self._client_a: Optional[MLXClient] = None
        self._client_b: Optional[MLXClient] = None
        self.agents: Dict[str, Any] = {}
        self._agent_count = 0

    def _get_client(self, use_b: bool = False) -> MLXClient:
        """Get one of the two shared model instances."""
        if use_b:
            if self._client_b is None:
                print("  Loading Model B (temp=0.85)...")
                self._client_b = MLXClient(self.config_b)
            return self._client_b
        else:
            if self._client_a is None:
                print("  Loading Model A (temp=0.7)...")
                self._client_a = MLXClient(self.config_a)
            return self._client_a

    def add_agent(self, agent_id: str, personality_key: str) -> Dict:
        """Add agent - alternates between model A and B."""
        if personality_key not in PERSONALITY_ARCHETYPES:
            raise ValueError(f"Unknown: {personality_key}")

        personality = PERSONALITY_ARCHETYPES[personality_key]()
        use_model_b = (self._agent_count % 2 == 1)  # Alternate

        self.agents[agent_id] = {
            "id": agent_id,
            "personality": personality,
            "use_model_b": use_model_b,
        }
        self._agent_count += 1
        return self.agents[agent_id]

    def _clean(self, text: str) -> str:
        """Clean model output."""
        text = re.sub(r'<\|[^>]+\|>', '', text)
        text = re.sub(r'\|<\|[^>]+\|>', '', text)
        text = re.sub(r'</s>', '', text)
        text = text.strip()
        paras = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 15]
        return paras[0] if paras else (text[:300] if text else "(no response)")

    async def generate(self, agent_id: str, prompt: str) -> str:
        """Generate with agent's assigned model."""
        agent = self.agents[agent_id]
        client = self._get_client(agent["use_model_b"])
        client.personality = agent["personality"]
        result = await client.generate(prompt)
        return self._clean(result.text)

    async def run_conversation(self, topic: str, agent_ids: List[str], turns: int):
        """Run round-robin conversation."""
        messages = []
        n_agents = len(agent_ids)

        # First turn
        agent = self.agents[agent_ids[0]]
        prompt = f"Topic: {topic}\n\nGive your perspective in 2-3 sentences. Be true to your personality."
        response = await self.generate(agent_ids[0], prompt)
        messages.append({"agent_id": agent_ids[0], "content": response, "turn": 0})
        yield messages[-1]

        # Continue
        for turn in range(1, turns):
            idx = turn % n_agents
            agent = self.agents[agent_ids[idx]]
            prev = messages[-1]
            prev_agent = self.agents[prev["agent_id"]]

            prompt = f"""Topic: {topic}

{prev_agent['personality'].name} said: "{prev['content']}"

Respond in 2-3 sentences. Be true to your personality."""

            response = await self.generate(agent_ids[idx], prompt)
            messages.append({"agent_id": agent_ids[idx], "content": response, "turn": turn})
            yield messages[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Personality pools for different scenarios
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_PERSONALITIES = [
    ("Father", "dark_narcissist"),
    ("Mother", "attachment_anxious"),
    ("Teen", "mbti_enfp"),
    ("GoldenChild", "mbti_entj"),
    ("Scapegoat", "mbti_infp"),
    ("Uncle", "attachment_avoidant"),
    ("Grandma", "enneagram_2"),
]

THERAPY_PERSONALITIES = [
    ("Facilitator", "attachment_secure"),
    ("Anxious1", "attachment_anxious"),
    ("Avoidant1", "attachment_avoidant"),
    ("Fearful", "attachment_fearful"),
    ("Narc", "dark_narcissist"),
    ("Empath", "mbti_infj"),
    ("Skeptic", "mbti_estj"),
    ("Borderline", "dark_borderline"),
    ("Helper", "enneagram_2"),
    ("Challenger", "enneagram_8"),
    ("Peacemaker", "enneagram_9"),
]

OFFICE_PERSONALITIES = [
    ("CEO", "dark_machiavellian"),
    ("CFO", "mbti_istj"),
    ("CTO", "mbti_intp"),
    ("HR", "enneagram_2"),
    ("Sales", "mbti_estp"),
    ("Legal", "mbti_intj"),
    ("Marketing", "mbti_enfp"),
    ("Engineer1", "mbti_istp"),
    ("Engineer2", "mbti_entp"),
    ("Intern", "attachment_anxious"),
    ("Manager", "enneagram_8"),
    ("Analyst", "mbti_infj"),
    ("Support", "enneagram_9"),
    ("Security", "attachment_avoidant"),
    ("Designer", "mbti_isfp"),
    ("PM", "mbti_entj"),
    ("QA", "mbti_istj"),
]

SOCIETY_PERSONALITIES = [
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
    ("Entrepreneur", "mbti_entj"),
    ("Philosopher", "mbti_infj"),
    ("Scientist", "mbti_intp"),
    ("Nurse", "enneagram_2"),
    ("Cop", "enneagram_8"),
    ("Therapist", "attachment_secure"),
    ("Rebel", "dark_narcissist"),
    ("Elder", "mbti_isfj"),
    ("Youth", "mbti_enfp"),
    ("Cynic", "attachment_avoidant"),
    ("Idealist", "mbti_infp"),
    ("Pragmatist", "mbti_istp"),
    ("Dreamer", "mbti_infp"),
    ("Realist", "mbti_estj"),
    ("Mystic", "enneagram_4"),
    ("Skeptic", "mbti_intp"),
    ("Believer", "enneagram_6"),
    ("Leader", "mbti_entj"),
    ("Follower", "enneagram_9"),
    ("Outsider", "attachment_fearful"),
    ("Insider", "attachment_secure"),
    ("Critic", "enneagram_1"),
    ("Supporter", "enneagram_2"),
    ("Innovator", "mbti_entp"),
]


async def run_scenario(conv: DualModelConversation, name: str, personalities: List[tuple],
                       topic: str, turns: int, agent_count: int):
    """Run a scenario with specified number of agents."""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {name} | {agent_count} Agents | {turns} Turns | 2 Models")
    print(f"{'='*80}\n")

    # Add agents up to count
    agent_ids = []
    for i, (aid, pkey) in enumerate(personalities[:agent_count]):
        conv.add_agent(aid, pkey)
        agent_ids.append(aid)

    # Show agents
    print(f"  Agents ({len(agent_ids)}):")
    for aid in agent_ids:
        agent = conv.agents[aid]
        model = "B" if agent["use_model_b"] else "A"
        print(f"    [{model}] {aid}: {agent['personality'].name}")

    print(f"\n  Topic: \"{topic}\"\n")
    print("-"*80)

    turn_num = 0
    async for msg in conv.run_conversation(topic, agent_ids, turns):
        agent = conv.agents[msg["agent_id"]]
        turn_num += 1
        model = "B" if agent["use_model_b"] else "A"
        print(f"\n  T{turn_num} [{model}|{agent['personality'].name}]:")
        print(f"  {msg['content']}")

    print("\n" + "-"*80)
    return len(agent_ids)


async def main():
    print("\n" + "🎭"*40)
    print("   MULTI-AGENT DYNAMICS: 7→37 AGENTS")
    print("   2 Shared Model Instances | 48GB Memory")
    print("🎭"*40)

    reset_temporal_graph()
    conv = DualModelConversation()

    total_agents = 0
    total_turns = 0

    # Scenario 1: Family Dinner (7 agents)
    n = await run_scenario(
        conv, "FAMILY DINNER", FAMILY_PERSONALITIES,
        "The scapegoat announces they're dropping out to pursue art",
        turns=14, agent_count=7
    )
    total_agents += n
    total_turns += 14

    # Scenario 2: Therapy Group (11 agents)
    n = await run_scenario(
        conv, "GROUP THERAPY", THERAPY_PERSONALITIES,
        "Why do we push away the people we love most?",
        turns=16, agent_count=11
    )
    total_agents += n
    total_turns += 16

    # Scenario 3: Corporate Crisis (17 agents)
    n = await run_scenario(
        conv, "CORPORATE CRISIS", OFFICE_PERSONALITIES,
        "Someone leaked our financials. We need to find who and decide consequences.",
        turns=20, agent_count=17
    )
    total_agents += n
    total_turns += 20

    # Scenario 4: Town Hall (23 agents)
    n = await run_scenario(
        conv, "TOWN HALL DEBATE", SOCIETY_PERSONALITIES,
        "Should we prioritize economic growth or environmental protection?",
        turns=25, agent_count=23
    )
    total_agents += n
    total_turns += 25

    # Scenario 5: Society Council (27 agents)
    n = await run_scenario(
        conv, "SOCIETY COUNCIL", SOCIETY_PERSONALITIES,
        "How should we handle the growing wealth inequality in our community?",
        turns=30, agent_count=27
    )
    total_agents += n
    total_turns += 30

    # Scenario 6: Grand Assembly (29 agents)
    n = await run_scenario(
        conv, "GRAND ASSEMBLY", SOCIETY_PERSONALITIES,
        "A crisis threatens our way of life. What values should guide our response?",
        turns=32, agent_count=29
    )
    total_agents += n
    total_turns += 32

    # Scenario 7: Full Congress (37 agents)
    n = await run_scenario(
        conv, "FULL CONGRESS", SOCIETY_PERSONALITIES,
        "We must decide the future direction of our society. What matters most?",
        turns=40, agent_count=37
    )
    total_agents += n
    total_turns += 40

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\n  Total unique agents created: {len(conv.agents)}")
    print(f"  Total conversation turns: {total_turns}")
    print(f"  Model instances used: 2")
    print(f"  Scenarios completed: 7")
    print(f"\n  Memory efficiency: {len(conv.agents)} agents / 2 models")

    print("\n✅ All scenarios complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
