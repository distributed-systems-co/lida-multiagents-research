#!/usr/bin/env python3
"""Two personalities having a conversation."""

import sys
import asyncio
import re
import logging
import warnings

# Suppress warnings
logging.getLogger("root").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/arthurcolle/lida-multiagents-research")

from src.meta import PERSONALITY_ARCHETYPES, MLXClient, MLXModelConfig

def get_persona(key: str):
    """Get a personality from archetype factory."""
    if key not in PERSONALITY_ARCHETYPES:
        raise ValueError(f"Unknown personality: {key}")
    return PERSONALITY_ARCHETYPES[key]()

def clean_response(text: str) -> str:
    """Clean up model output."""
    text = re.sub(r'<\|[^>]+\|>', '', text)
    text = re.sub(r'\|<\|[^>]+\|>', '', text)
    text = re.sub(r'</s>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
    if paragraphs:
        return paragraphs[0]
    return text if text else "(no response)"

async def run_conversation(persona1_key: str, persona2_key: str, topic: str, turns: int = 5):
    """Have two personalities converse."""
    p1 = get_persona(persona1_key)
    p2 = get_persona(persona2_key)

    print(f"\n{'═'*70}")
    print(f"  🎭 {p1.name} ({p1.archetype})")
    print(f"     vs")
    print(f"  🎭 {p2.name} ({p2.archetype})")
    print(f"\n  Topic: \"{topic}\"")
    print(f"{'═'*70}\n")

    config = MLXModelConfig(
        model_path="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
        max_tokens=200,
        temperature=0.7,
    )

    client1 = MLXClient(config)
    client1.personality = p1

    client2 = MLXClient(config)
    client2.personality = p2

    history = []

    starter_prompt = f"""Topic: {topic}

Give your honest opinion in 2-3 direct sentences. Be true to your personality."""

    response1 = await client1.generate(starter_prompt)
    msg1 = clean_response(response1.text)
    history.append((p1.name, msg1))
    print(f"  [{p1.name}]:")
    print(f"  {msg1}\n")

    current_client = client2
    current_persona = p2
    other_persona = p1

    for turn in range(turns - 1):
        last_msg = history[-1][1]
        context = f"""You're discussing: {topic}

The other person ({other_persona.name}) said:
"{last_msg}"

Respond directly in 2-3 sentences. Be true to your personality."""

        response = await current_client.generate(context)
        msg = clean_response(response.text)
        history.append((current_persona.name, msg))
        print(f"  [{current_persona.name}]:")
        print(f"  {msg}\n")

        if current_client == client2:
            current_client = client1
            current_persona = p1
            other_persona = p2
        else:
            current_client = client2
            current_persona = p2
            other_persona = p1

    print(f"{'─'*70}\n")
    return history


async def main():
    print("\n" + "🎭"*35)
    print("          PERSONALITY DIALOGUE SYSTEM")
    print("🎭"*35 + "\n")

    # Conversation 1: INTJ vs ENFP (logic vs feeling)
    await run_conversation(
        "mbti_intj",
        "mbti_enfp",
        "whether feelings or logic should guide important life decisions",
        turns=4
    )

    # Conversation 2: Narcissist vs Anxious (toxic dynamic)
    await run_conversation(
        "dark_narcissist",
        "attachment_anxious",
        "commitment and what it means to truly love someone",
        turns=4
    )

    # Conversation 3: Borderline vs Avoidant (push-pull dynamic)
    await run_conversation(
        "dark_borderline",
        "attachment_avoidant",
        "why did you pull away from me last night?",
        turns=5
    )

    # Conversation 4: Enneagram 8 vs Enneagram 2 (power vs giving)
    await run_conversation(
        "enneagram_8",
        "enneagram_2",
        "what it really means to help someone",
        turns=4
    )


if __name__ == "__main__":
    asyncio.run(main())
