#!/usr/bin/env python3
"""
Test SOTA 2026 Models

Tests all 3 SOTA models with live GDELT feeds:
1. DeepSeek-R1-Distill-Qwen-7B (reasoning champion)
2. Qwen 2.5 14B (quality leader)
3. Llama 3.2 3B (speed king)
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.meta import MLXClient, MLXModelConfig, PERSONALITY_ARCHETYPES
from src.meta.sota_models_2026 import SOTA_2026, get_sota_config
from src.meta.industrial_intelligence import IndustrialEvent, IndustrialEventType
from run_live_quorum import GDELTLiveFeed


# Colors
class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    MAG = '\033[35m'


async def test_model_streaming(
    model_id: str,
    event: IndustrialEvent,
    personality_key: str = "mbti_intj"
):
    """Test a single model with streaming."""
    model = SOTA_2026[model_id]
    config_dict = get_sota_config(model_id)

    print(f"\n{C.CYAN}{'─'*70}{C.END}")
    print(f"{C.BOLD}Testing: {model.name}{C.END}")
    print(f"{C.DIM}Repo: {model.repo}{C.END}")
    print(f"{C.DIM}Size: {model.size_gb}GB | Released: {model.released}{C.END}")
    print(f"{C.CYAN}{'─'*70}{C.END}\n")

    # Create client
    config = MLXModelConfig(**config_dict)
    client = MLXClient(config)

    # Set personality
    if personality_key in PERSONALITY_ARCHETYPES:
        client.personality = PERSONALITY_ARCHETYPES[personality_key]()

    # Build concise prompt
    prompt = f"""Analyze this corporate event in 2-3 sentences:

Event: {event.title}
Company: {event.primary_company}

Your analysis:"""

    print(f"{C.BOLD}Streaming response:{C.END}\n")
    print(f"{C.GREEN}► {C.END}", end="", flush=True)

    start_time = time.time()
    full_response = ""
    token_count = 0

    try:
        async for token in client.generate_stream(prompt):
            print(token, end="", flush=True)
            full_response += token
            token_count += 1

            # Cap at reasonable length
            if len(full_response) > 400:
                break

    except Exception as e:
        print(f"\n{C.RED}✗ Error: {e}{C.END}")
        return None

    elapsed = time.time() - start_time
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

    print(f"{C.END}\n")
    print(f"{C.DIM}Tokens: {token_count} | Time: {elapsed:.2f}s | Speed: {tokens_per_sec:.1f} tok/s{C.END}")

    return {
        "model": model.name,
        "response": full_response,
        "tokens": token_count,
        "time": elapsed,
        "speed": tokens_per_sec
    }


async def main():
    """Test all SOTA 2026 models."""
    print(f"\n{C.BOLD}{C.MAG}{'='*70}")
    print(" TESTING SOTA 2026 MODELS")
    print(" DeepSeek-R1 | Qwen 2.5 | Llama 3.2")
    print(f"{'='*70}{C.END}\n")

    # Fetch live event
    print(f"{C.DIM}Fetching live GDELT feed...{C.END}")
    gdelt = GDELTLiveFeed()
    events = await gdelt.fetch_latest_events(limit=100)

    if events:
        relevant = gdelt.filter_relevant_events(events)
        if relevant:
            gdelt_event = relevant[0]
            company = gdelt_event.get('matched_company', 'Google')
            headline = f"{company}: {gdelt_event.get('Actor1Name', 'COMPANY')} - {gdelt_event.get('Actor2Name', 'EVENT')}"

            event = IndustrialEvent(
                event_id=gdelt_event['GLOBALEVENTID'],
                event_type=None,
                timestamp=datetime.utcnow(),
                primary_company=company,
                title=headline[:80]
            )
            print(f"{C.GREEN}✓ Live event: {event.title}{C.END}")
        else:
            # Sample event
            event = IndustrialEvent(
                event_id="test",
                event_type=IndustrialEventType.ACQUISITION_ANNOUNCED,
                timestamp=datetime.utcnow(),
                primary_company="Nvidia",
                title="Nvidia announces $20B acquisition of Cerebras Systems",
                value_billions=20.0
            )
            print(f"{C.YELLOW}Using sample event{C.END}")
    else:
        event = IndustrialEvent(
            event_id="test",
            event_type=IndustrialEventType.ACQUISITION_ANNOUNCED,
            timestamp=datetime.utcnow(),
            primary_company="Nvidia",
            title="Nvidia announces $20B acquisition of Cerebras Systems",
            value_billions=20.0
        )
        print(f"{C.YELLOW}Using sample event{C.END}")

    print(f"\n{C.BOLD}Event to analyze:{C.END}")
    print(f"  {event.title}")
    print()

    # Test each model
    results = []

    # 1. DeepSeek-R1 (Best reasoning)
    print(f"{C.BOLD}\n[1/3] DeepSeek-R1 Distill Qwen 7B{C.END}")
    print(f"{C.DIM}Expected: Best reasoning, step-by-step thinking{C.END}")
    result1 = await test_model_streaming("deepseek-r1-7b", event, "mbti_intj")
    if result1:
        results.append(result1)

    await asyncio.sleep(1)

    # 2. Qwen 2.5 (Best quality)
    print(f"{C.BOLD}\n[2/3] Qwen 2.5 14B Instruct{C.END}")
    print(f"{C.DIM}Expected: Highest quality, best instruction following{C.END}")
    result2 = await test_model_streaming("qwen-2.5-14b", event, "enneagram_6")
    if result2:
        results.append(result2)

    await asyncio.sleep(1)

    # 3. Llama 3.2 (Fastest)
    print(f"{C.BOLD}\n[3/3] Llama 3.2 3B Instruct{C.END}")
    print(f"{C.DIM}Expected: Fastest streaming, 50+ tok/s{C.END}")
    result3 = await test_model_streaming("llama-3.2-3b", event, "mbti_enfp")
    if result3:
        results.append(result3)

    # Summary
    print(f"\n{C.CYAN}{'='*70}{C.END}")
    print(f"{C.BOLD}PERFORMANCE COMPARISON{C.END}")
    print(f"{C.CYAN}{'='*70}{C.END}\n")

    if results:
        print(f"{'Model':<35} {'Tokens':<10} {'Time':<10} {'Speed':<15}")
        print(f"{C.DIM}{'─'*70}{C.END}")

        for r in results:
            print(f"{r['model']:<35} {r['tokens']:<10} {r['time']:<10.2f}s {r['speed']:<10.1f} tok/s")

        # Find fastest
        fastest = max(results, key=lambda x: x['speed'])
        print(f"\n{C.GREEN}⚡ Fastest: {fastest['model']} ({fastest['speed']:.1f} tok/s){C.END}")

        # Memory used
        total_mem = sum(SOTA_2026[mid].size_gb for mid in ["deepseek-r1-7b", "qwen-2.5-14b", "llama-3.2-3b"])
        print(f"{C.DIM}💾 Total memory: {total_mem:.1f}GB / 15GB budget{C.END}")

    print(f"\n{C.GREEN}{C.BOLD}✓ SOTA 2026 Testing Complete!{C.END}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted.{C.END}")
    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.END}")
        import traceback
        traceback.print_exc()
