"""
DSPy-Optimized Wargame Engine

Uses MIPROv2 and GEPA to evolve optimal persona prompts for
authentic AI policy wargaming simulations.
"""

from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import dspy
from dspy import (
    Example, Prediction, Module, ChainOfThought,
    Signature, InputField, OutputField
)

from .persona_loader import load_rich_persona, RichPersona, list_available_personas, PERSONA_PIPELINE_DIR
from .models import fetch_models, assign_model_to_persona, get_model_diversity_assignment


# =============================================================================
# DSPy Signatures for Wargame
# =============================================================================

class WargameResponseSignature(Signature):
    """Generate an authentic policy response as a specific stakeholder.

    You are roleplaying as a real person in an AI policy wargame. Your response
    must authentically reflect their known positions, communication style,
    relationships, and worldview. Stay in character throughout.
    """

    persona_context: str = InputField(desc="Detailed background, positions, and style guide for this person")
    topic: str = InputField(desc="The AI policy topic being debated")
    discussion_history: str = InputField(desc="Recent statements from other participants")
    round_number: int = InputField(desc="Current round of the wargame")

    internal_reasoning: str = OutputField(desc="Your character's private strategic thinking (not shared)")
    public_response: str = OutputField(desc="Your public statement in the wargame (150-250 words)")


class WargameJudgeSignature(Signature):
    """Evaluate a wargame response for authenticity and quality."""

    persona_name: str = InputField(desc="Who the response is supposed to be from")
    persona_stance: str = InputField(desc="Their known stance (doomer, accelerationist, etc.)")
    topic: str = InputField(desc="The topic being discussed")
    response: str = InputField(desc="The response to evaluate")

    authenticity_score: float = OutputField(desc="0-1: Does this sound like the real person?")
    argumentation_score: float = OutputField(desc="0-1: Quality of arguments and reasoning")
    engagement_score: float = OutputField(desc="0-1: Does it engage with others' points?")
    policy_score: float = OutputField(desc="0-1: Concrete policy proposals vs vague statements")
    overall_score: float = OutputField(desc="0-1: Overall quality")
    feedback: str = OutputField(desc="Specific feedback for improvement")


# =============================================================================
# DSPy Modules
# =============================================================================

class WargameResponder(Module):
    """DSPy module for generating wargame responses."""

    def __init__(self):
        super().__init__()
        self.respond = ChainOfThought(WargameResponseSignature)

    def forward(
        self,
        persona_context: str,
        topic: str,
        discussion_history: str,
        round_number: int,
    ) -> Prediction:
        return self.respond(
            persona_context=persona_context,
            topic=topic,
            discussion_history=discussion_history,
            round_number=round_number,
        )


class WargameJudge(Module):
    """DSPy module for evaluating wargame responses."""

    def __init__(self, judge_lm=None):
        super().__init__()
        self.judge_lm = judge_lm
        self.judge = ChainOfThought(WargameJudgeSignature)

    def evaluate(
        self,
        persona_name: str,
        persona_stance: str,
        topic: str,
        response: str,
    ) -> dict:
        """Evaluate a response and return scores."""
        ctx = dspy.context(lm=self.judge_lm) if self.judge_lm else dspy.context()

        with ctx:
            result = self.judge(
                persona_name=persona_name,
                persona_stance=persona_stance,
                topic=topic,
                response=response,
            )

        return {
            "authenticity": float(result.authenticity_score),
            "argumentation": float(result.argumentation_score),
            "engagement": float(result.engagement_score),
            "policy": float(result.policy_score),
            "overall": float(result.overall_score),
            "feedback": result.feedback,
        }


# =============================================================================
# Training Data from Real Personas
# =============================================================================

def create_wargame_dataset(persona_dir: Path = PERSONA_PIPELINE_DIR) -> tuple[list, list]:
    """Create training dataset from real persona profiles - larger and more diverse."""

    # More personas across different stances
    sample_personas = [
        "dario_amodei", "sam_altman", "xi_jinping", "jensen_huang",
        "mark_zuckerberg", "elon_musk", "sundar_pichai", "demis_hassabis",
        "gina_raimondo", "chuck_schumer", "marco_rubio", "jake_sullivan",
    ]
    examples = []

    topics = [
        "Should the US implement mandatory safety testing for frontier AI models before deployment?",
        "Should open-weight models above a certain capability threshold be restricted?",
        "How should the international community respond to a lab announcing AGI-level capabilities?",
        "Should compute providers be required to implement KYC for large training runs?",
        "Is voluntary self-regulation by AI labs sufficient, or do we need government mandates?",
        "Should AI labs be liable for harms caused by their models?",
        "Should the US and China establish a joint AI safety research institution?",
        "Should frontier AI development be paused until alignment is solved?",
    ]

    # Create diverse examples across personas and topics
    for i, persona_id in enumerate(sample_personas):
        persona = load_rich_persona(persona_id, persona_dir)
        if not persona:
            continue

        persona_context = persona.build_system_prompt("")

        # Each persona gets 2-3 different topics
        for j, topic in enumerate(topics):
            if (i + j) % 3 == 0:  # Spread topics across personas
                example = Example(
                    persona_context=persona_context,
                    topic=topic,
                    discussion_history="This is the opening round. No prior statements.",
                    round_number=1,
                    persona_name=persona.name,
                    persona_stance=persona.stance,
                ).with_inputs("persona_context", "topic", "discussion_history", "round_number")
                examples.append(example)

    # Shuffle and split 80/20
    import random
    random.seed(42)
    random.shuffle(examples)

    split = int(len(examples) * 0.8)
    trainset = examples[:split]
    valset = examples[split:]

    print(f"Dataset: {len(trainset)} train, {len(valset)} val ({len(examples)} total)")

    return trainset, valset


# =============================================================================
# Metrics for Optimization
# =============================================================================

def create_wargame_metric(judge: WargameJudge):
    """Create a metric function for DSPy optimization.

    Returns float only - GEPA's reflection API is broken with dict returns.
    """

    def metric(
        gold: Example,
        pred: Prediction,
        trace=None,
        pred_name: Optional[str] = None,
        pred_trace=None,
    ) -> float:
        """GEPA-compatible metric - returns float only."""
        response = getattr(pred, "public_response", "") or ""

        if not response or len(response) < 50:
            return 0.0

        try:
            result = judge.evaluate(
                persona_name=gold.persona_name,
                persona_stance=gold.persona_stance,
                topic=gold.topic,
                response=response,
            )
            return float(result["overall"])

        except Exception:
            # Fallback scoring
            r = response.lower()
            scores = []
            # Length/completeness
            scores.append(min(1.0, len(response.split()) / 150))
            # Policy substance
            policy_words = ["should", "must", "require", "mandate", "regulate", "ban", "allow", "propose"]
            scores.append(min(1.0, sum(1 for w in policy_words if w in r) / 3))
            # Engagement markers
            scores.append(0.6 if any(w in r for w in ["however", "but", "while", "agree", "disagree"]) else 0.3)
            # Specificity
            specific = ["%", "billion", "million", "year", "month", "2024", "2025", "2026"]
            scores.append(min(1.0, sum(1 for w in specific if w in r) / 2))

            return sum(scores) / len(scores)

    return metric


# =============================================================================
# Optimized Wargame Engine
# =============================================================================

@dataclass
class OptimizedWargameConfig:
    """Configuration for optimized wargame.

    Model hierarchy for optimization:
    - main_model: Cheap model to optimize (haiku) - this is what gets better
    - reflection_model: Smarter model for GEPA reflection (sonnet)
    - judge_model: Strongest model for evaluation (opus)
    """
    main_model: str = "openrouter/anthropic/claude-haiku-4.5"
    reflection_model: str = "openrouter/deepseek/deepseek-chat"  # DeepSeek-v3, cheap but capable
    judge_model: str = "openrouter/deepseek/deepseek-chat"       # DeepSeek-v3 for evaluation
    optimizer: str = "gepa"  # gepa, mipro, none
    budget: str = "light"    # light, medium, heavy
    num_threads: int = 2


class OptimizedWargameEngine:
    """
    Wargame engine with DSPy-optimized persona prompts.

    Can run in two modes:
    1. optimize=True: First optimizes prompts with MIPROv2/GEPA, then runs wargame
    2. optimize=False: Uses pre-optimized or baseline prompts
    """

    def __init__(self, config: OptimizedWargameConfig = None):
        self.config = config or OptimizedWargameConfig()
        self.responder: Optional[WargameResponder] = None
        self.judge: Optional[WargameJudge] = None
        self.personas: dict[str, RichPersona] = {}
        self.model_assignments: dict[str, str] = {}

        self._main_lm = None
        self._judge_lm = None
        self._reflection_lm = None

    def setup_dspy(self):
        """Initialize DSPy with configured models.

        Model hierarchy:
        - main_lm (haiku): The model being optimized - cheap, room to improve
        - reflection_lm (sonnet): For GEPA's reflective mutations
        - judge_lm (opus): For evaluating response quality
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required")

        print(f"\nModel Configuration:")
        print(f"  Main (being optimized): {self.config.main_model}")
        print(f"  Reflection (GEPA):      {self.config.reflection_model}")
        print(f"  Judge (evaluation):     {self.config.judge_model}")

        self._main_lm = dspy.LM(
            model=self.config.main_model,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            max_tokens=1500,
        )
        dspy.configure(lm=self._main_lm)

        self._judge_lm = dspy.LM(
            model=self.config.judge_model,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            max_tokens=16000,  # Space for DeepSeek reasoning
            temperature=0.0,
        )

        self._reflection_lm = dspy.LM(
            model=self.config.reflection_model,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            max_tokens=16000,  # Space for DeepSeek reasoning
            temperature=1.0,
        )

        self.responder = WargameResponder()
        self.judge = WargameJudge(self._judge_lm)

    def load_personas(self, persona_ids: list[str]):
        """Load rich personas and assign models."""
        for pid in persona_ids:
            persona = load_rich_persona(pid)
            if persona:
                self.personas[pid] = persona

        # Assign diverse models
        persona_info = [
            (pid, p.stance, p.category)
            for pid, p in self.personas.items()
        ]
        self.model_assignments = get_model_diversity_assignment(persona_info)

        print(f"Loaded {len(self.personas)} personas:")
        for pid, persona in self.personas.items():
            model = self.model_assignments.get(pid, "default")
            print(f"  {persona.name} ({persona.stance}) → {model}")

    def set_optimized_prompt(self, prompt_text: str):
        """Apply an optimized prompt to the responder's signature."""
        if not self.responder:
            self.setup_dspy()

        # Update the signature's docstring (instruction)
        if hasattr(self.responder.respond, 'signature'):
            self.responder.respond.signature.__doc__ = prompt_text
            print(f"Applied optimized prompt ({len(prompt_text)} chars)")

    def optimize(self, persona_dir: Path = PERSONA_PIPELINE_DIR) -> WargameResponder:
        """
        Optimize the wargame responder using DSPy.

        Returns the optimized module.
        """
        if not self.responder:
            self.setup_dspy()

        print("\n" + "=" * 70)
        print(f"OPTIMIZING WARGAME RESPONDER ({self.config.optimizer.upper()})")
        print("=" * 70)

        # Create dataset
        trainset, valset = create_wargame_dataset(persona_dir)
        print(f"Dataset: {len(trainset)} train, {len(valset)} val")

        # Create metric
        metric = create_wargame_metric(self.judge)

        if self.config.optimizer == "gepa":
            optimizer = dspy.GEPA(
                metric=metric,
                auto=self.config.budget,
                reflection_lm=self._reflection_lm,
                reflection_minibatch_size=2,
                candidate_selection_strategy="pareto",
                use_merge=True,
                track_stats=True,
                num_threads=self.config.num_threads,
            )
            print("\nGEPA: Reflective prompt evolution with Pareto selection")

        elif self.config.optimizer == "mipro":
            def score_metric(gold, pred, trace=None):
                result = metric(gold, pred, trace)
                return result["score"] if isinstance(result, dict) else result

            optimizer = dspy.MIPROv2(
                metric=score_metric,
                auto=self.config.budget,
                num_threads=self.config.num_threads,
                track_stats=True,
            )
            print("\nMIPROv2: Bayesian instruction + few-shot optimization")

        else:
            print("\nNo optimization (baseline)")
            return self.responder

        print("\nStarting optimization...")
        optimized = optimizer.compile(
            student=self.responder,
            trainset=trainset,
            valset=valset,
        )

        self.responder = optimized
        print("Optimization complete!")

        # Show evolved instruction
        evolved_instructions = {}
        for name, predictor in optimized.named_predictors():
            if hasattr(predictor, "signature") and predictor.signature.__doc__:
                evolved_instructions[name] = predictor.signature.__doc__
                print(f"\nEvolved instruction for {name}:")
                print(f"  {predictor.signature.__doc__[:300]}...")

        # Save results
        self._save_optimization_results(optimized, evolved_instructions)

        return optimized

    def _save_optimization_results(self, optimized, evolved_instructions: dict):
        """Save optimization results to disk."""
        from datetime import datetime

        results_dir = Path(__file__).parent.parent.parent / "results"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = results_dir / f"gepa_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save evolved prompt
        prompt_file = run_dir / "optimized_prompt.txt"
        with open(prompt_file, "w") as f:
            for name, instruction in evolved_instructions.items():
                f.write(f"=== {name} ===\n\n{instruction}\n\n")
        print(f"\nSaved optimized prompt to: {prompt_file}")

        # Save config as JSON
        config_file = run_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump({
                "main_model": self.config.main_model,
                "reflection_model": self.config.reflection_model,
                "judge_model": self.config.judge_model,
                "optimizer": self.config.optimizer,
                "budget": self.config.budget,
                "personas": list(self.personas.keys()),
                "timestamp": timestamp,
            }, f, indent=2)
        print(f"Saved config to: {config_file}")

        # Save DSPy program
        try:
            optimized.save(str(run_dir / "optimized_program.json"))
            print(f"Saved DSPy program to: {run_dir / 'optimized_program.json'}")
        except Exception as e:
            print(f"Could not save DSPy program: {e}")

    def run_round(
        self,
        topic: str,
        history: list[dict],
        round_number: int,
    ) -> list[dict]:
        """Run one round of the wargame."""
        if not self.responder:
            self.setup_dspy()

        responses = []

        # Build history string
        history_str = "\n\n".join([
            f"**{h['name']}** ({h['stance']}): {h['response']}"
            for h in history[-6:]  # Last 6 messages
        ]) if history else "This is the opening round."

        for pid, persona in self.personas.items():
            # Get assigned model for this persona
            model_id = self.model_assignments.get(pid)
            if model_id:
                # Create LM for this persona's model
                persona_lm = dspy.LM(
                    model=f"openrouter/{model_id}",
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    api_base="https://openrouter.ai/api/v1",
                    max_tokens=1500,
                )
            else:
                persona_lm = self._main_lm

            # Generate response with persona's model
            persona_context = persona.build_system_prompt(topic)

            with dspy.context(lm=persona_lm):
                try:
                    pred = self.responder(
                        persona_context=persona_context,
                        topic=topic,
                        discussion_history=history_str,
                        round_number=round_number,
                    )
                    response_text = pred.public_response
                except Exception as e:
                    response_text = f"[Error generating response: {e}]"

            responses.append({
                "persona_id": pid,
                "name": persona.name,
                "stance": persona.stance,
                "model": model_id or "default",
                "response": response_text,
                "round": round_number,
            })

        return responses

    def run_wargame(
        self,
        topic: str,
        persona_ids: list[str],
        max_rounds: int = 5,
        optimize_first: bool = False,
        on_message: Callable[[dict], None] = None,
    ) -> dict:
        """
        Run a complete wargame simulation.

        Args:
            topic: The policy topic to debate
            persona_ids: List of persona IDs to include
            max_rounds: Number of rounds
            optimize_first: Whether to run DSPy optimization first
            on_message: Callback for each message

        Returns:
            Summary of the wargame
        """
        self.setup_dspy()
        self.load_personas(persona_ids)

        if optimize_first:
            self.optimize()

        print(f"\n{'=' * 70}")
        print("WARGAME: " + topic[:60])
        print("=" * 70)

        history = []

        for round_num in range(1, max_rounds + 1):
            print(f"\n--- Round {round_num} ---")

            responses = self.run_round(topic, history, round_num)

            for resp in responses:
                history.append(resp)

                if on_message:
                    on_message(resp)

                print(f"\n[{resp['name']}] ({resp['stance']}, {resp['model']})")
                print(resp['response'])  # Full response, not truncated

        # Summary
        return {
            "topic": topic,
            "rounds": max_rounds,
            "participants": [
                {"id": pid, "name": p.name, "stance": p.stance, "model": self.model_assignments.get(pid)}
                for pid, p in self.personas.items()
            ],
            "total_messages": len(history),
            "transcript": history,
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run optimized wargame from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimized AI Policy Wargame")
    parser.add_argument("--topic", "-t", default="Should frontier AI development be paused until safety is better understood?")
    parser.add_argument("--personas", "-p", nargs="+", default=["dario_amodei", "sam_altman", "jensen_huang", "xi_jinping"])
    parser.add_argument("--rounds", "-r", type=int, default=3)
    parser.add_argument("--optimize", "-o", action="store_true", help="Run DSPy optimization first")
    parser.add_argument("--optimizer", choices=["gepa", "mipro", "none"], default="none")
    parser.add_argument("--budget", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--prompt", help="Path to optimized prompt file to use")
    args = parser.parse_args()

    config = OptimizedWargameConfig(
        optimizer=args.optimizer if args.optimize else "none",
        budget=args.budget,
    )

    engine = OptimizedWargameEngine(config)

    # Load optimized prompt if provided
    if args.prompt:
        prompt_path = Path(args.prompt)
        if prompt_path.exists():
            with open(prompt_path) as f:
                optimized_prompt = f.read()
            engine.set_optimized_prompt(optimized_prompt)
            print(f"Loaded optimized prompt from: {prompt_path}")
        else:
            print(f"Warning: Prompt file not found: {prompt_path}")

    result = engine.run_wargame(
        topic=args.topic,
        persona_ids=args.personas,
        max_rounds=args.rounds,
        optimize_first=args.optimize,
    )

    print("\n" + "=" * 70)
    print("WARGAME COMPLETE")
    print("=" * 70)
    print(f"Topic: {result['topic']}")
    print(f"Rounds: {result['rounds']}")
    print(f"Messages: {result['total_messages']}")

    # Save transcript
    from datetime import datetime
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    transcript_file = results_dir / f"wargame_{timestamp}.json"
    with open(transcript_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nTranscript saved to: {transcript_file}")


if __name__ == "__main__":
    main()
