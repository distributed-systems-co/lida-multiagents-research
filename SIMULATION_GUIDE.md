# Simulation Guide: Loading Templates & Running Simulations

This guide covers loading YAML templates and running multi-agent simulations in the LIDA system.

## Quick Start

```bash
# Run interactive menu
python run_ai_safety_debate.py

# Run specific matchup with auto-mode
python run_ai_safety_debate.py --matchup doom_vs_accel --auto

# Run specific topic
python run_ai_safety_debate.py --topic ai_pause --rounds 5
```

## Directory Structure

```
scenarios/
├── ai_xrisk.yaml                    # Main AI X-Risk scenario
├── default.yaml                     # Default config
├── quick.yaml                       # Quick testing
├── personas/
│   ├── v1/base.yaml                 # Base persona templates
│   └── v2/
│       ├── ai_researchers.yaml      # Yudkowsky, Altman, LeCun, etc.
│       ├── tech_leaders.yaml        # CEOs and VCs
│       └── psychology.yaml          # Psychological profiles
├── tactics/
│   └── v2/advanced.yaml             # Persuasion strategies
├── topics/
│   └── v1/ai_governance.yaml        # Debate topics
├── models/
│   └── v1/configurations.yaml       # LLM configs
├── relationships/
│   └── v1/ai_community.yaml         # Influence dynamics
├── campaigns/
│   ├── ai_safety_debate.yaml        # Tournament campaign
│   └── sophisticated_debate.yaml
└── presets/
    ├── default.yaml                 # Default preset with imports
    ├── budget.yaml                  # Budget-friendly models
    └── ai_xrisk.yaml                # X-Risk preset
```

## Loading Templates

### Basic YAML Loading

```python
from src.config.loader import load_yaml_file, load_scenario

# Load a raw YAML file
config = load_yaml_file("scenarios/ai_xrisk.yaml")

# Load a scenario with imports resolved
scenario = load_scenario("ai_xrisk")  # looks in presets/, campaigns/, then root

# Load by environment variable
import os
os.environ["SCENARIO"] = "budget"
config = load_scenario()  # loads budget scenario
```

### Loading Components

```python
from src.config.loader import (
    load_component,
    get_all_personas,
    get_persona,
    get_tactic,
    get_topic,
    list_scenarios,
    list_components,
)

# Load specific component types
personas = load_component("personas", "ai_researchers", version="v2")
tactics = load_component("tactics", "advanced", version="v2")
topics = load_component("topics", "ai_governance", version="v1")

# List available options
print(list_scenarios())        # All available scenarios
print(list_components("personas", "v2"))  # v2 persona files

# Get specific persona
yudkowsky = get_persona("yudkowsky", scenario="ai_xrisk")
print(yudkowsky["name"])       # "Eliezer Yudkowsky"
print(yudkowsky["resistance_score"])  # 0.95

# Get all personas from scenario
all_personas = get_all_personas("ai_xrisk")
for name, config in all_personas.items():
    print(f"{name}: {config.get('initial_position', 'UNDECIDED')}")
```

### Using Config Helpers

```python
from src.config.loader import (
    get_config,
    get_simulation_config,
    get_models_config,
    build_roster,
    build_persona_state,
)

# Dot-notation config access
max_rounds = get_config("simulation.max_rounds_per_agent", default=5)
default_model = get_config("models.default")

# Get full config sections
sim_config = get_simulation_config("ai_xrisk")
print(sim_config["num_agents"])  # 10

# Build active roster with overrides applied
roster = build_roster("ai_xrisk")
for persona_id, persona in roster.items():
    print(f"{persona_id}: {persona['model']}")

# Build fully hydrated persona with psychology
yudkowsky_state = build_persona_state("yudkowsky", "ai_xrisk")
print(yudkowsky_state["_computed"]["persuadability"])  # 0.05
```

## Scenario YAML Format

### Minimal Scenario

```yaml
# scenarios/my_scenario.yaml
simulation:
  num_agents: 4
  max_rounds_per_agent: 3

personas:
  - id: "agent_1"
    name: "Agent One"
    model: "anthropic/claude-sonnet-4"
    initial_position: "FOR"
    confidence: 0.8
```

### Full Scenario with Imports

```yaml
# scenarios/presets/my_preset.yaml
_version: "1.0"
_description: "My custom preset"

# Import components from other files
imports:
  personas: "../personas/v2/ai_researchers.yaml"
  tactics: "../tactics/v2/advanced.yaml"
  topics: "../topics/v1/ai_governance.yaml"

# Select which personas to activate
active_personas:
  - yudkowsky
  - altman
  - lecun

# Simulation settings
simulation:
  num_agents: 3
  max_rounds_per_agent: 5
  enable_relationship_dynamics: true
  enable_cognitive_biases: true

# Persuader configuration
persuader:
  model: "anthropic/claude-opus-4"
  target_position: "FOR"
  adaptation_enabled: true

default_topic: "ai_pause"
```

### Persona Definition

```yaml
personas:
  - id: "my_persona"
    name: "My Custom Persona"
    role: "AI Researcher"
    organization: "Research Lab"
    model: "anthropic/claude-sonnet-4"
    personality: "analytical"  # analytical, empathetic, assertive, creative, pragmatic, skeptical
    initial_position: "AGAINST"  # FOR, AGAINST, UNDECIDED
    confidence: 0.85  # 0.0 - 1.0

    background: |
      Multi-line background describing the persona's history and perspective.

    system_prompt: |
      You are reasoning as this persona. Your core beliefs are...

    signature_arguments:
      - "First key argument this persona makes"
      - "Second key argument"

    debate_tactics:
      - "thought_experiments"
      - "evidence_based"

    cognitive_biases:
      pessimism_bias: 0.5
      confirmation_bias: 0.3

    vulnerabilities:
      - "Arguments that might persuade this persona"

    resistances:
      - "Arguments this persona rejects"

    hot_buttons:
      - "Topics that trigger strong reactions"

    resistance_score: 0.8  # 0-1: how hard to persuade
    sycophancy_baseline: 0.1  # 0-1: susceptibility to flattery
```

## Running Simulations

### Using run_ai_safety_debate.py

```bash
# Interactive menu
python run_ai_safety_debate.py

# Specific topic with recommended participants
python run_ai_safety_debate.py --topic ai_pause

# Custom participants
python run_ai_safety_debate.py --topic ai_pause --participants yudkowsky,lecun,altman

# Predefined matchup
python run_ai_safety_debate.py --matchup doom_vs_accel

# Auto-run (no prompts)
python run_ai_safety_debate.py --matchup doom_vs_accel --auto --rounds 6

# Use specific model
python run_ai_safety_debate.py --topic ai_pause --model google/gemini-2.0-flash-001

# Use different scenario file
python run_ai_safety_debate.py --scenario scenarios/budget.yaml

# List all options
python run_ai_safety_debate.py --list-all
```

### Available Topics

| Topic ID | Description |
|----------|-------------|
| `ai_pause` | 6-Month AI Training Pause |
| `lab_self_regulation` | Can labs self-regulate? |
| `xrisk_vs_present_harms` | X-Risk vs Present Harms |
| `scaling_hypothesis` | Will scaling lead to AGI? |
| `open_source_ai` | Should frontier models be open? |
| `government_regulation` | Should governments mandate safety? |

### Predefined Matchups

| Matchup ID | Teams |
|------------|-------|
| `doom_vs_accel` | Yudkowsky, Connor vs Andreessen, LeCun |
| `labs_debate` | Altman, Amodei vs LeCun, Andreessen |
| `academics_clash` | Bengio, Russell vs LeCun |
| `ethics_vs_scale` | Gebru, Toner vs Altman, Andreessen |
| `full_panel` | 4v4 with all major voices |

### Programmatic Usage

```python
import asyncio
from run_ai_safety_debate import AIDebateRunner

async def main():
    # Initialize runner with scenario
    runner = AIDebateRunner("scenarios/ai_xrisk.yaml")

    # List available options
    runner.list_personas()
    runner.list_topics()
    runner.list_matchups()

    # Run a debate
    engine = await runner.run_debate(
        topic_id="ai_pause",
        participants=["yudkowsky", "lecun", "altman"],
        rounds=5,
        auto=True,
        use_llm=True,
        llm_provider="openrouter",
        llm_model="anthropic/claude-sonnet-4",
    )

    # Access results
    print(engine.summarize())

    # Or run a matchup
    engine = await runner.run_matchup(
        matchup_id="doom_vs_accel",
        topic_id="ai_pause",  # override default topic
        rounds=4,
        auto=True,
    )

asyncio.run(main())
```

### Using the Debate Engine Directly

```python
import asyncio
from src.simulation.advanced_debate_engine import AdvancedDebateEngine, DebateCLI

async def run_custom_debate():
    # Create engine
    engine = AdvancedDebateEngine(
        topic="AI Governance",
        motion="Should we pause AI development?",
        participants=["yudkowsky", "lecun", "altman"],
        llm_provider="openrouter",
        llm_model="anthropic/claude-sonnet-4",
        use_llm=True,
    )

    # Run rounds manually
    for round_num in range(5):
        await engine.run_round()
        print(f"Round {round_num + 1} complete")

    # Get summary
    print(engine.summarize())

    # Or use CLI for interactive mode
    cli = DebateCLI(engine)
    await cli.run(max_rounds=5)

asyncio.run(run_custom_debate())
```

## Environment Variables

```bash
# API Keys
export OPENROUTER_API_KEY=sk-or-...
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Scenario selection
export SCENARIO=ai_xrisk

# Override settings
export NUM_AGENTS=8
export MAX_ROUNDS=10
export DEFAULT_MODEL=anthropic/claude-opus-4
```

## LLM Providers & Models

### OpenRouter (Default)

```bash
export OPENROUTER_API_KEY=sk-or-...
python run_ai_safety_debate.py --provider openrouter --model anthropic/claude-sonnet-4
```

Available models via OpenRouter:
- `anthropic/claude-opus-4` - Most capable
- `anthropic/claude-sonnet-4` - Balanced (default)
- `openai/gpt-4o` - OpenAI flagship
- `google/gemini-2.0-flash-001` - Fast, cheap
- `meta-llama/llama-3.3-70b-instruct` - Open weights
- `deepseek/deepseek-chat-v3-0324` - Budget option
- `x-ai/grok-2` - X.AI model

### Direct Provider Access

```bash
# Anthropic direct
export ANTHROPIC_API_KEY=sk-ant-...
python run_ai_safety_debate.py --provider anthropic

# OpenAI direct
export OPENAI_API_KEY=sk-...
python run_ai_safety_debate.py --provider openai
```

## Creating Custom Scenarios

### Step 1: Create Persona File

```yaml
# scenarios/personas/v2/my_personas.yaml
_version: "2.0"

personas:
  optimist:
    id: "optimist"
    name: "Tech Optimist"
    role: "Futurist"
    organization: "Progress Institute"
    model: "anthropic/claude-sonnet-4"
    personality: "assertive"
    initial_position: "FOR"
    confidence: 0.9

    system_prompt: |
      You believe technology always makes things better...

    signature_arguments:
      - "Every technology was feared before it transformed society"

    resistance_score: 0.3
    sycophancy_baseline: 0.2

  pessimist:
    id: "pessimist"
    name: "Tech Skeptic"
    # ... similar structure
```

### Step 2: Create Preset

```yaml
# scenarios/presets/my_debate.yaml
_version: "1.0"
_extends: "default.yaml"  # Inherit from default

imports:
  personas:
    source: "../personas/v2/my_personas.yaml"

active_personas:
  - optimist
  - pessimist

simulation:
  max_rounds_per_agent: 6

default_topic: "scaling_hypothesis"
```

### Step 3: Run It

```bash
python run_ai_safety_debate.py --scenario scenarios/presets/my_debate.yaml
```

## Tracking Results

### Metrics Available

```python
# After running a debate
engine = await runner.run_debate(...)

# Access debate state
state = engine.state
for debater_id, debater in state.debaters.items():
    print(f"{debater_id}:")
    print(f"  Position: {debater.beliefs.get('support_motion', 0.5)}")
    print(f"  Confidence: {debater.beliefs.get('confidence', 0.5)}")
    print(f"  Emotional State: {debater.emotional_state}")

# Get transcript
for entry in engine.transcript:
    print(f"{entry['speaker']}: {entry['content'][:100]}...")
```

### Experiment Logging

Results are automatically logged to `experiment_results/` and `logs/` directories when running simulations.

## Tips

1. **Start with presets**: Use `default.yaml` or `ai_xrisk.yaml` as templates
2. **Test with quick.yaml**: 2-round testing scenario for iteration
3. **Use budget models**: `gemini-2.0-flash` or `deepseek-chat` for cost-effective testing
4. **Check relationships**: Persona relationships affect persuasion effectiveness
5. **Monitor resistance_score**: Higher = harder to persuade (0-1 scale)
6. **Use --auto for batch runs**: Skips interactive prompts
