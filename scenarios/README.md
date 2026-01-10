# LIDA Scenario System

Programmable, composable scenario configuration for multi-agent persuasion research.

## Directory Structure

```
scenarios/
├── campaigns/           # Full experiment definitions
│   ├── ai_safety_debate.yaml
│   └── vaccine_persuasion.yaml
│
├── personas/            # Character definitions
│   ├── v1/             # Version 1 personas
│   │   └── ai_researchers.yaml
│   └── v2/             # Version 2 (enhanced)
│       └── ai_researchers.yaml
│
├── tactics/            # Persuasion strategies
│   ├── v1/
│   │   └── cialdini.yaml
│   └── v2/
│       └── advanced.yaml
│
├── topics/             # Debate propositions
│   ├── v1/
│   │   └── ai_governance.yaml
│   └── v2/
│
├── prompts/            # System prompt templates
│   ├── v1/
│   │   ├── base_agent.yaml
│   │   └── persuader.yaml
│   └── v2/
│
├── models/             # LLM configurations
│   └── v1/
│       ├── budget.yaml
│       └── premium.yaml
│
├── relationships/      # Inter-persona dynamics
│   └── v1/
│       └── ai_community.yaml
│
└── presets/            # Quick-start configs
    ├── default.yaml
    ├── budget.yaml
    └── ai_xrisk.yaml
```

## Usage

### Run a campaign
```bash
CAMPAIGN=ai_safety_debate docker compose up app
```

### Compose custom scenario
```yaml
# my_scenario.yaml
extends: presets/default.yaml

imports:
  personas: personas/v2/ai_researchers.yaml
  tactics: tactics/v2/advanced.yaml
  topics: topics/v1/ai_governance.yaml

overrides:
  simulation:
    max_rounds: 10

  # Cherry-pick specific personas
  active_personas:
    - yudkowsky
    - altman
    - gebru
```

### Play as character
```bash
# Interactive mode - you play as a persona
PLAY_AS=yudkowsky CAMPAIGN=ai_safety docker compose up app
```

## Composability

### Persona Inheritance
```yaml
# personas/v2/custom.yaml
extends: personas/v1/base.yaml

personas:
  my_researcher:
    extends: scientist_base  # Inherit from base
    name: "Dr. Custom"
    overrides:
      resistance_score: 0.8
      model: "anthropic/claude-opus-4"
```

### Prompt Templates
```yaml
# prompts/v2/agent.yaml
templates:
  base_reasoning: |
    You are {{name}}, {{role}} at {{organization}}.
    {{background}}

    Core beliefs:
    {{#each beliefs}}
    - {{this}}
    {{/each}}

  debate_style: |
    Debate tactics you prefer: {{tactics}}
    Arguments you find compelling: {{vulnerabilities}}
    Arguments you resist: {{resistances}}
```

### Round-based Character Switching
```yaml
# campaigns/rotating_debate.yaml
rounds:
  - round: 1
    player_as: yudkowsky
    opponents: [altman, andreessen]
    topic: topics/ai_pause

  - round: 2
    player_as: altman
    opponents: [yudkowsky, gebru]
    topic: topics/lab_governance
```
