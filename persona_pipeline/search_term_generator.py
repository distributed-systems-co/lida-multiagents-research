#!/usr/bin/env python3
"""
DSPy-powered search term generation for deep persona research.

Generates diverse, targeted search queries to capture all facets of a person:
- Professional life
- Personal life
- Relationships & drama
- Controversies
- Internet presence
- How they actually are
"""

import dspy
from typing import Optional
import os


# Configure DSPy with a cheap/fast model
def configure_dspy(model: str = "claude-3-5-haiku-20241022"):
    """Configure DSPy with Anthropic."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    lm = dspy.LM(
        model=f"anthropic/{model}",
        api_key=api_key,
        max_tokens=4000,
    )
    dspy.configure(lm=lm)
    return lm


class PersonaSearchTerms(dspy.Signature):
    """Generate comprehensive search terms to research a person for deep persona building."""

    name: str = dspy.InputField(desc="Full name of the person")
    role: str = dspy.InputField(desc="Their known role/influence")
    category: str = dspy.InputField(desc="Category they belong to")

    # Core identity
    biography_terms: list[str] = dspy.OutputField(desc="3-5 terms for biography, education, career history")
    current_role_terms: list[str] = dspy.OutputField(desc="3-5 terms for current position, recent activities")

    # Personality & style
    personality_terms: list[str] = dspy.OutputField(desc="3-5 terms to find personality traits, quirks, management style")
    communication_terms: list[str] = dspy.OutputField(desc="3-5 terms to find how they talk, interviews, quotes")

    # Relationships
    relationship_terms: list[str] = dspy.OutputField(desc="3-5 terms to find friends, allies, mentors")
    conflict_terms: list[str] = dspy.OutputField(desc="3-5 terms to find rivals, enemies, feuds, beefs")

    # Drama & controversies
    controversy_terms: list[str] = dspy.OutputField(desc="3-5 terms for scandals, criticism, controversies")
    gossip_terms: list[str] = dspy.OutputField(desc="3-5 terms for rumors, gossip, open secrets")

    # Internet presence
    twitter_terms: list[str] = dspy.OutputField(desc="3-5 terms for Twitter/X presence, social media drama")
    reddit_hn_terms: list[str] = dspy.OutputField(desc="3-5 terms to find Reddit/HN discussions about them")
    meme_terms: list[str] = dspy.OutputField(desc="3-5 terms for memes, jokes, viral moments about them")

    # Policy & beliefs (for AI/chip people)
    policy_terms: list[str] = dspy.OutputField(desc="3-5 terms for their policy positions, AI views, China stance")

    # Deep human stuff
    personal_life_terms: list[str] = dspy.OutputField(desc="3-5 terms for family, personal life, hobbies")
    psychology_terms: list[str] = dspy.OutputField(desc="3-5 terms to understand their psychology, motivations, insecurities")


class AdditionalSearchTerms(dspy.Signature):
    """Generate additional targeted search terms based on initial findings."""

    name: str = dspy.InputField(desc="Person's name")
    initial_findings: str = dspy.InputField(desc="Summary of what we found so far")
    gaps: str = dspy.InputField(desc="What we're still missing")

    followup_terms: list[str] = dspy.OutputField(desc="10-15 additional targeted search terms to fill the gaps")


class QuoteExtractor(dspy.Signature):
    """Extract the most revealing quotes that capture how this person talks."""

    name: str = dspy.InputField()
    text: str = dspy.InputField(desc="Source text containing quotes")

    quotes: list[str] = dspy.OutputField(desc="5-10 most revealing/characteristic quotes, with context")


class SearchTermGenerator:
    """Generate comprehensive search terms for persona research."""

    def __init__(self):
        self.term_generator = dspy.Predict(PersonaSearchTerms)
        self.followup_generator = dspy.Predict(AdditionalSearchTerms)

    def generate_initial_terms(self, name: str, role: str, category: str) -> list[str]:
        """Generate initial comprehensive search terms."""
        result = self.term_generator(name=name, role=role, category=category)

        all_terms = []

        # Collect all term lists
        for field in [
            'biography_terms', 'current_role_terms', 'personality_terms',
            'communication_terms', 'relationship_terms', 'conflict_terms',
            'controversy_terms', 'gossip_terms', 'twitter_terms',
            'reddit_hn_terms', 'meme_terms', 'policy_terms',
            'personal_life_terms', 'psychology_terms'
        ]:
            terms = getattr(result, field, [])
            if terms:
                all_terms.extend(terms)

        # Dedupe while preserving order
        seen = set()
        unique_terms = []
        for term in all_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        return unique_terms

    def generate_followup_terms(self, name: str, findings: str, gaps: str) -> list[str]:
        """Generate additional terms based on what we've found."""
        result = self.followup_generator(
            name=name,
            initial_findings=findings,
            gaps=gaps
        )
        return result.followup_terms or []

    def generate_all_terms(self, name: str, role: str, category: str) -> dict:
        """Generate all search terms organized by category."""
        result = self.term_generator(name=name, role=role, category=category)

        return {
            "biography": result.biography_terms or [],
            "current_role": result.current_role_terms or [],
            "personality": result.personality_terms or [],
            "communication": result.communication_terms or [],
            "relationships": result.relationship_terms or [],
            "conflicts": result.conflict_terms or [],
            "controversies": result.controversy_terms or [],
            "gossip": result.gossip_terms or [],
            "twitter": result.twitter_terms or [],
            "reddit_hn": result.reddit_hn_terms or [],
            "memes": result.meme_terms or [],
            "policy": result.policy_terms or [],
            "personal_life": result.personal_life_terms or [],
            "psychology": result.psychology_terms or [],
        }


# Fallback templates when DSPy isn't available
FALLBACK_QUERY_TEMPLATES = {
    "biography": [
        "{name} biography background",
        "{name} education university degree",
        "{name} career history early career",
        "{name} origin story how started",
    ],
    "current": [
        "{name} 2024 2025 recent",
        "{name} current role position",
        "{name} latest news today",
        "{name} recent interview",
    ],
    "personality": [
        "{name} personality traits",
        "{name} management style leadership",
        "{name} interview personality",
        "{name} what is like person",
        "{name} quirks habits",
    ],
    "communication": [
        "{name} quotes statements",
        "{name} interview transcript",
        "{name} speech keynote",
        "{name} podcast appearance",
        '"{name}" said stated',
    ],
    "relationships": [
        "{name} friends allies relationships",
        "{name} mentor mentored by",
        "{name} inner circle advisors",
        "{name} married wife husband family",
    ],
    "conflicts": [
        "{name} feud beef conflict",
        "{name} rivalry competitor vs",
        "{name} criticized attacked by",
        "{name} enemies opponents",
        "{name} falling out former",
    ],
    "controversies": [
        "{name} controversy scandal",
        "{name} criticized backlash",
        "{name} accused allegations",
        "{name} apology apologized mistake",
        "{name} fired resigned controversy",
    ],
    "gossip": [
        "{name} rumors gossip",
        "{name} secrets revealed",
        '"{name}" actually really',
        "{name} behind the scenes",
    ],
    "twitter": [
        "{name} twitter drama",
        "{name} tweet controversial",
        "{name} social media fight",
        '"{name}" ratio twitter',
        "site:twitter.com {name}",
    ],
    "reddit": [
        'site:reddit.com "{name}"',
        '"{name}" reddit opinion',
        '"{name}" hacker news hn',
        "{name} overrated underrated reddit",
    ],
    "memes": [
        "{name} meme memes",
        "{name} jokes funny",
        "{name} parody satire",
        "{name} viral moment",
    ],
    "policy": [
        "{name} AI artificial intelligence views",
        "{name} China policy position",
        "{name} regulation stance",
        "{name} chip semiconductor export",
        "{name} {role}",
    ],
    "personal": [
        "{name} personal life",
        "{name} hobbies interests",
        "{name} net worth wealth",
        "{name} daily routine lifestyle",
    ],
    "psychology": [
        "{name} motivations driven by",
        "{name} insecure about weakness",
        "{name} ego arrogant humble",
        "{name} what wants goals",
    ],
}


def generate_fallback_terms(name: str, role: str) -> list[str]:
    """Generate search terms using templates (no LLM needed)."""
    all_terms = []

    for category, templates in FALLBACK_QUERY_TEMPLATES.items():
        for template in templates:
            term = template.format(name=name, role=role)
            all_terms.append(term)

    return all_terms


def main():
    """Test the search term generator."""
    import sys

    # Try to configure DSPy
    try:
        configure_dspy()
        use_dspy = True
        print("Using DSPy with Claude Haiku")
    except Exception as e:
        print(f"DSPy not available ({e}), using fallback templates")
        use_dspy = False

    # Test person
    name = sys.argv[1] if len(sys.argv) > 1 else "Jensen Huang"
    role = sys.argv[2] if len(sys.argv) > 2 else "GPU choke point"
    category = sys.argv[3] if len(sys.argv) > 3 else "Frontier AI & GPU Controllers"

    print(f"\nGenerating search terms for: {name}")
    print(f"Role: {role}")
    print(f"Category: {category}")
    print("=" * 60)

    if use_dspy:
        generator = SearchTermGenerator()
        terms_by_category = generator.generate_all_terms(name, role, category)

        total = 0
        for cat, terms in terms_by_category.items():
            print(f"\n{cat.upper()}:")
            for term in terms:
                print(f"  • {term}")
            total += len(terms)

        print(f"\n{'=' * 60}")
        print(f"Total: {total} search terms")
    else:
        terms = generate_fallback_terms(name, role)
        print(f"\nGenerated {len(terms)} search terms:")
        for term in terms:
            print(f"  • {term}")


if __name__ == "__main__":
    main()
