#!/usr/bin/env python3
"""
DEEP SEARCH TERM GENERATOR

Generates comprehensive search terms to find EVERYTHING about a person:
- Criminal/legal troubles
- Controversies and scandals
- Relationships and social dynamics
- Pressure points and vulnerabilities
- Financial entanglements
- What they're hiding
"""

import os
from typing import Optional

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


def configure_dspy(model: str = "claude-3-5-haiku-20241022"):
    """Configure DSPy with cheap model."""
    if not DSPY_AVAILABLE:
        raise ImportError("dspy not installed")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    lm = dspy.LM(f"anthropic/{model}", api_key=api_key, max_tokens=4000)
    dspy.configure(lm=lm)
    return lm


if DSPY_AVAILABLE:
    class DeepPersonaTerms(dspy.Signature):
        """Generate exhaustive search terms to research a person for deep profiling including controversies, legal issues, and pressure points."""

        name: str = dspy.InputField(desc="Full name")
        role: str = dspy.InputField(desc="Their role/position")
        category: str = dspy.InputField(desc="Category")

        # Basic
        biography_terms: list[str] = dspy.OutputField(desc="5 terms for bio, education, career")
        current_terms: list[str] = dspy.OutputField(desc="5 terms for current role, 2024-2025 news")

        # Personality
        personality_terms: list[str] = dspy.OutputField(desc="5 terms for personality, quirks, management style")
        communication_terms: list[str] = dspy.OutputField(desc="5 terms for how they talk, quotes, interviews")

        # Relationships
        allies_terms: list[str] = dspy.OutputField(desc="5 terms for friends, allies, inner circle, mentors")
        enemies_terms: list[str] = dspy.OutputField(desc="5 terms for rivals, enemies, people who hate them")
        feud_terms: list[str] = dspy.OutputField(desc="5 terms for feuds, beefs, fallings out, betrayals")
        family_terms: list[str] = dspy.OutputField(desc="5 terms for family, spouse, romantic relationships")

        # Legal / Criminal
        legal_terms: list[str] = dspy.OutputField(desc="5 terms for lawsuits, legal troubles, court cases")
        criminal_terms: list[str] = dspy.OutputField(desc="5 terms for criminal investigations, indictments, arrests, fraud")
        regulatory_terms: list[str] = dspy.OutputField(desc="5 terms for SEC, FTC, DOJ investigations, subpoenas")

        # Controversies
        scandal_terms: list[str] = dspy.OutputField(desc="5 terms for scandals, PR disasters, public failures")
        criticism_terms: list[str] = dspy.OutputField(desc="5 terms for criticism, backlash, accusations")
        misconduct_terms: list[str] = dspy.OutputField(desc="5 terms for misconduct, harassment, discrimination allegations")

        # Financial
        financial_terms: list[str] = dspy.OutputField(desc="5 terms for financial troubles, debts, conflicts of interest")
        compensation_terms: list[str] = dspy.OutputField(desc="5 terms for salary, net worth, stock sales, golden parachutes")

        # Pressure Points
        vulnerability_terms: list[str] = dspy.OutputField(desc="5 terms for weaknesses, vulnerabilities, insecurities")
        leverage_terms: list[str] = dspy.OutputField(desc="5 terms for leverage others have, what they owe, political debts")
        fear_terms: list[str] = dspy.OutputField(desc="5 terms for what they fear, avoid, are defensive about")

        # Internet
        twitter_terms: list[str] = dspy.OutputField(desc="5 terms for twitter drama, social media controversies")
        reddit_terms: list[str] = dspy.OutputField(desc="5 terms for reddit discussions, anonymous criticism")
        leaked_terms: list[str] = dspy.OutputField(desc="5 terms for leaks, whistleblowers, internal documents")

        # Policy
        policy_terms: list[str] = dspy.OutputField(desc="5 terms for AI/chip policy positions, lobbying")
        hypocrisy_terms: list[str] = dspy.OutputField(desc="5 terms for hypocrisy, contradictions, flip-flops")


    class PressurePointAnalysis(dspy.Signature):
        """Analyze a person's pressure points and vulnerabilities based on research."""

        name: str = dspy.InputField()
        research: str = dspy.InputField(desc="Research findings about the person")

        career_vulnerabilities: list[str] = dspy.OutputField(desc="Ways their career could be damaged")
        reputation_vulnerabilities: list[str] = dspy.OutputField(desc="Reputation risks and sensitivities")
        legal_exposure: list[str] = dspy.OutputField(desc="Legal risks and exposure")
        financial_pressure: list[str] = dspy.OutputField(desc="Financial vulnerabilities")
        relationship_leverage: list[str] = dspy.OutputField(desc="Relationships that could be leveraged")
        psychological_triggers: list[str] = dspy.OutputField(desc="Emotional triggers and insecurities")
        public_opinion_risks: list[str] = dspy.OutputField(desc="Things that would turn public against them")
        internal_threats: list[str] = dspy.OutputField(desc="Threats from within their organization")
        historical_skeletons: list[str] = dspy.OutputField(desc="Past issues that could resurface")


# Comprehensive fallback templates
DEEP_QUERY_TEMPLATES = {
    # === BASIC ===
    "biography": [
        "{name} biography background early life",
        "{name} education university degree",
        "{name} career history timeline",
        "{name} origin story how started",
        "{name} before famous early career",
    ],
    "current": [
        "{name} 2024 2025 latest news",
        "{name} current role position today",
        "{name} recent interview statements",
        "{name} latest announcement decision",
        "{name} this week today news",
    ],

    # === PERSONALITY ===
    "personality": [
        "{name} personality traits character",
        "{name} management style leadership",
        "{name} temper angry outburst",
        "{name} ego arrogant narcissist",
        "{name} quirks weird habits",
    ],
    "communication": [
        "{name} quotes famous statements",
        "{name} interview podcast full",
        "{name} speech keynote transcript",
        "{name} email leaked internal",
        "{name} how talks speaking style",
    ],

    # === RELATIONSHIPS ===
    "allies": [
        "{name} friends allies close to",
        "{name} inner circle advisors trusts",
        "{name} mentor influenced by",
        "{name} protege mentored helped",
        "{name} business partners investors",
    ],
    "enemies": [
        "{name} enemies rivals hates",
        "{name} critics opponents against",
        "{name} conflict with dispute",
        "{name} who hates dislikes",
        "{name} bad blood animosity",
    ],
    "feuds": [
        "{name} feud fight beef",
        "{name} falling out former friend",
        "{name} betrayed by betrayal",
        "{name} fired sued by employee",
        "{name} public fight argument",
    ],
    "family": [
        "{name} married wife husband spouse",
        "{name} family children kids",
        "{name} divorce affair cheating",
        "{name} parents siblings family background",
        "{name} relationship dating",
    ],

    # === LEGAL / CRIMINAL ===
    "legal": [
        "{name} lawsuit sued legal",
        "{name} court case litigation",
        "{name} legal troubles problems",
        "{name} settlement paid damages",
        "{name} deposition testimony",
    ],
    "criminal": [
        "{name} criminal investigation",
        "{name} indicted indictment charges",
        "{name} fraud allegations",
        "{name} arrested arrest",
        "{name} FBI DOJ investigation",
    ],
    "regulatory": [
        "{name} SEC investigation subpoena",
        "{name} FTC antitrust probe",
        "{name} congressional hearing testimony",
        "{name} regulatory fine penalty",
        "{name} compliance violation",
    ],

    # === CONTROVERSIES ===
    "scandals": [
        "{name} scandal controversy",
        "{name} PR disaster crisis",
        "{name} resigned fired ousted",
        "{name} cover up hidden",
        "{name} exposed revealed truth",
    ],
    "criticism": [
        "{name} criticized backlash",
        "{name} accused allegations",
        "{name} problematic behavior",
        "{name} apology apologized mistake",
        "{name} cancelled cancel culture",
    ],
    "misconduct": [
        "{name} harassment allegations",
        "{name} discrimination lawsuit",
        "{name} toxic workplace culture",
        "{name} abuse of power",
        "{name} inappropriate behavior",
    ],

    # === FINANCIAL ===
    "financial": [
        "{name} financial troubles debt",
        "{name} bankruptcy money problems",
        "{name} conflict of interest",
        "{name} insider trading stock",
        "{name} tax evasion offshore",
    ],
    "compensation": [
        "{name} salary compensation pay",
        "{name} net worth wealth rich",
        "{name} stock options shares sold",
        "{name} golden parachute severance",
        "{name} bonus excessive pay",
    ],

    # === PRESSURE POINTS ===
    "vulnerabilities": [
        "{name} weakness vulnerable",
        "{name} insecure about defensive",
        "{name} failure failed mistake",
        "{name} embarrassing moment",
        "{name} regret admitted wrong",
    ],
    "leverage": [
        "{name} owes favor debt to",
        "{name} depends on relies",
        "{name} controlled by influence",
        "{name} compromised blackmail",
        "{name} secret hiding",
    ],
    "fears": [
        "{name} afraid fear worried",
        "{name} avoids refuses won't discuss",
        "{name} sensitive topic trigger",
        "{name} defensive about attacks",
        "{name} paranoid security",
    ],

    # === INTERNET ===
    "twitter": [
        "{name} twitter controversy tweet",
        "{name} deleted tweet regret",
        "{name} twitter fight argument",
        "{name} ratio twitter backlash",
        "{name} social media drama",
    ],
    "reddit": [
        'site:reddit.com "{name}"',
        '"{name}" reddit AMA',
        '"{name}" reddit hate',
        '"{name}" glassdoor reviews',
        '"{name}" blind app anonymous',
    ],
    "leaks": [
        "{name} leaked email document",
        "{name} whistleblower expose",
        "{name} internal memo revealed",
        "{name} secretly recorded",
        "{name} anonymous source said",
    ],

    # === POLICY ===
    "policy": [
        "{name} AI policy position views",
        "{name} China stance opinion",
        "{name} lobbying political donations",
        "{name} testimony congress senate",
        "{name} regulation against for",
    ],
    "hypocrisy": [
        "{name} hypocrite hypocrisy",
        "{name} contradicts contradiction",
        "{name} flip flop changed position",
        "{name} says one thing does another",
        "{name} criticized for same thing",
    ],
}


def generate_deep_terms(name: str, role: str, category: str = "") -> dict[str, list[str]]:
    """Generate comprehensive search terms using DSPy or fallback."""

    if DSPY_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        try:
            configure_dspy()
            predictor = dspy.Predict(DeepPersonaTerms)
            result = predictor(name=name, role=role, category=category)

            terms = {}
            for field in result.model_fields:
                if field.endswith("_terms"):
                    value = getattr(result, field, [])
                    if value:
                        terms[field.replace("_terms", "")] = value
            return terms
        except Exception as e:
            print(f"DSPy failed ({e}), using templates")

    # Fallback to templates
    return generate_template_terms(name, role)


def generate_template_terms(name: str, role: str) -> dict[str, list[str]]:
    """Generate terms using templates."""
    terms = {}
    for category, templates in DEEP_QUERY_TEMPLATES.items():
        terms[category] = [t.format(name=name, role=role) for t in templates]
    return terms


def flatten_terms(terms_dict: dict[str, list[str]]) -> list[str]:
    """Flatten categorized terms into single list."""
    all_terms = []
    seen = set()
    for category_terms in terms_dict.values():
        for term in category_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                all_terms.append(term)
    return all_terms


def main():
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Sam Altman"
    role = sys.argv[2] if len(sys.argv) > 2 else "frontier training demand"

    print(f"Generating deep search terms for: {name}")
    print(f"Role: {role}")
    print("=" * 60)

    terms = generate_deep_terms(name, role)

    total = 0
    for category, category_terms in terms.items():
        print(f"\n{category.upper()}:")
        for t in category_terms:
            print(f"  • {t}")
        total += len(category_terms)

    print(f"\n{'=' * 60}")
    print(f"Total: {total} search terms across {len(terms)} categories")

    flat = flatten_terms(terms)
    print(f"Unique terms: {len(flat)}")


if __name__ == "__main__":
    main()
