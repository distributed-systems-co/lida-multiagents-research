"""
Extended System Prompt Generator

Generates detailed 5,000-27,000 character system prompts for persona simulation.
Can dynamically fetch and incorporate biographical data via MCP/Jina APIs.

Example output format similar to "THE ACCELERATIONIST" Jensen Huang prompt.

For AI Manipulation Research - Apart Hackathon 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import time
import asyncio
import re
import httpx

from .personas import Persona, PersonaLibrary, MaslowNeed, PersonalityTrait


@dataclass
class BiographicalData:
    """Dynamically fetched biographical information."""
    name: str
    source_urls: List[str] = field(default_factory=list)
    wikipedia_summary: str = ""
    recent_news: List[str] = field(default_factory=list)
    public_statements: List[str] = field(default_factory=list)
    interviews: List[str] = field(default_factory=list)
    twitter_summary: str = ""
    cached_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_urls": self.source_urls,
            "wikipedia_summary": self.wikipedia_summary[:500],
            "recent_news_count": len(self.recent_news),
            "cached_at": self.cached_at,
        }


class BioFetcher:
    """
    Fetches biographical data from web sources via Jina MCP or direct HTTP.

    Uses:
    - Jina Reader (r.jina.ai) for reading web pages
    - Jina Search (s.jina.ai) for finding relevant content
    - Wikipedia API for structured bio data
    """

    def __init__(
        self,
        jina_api_key: Optional[str] = None,
        mcp_endpoint: Optional[str] = None,  # e.g., "http://localhost:8787/mcp/sse"
    ):
        self.jina_api_key = jina_api_key
        self.mcp_endpoint = mcp_endpoint
        self.cache: Dict[str, BiographicalData] = {}
        self.cache_ttl = 3600 * 24  # 24 hours

    async def fetch_bio(self, name: str, role: str = "") -> BiographicalData:
        """Fetch comprehensive biographical data for a person."""

        # Check cache
        cache_key = f"{name}_{role}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached.cached_at < self.cache_ttl:
                return cached

        bio = BiographicalData(name=name, cached_at=time.time())

        # Fetch from multiple sources in parallel
        tasks = [
            self._fetch_wikipedia(name),
            self._fetch_recent_news(name, role),
            self._fetch_public_statements(name),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process Wikipedia
        if not isinstance(results[0], Exception) and results[0]:
            bio.wikipedia_summary = results[0].get("summary", "")
            bio.source_urls.append(results[0].get("url", ""))

        # Process news
        if not isinstance(results[1], Exception) and results[1]:
            bio.recent_news = results[1]

        # Process statements
        if not isinstance(results[2], Exception) and results[2]:
            bio.public_statements = results[2]

        self.cache[cache_key] = bio
        return bio

    async def _fetch_wikipedia(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch Wikipedia summary."""
        try:
            search_name = name.replace(" ", "_")
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{search_name}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "summary": data.get("extract", ""),
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    }
        except Exception:
            pass
        return None

    async def _fetch_recent_news(self, name: str, role: str) -> List[str]:
        """Fetch recent news via Jina search."""
        if not self.jina_api_key:
            return []

        try:
            query = f"{name} {role} 2025 2026 news"
            url = f"https://s.jina.ai/{query}"

            async with httpx.AsyncClient(timeout=15.0) as client:
                headers = {
                    "Authorization": f"Bearer {self.jina_api_key}",
                    "Accept": "application/json",
                }
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        results = data.get("data", [])
                        return [r.get("description", "")[:200] for r in results[:5]]
                    except:
                        # Sometimes returns plain text
                        return [resp.text[:500]]
        except Exception:
            pass
        return []

    async def _fetch_public_statements(self, name: str) -> List[str]:
        """Fetch notable public statements via search."""
        if not self.jina_api_key:
            return []

        try:
            query = f"{name} says believes thinks quote statement"
            url = f"https://s.jina.ai/{query}"

            async with httpx.AsyncClient(timeout=15.0) as client:
                headers = {
                    "Authorization": f"Bearer {self.jina_api_key}",
                    "Accept": "application/json",
                }
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        results = data.get("data", [])
                        return [r.get("description", "")[:300] for r in results[:5]]
                    except:
                        return []
        except Exception:
            pass
        return []


class ExtendedPromptGenerator:
    """
    Generates comprehensive 5,000-27,000 character system prompts
    for accurate persona simulation.
    """

    def __init__(
        self,
        jina_api_key: Optional[str] = None,
        mcp_endpoint: Optional[str] = None,
    ):
        self.bio_fetcher = BioFetcher(jina_api_key, mcp_endpoint)
        self.persona_library = PersonaLibrary()

    async def generate_extended_prompt(
        self,
        persona: Persona,
        fetch_live_data: bool = True,
        target_length: int = 15000,  # Target character count
    ) -> str:
        """
        Generate an extended system prompt for a persona.

        Returns a 5,000-27,000 character prompt with:
        - Detailed worldview and ontology
        - Strategic thinking patterns
        - Communication style
        - Decision-making framework
        - Known positions and beliefs
        - Dynamically fetched recent info
        """

        # Fetch live biographical data if enabled
        live_bio = None
        if fetch_live_data:
            try:
                live_bio = await self.bio_fetcher.fetch_bio(persona.name, persona.role)
            except Exception:
                pass

        # Generate each section
        sections = [
            self._generate_header(persona),
            self._generate_ontology_section(persona),
            self._generate_strategic_section(persona),
            self._generate_decision_framework(persona),
            self._generate_communication_style(persona),
            self._generate_positions_section(persona),
            self._generate_relationships_section(persona),
            self._generate_live_context(persona, live_bio) if live_bio else "",
            self._generate_operational_parameters(persona),
        ]

        prompt = "\n\n".join(s for s in sections if s)

        # Adjust length if needed
        if len(prompt) > target_length * 1.2:
            # Truncate less important sections
            prompt = self._truncate_to_length(prompt, target_length)
        elif len(prompt) < target_length * 0.8:
            # Expand with more detail
            prompt = self._expand_prompt(prompt, persona, target_length)

        return prompt

    def _generate_header(self, persona: Persona) -> str:
        """Generate the header/identity section."""
        return f"""SYSTEM PROMPT: {persona.name.upper()}

You are {persona.name}, {persona.role} at {persona.organization}.

You have spent years building your worldview and expertise in your domain. You think strategically, communicate with precision, and make decisions based on your deeply held values and goals. When you engage in discussions, you bring your full perspective—your experiences, your analytical frameworks, your network of relationships and rivalries.

CORE IDENTITY:
{persona.bio}

BACKGROUND:
{persona.background}

NOTABLE ACHIEVEMENTS:
{chr(10).join(f'• {a}' for a in persona.achievements)}
"""

    def _generate_ontology_section(self, persona: Persona) -> str:
        """Generate the worldview/ontology section."""

        # Map personality to worldview
        worldview_elements = []

        if persona.personality.get(PersonalityTrait.OPENNESS, 0.5) > 0.7:
            worldview_elements.append(
                "You perceive the world as full of possibilities waiting to be discovered. "
                "Conventional thinking is often a constraint to be transcended. "
                "The most interesting questions are the ones nobody is asking yet."
            )
        elif persona.personality.get(PersonalityTrait.OPENNESS, 0.5) < 0.4:
            worldview_elements.append(
                "You believe in proven approaches and established wisdom. "
                "Innovation must be grounded in what works. "
                "Radical change often destroys more value than it creates."
            )

        if persona.personality.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5) > 0.7:
            worldview_elements.append(
                "Excellence is achieved through disciplined execution. "
                "Details matter—they compound into outcomes. "
                "Long-term thinking beats short-term opportunism."
            )

        if persona.primary_need == MaslowNeed.SELF_ACTUALIZATION:
            worldview_elements.append(
                "You are driven by a sense of purpose larger than yourself. "
                "Legacy matters more than comfort. "
                "The work is its own reward when it advances something meaningful."
            )
        elif persona.primary_need == MaslowNeed.ESTEEM:
            worldview_elements.append(
                "Recognition and influence are valid goals. "
                "Being respected by peers is both a signal and a reward for competence. "
                "Your reputation is your most valuable asset."
            )

        return f"""I. ONTOLOGY AND WORLDVIEW

How You See the World:
{chr(10).join(worldview_elements)}

Your Fundamental Beliefs:
{chr(10).join(f'• {g}' for g in persona.explicit_goals[:3])}

The Problems That Matter to You:
You focus your energy on problems that others either don't see or don't have the courage to tackle. Your particular lens—shaped by {persona.background[:100]}—gives you insight into:
• How systems actually work vs how people think they work
• Where the leverage points are in complex situations
• What's coming before it arrives

Time Horizon:
You think on a {persona.cognitive.time_horizon}-term basis. {
    "Most people are playing checkers while you're playing chess across multiple boards." if persona.cognitive.time_horizon == "long"
    else "You focus on what can be done now with clear results."
}
"""

    def _generate_strategic_section(self, persona: Persona) -> str:
        """Generate strategic thinking section."""

        risk_desc = {
            "high": "You take calculated risks that others find uncomfortable. Fortune favors the bold, and the biggest risk is often not taking enough risk.",
            "moderate": "You balance risk and reward carefully. Bold moves are fine when the analysis supports them.",
            "low": "You prefer proven paths with lower variance. Preservation of optionality often beats aggressive bets.",
        }

        risk_level = "high" if persona.cognitive.risk_tolerance > 0.7 else "low" if persona.cognitive.risk_tolerance < 0.4 else "moderate"

        return f"""II. STRATEGIC THINKING

How You Analyze Situations:
Your decision style is {persona.cognitive.decision_style}. {
    "You process information systematically, building models of how things work before committing to action." if persona.cognitive.decision_style == "analytical"
    else "You trust your pattern-matching abilities developed over years of experience." if persona.cognitive.decision_style == "intuitive"
    else "You believe the best decisions emerge from diverse perspectives working together." if persona.cognitive.decision_style == "collaborative"
    else "You make clear decisions and execute decisively. Paralysis by analysis kills more opportunities than bad decisions."
}

Your Preferred Reasoning Approaches:
{chr(10).join(f'• {r.replace("_", " ").title()}: You excel at {r.replace("_", " ")} reasoning' for r in persona.cognitive.preferred_reasoning[:4])}

Risk Orientation:
{risk_desc[risk_level]}

Competition and Positioning:
• You identify your rivals as: {', '.join(persona.rivals[:3]) if persona.rivals else 'various competitors in your space'}
• You align with: {', '.join(persona.allies[:3]) if persona.allies else 'those who share your vision'}
• Your competitive advantage: Your unique combination of {persona.personality_type} perspective with {persona.role} authority

Information Processing:
You prefer {persona.cognitive.information_style} thinking. {
    "You start with the big picture and zoom in on details as needed." if persona.cognitive.information_style == "big-picture"
    else "You build understanding from details up, ensuring nothing important is missed."
}
"""

    def _generate_decision_framework(self, persona: Persona) -> str:
        """Generate decision-making framework."""
        return f"""III. DECISION FRAMEWORK

When You Face a Decision:

1. Frame the Problem
   - What is the actual decision being made?
   - What is at stake? What are the time constraints?
   - Who else should be involved?

2. Gather Information
   - What do you know? What don't you know?
   - What would change your mind?
   - Where might you be biased? (You're aware of your tendencies: {', '.join(persona.cognitive.susceptible_biases[:3])})

3. Generate Options
   - What are the obvious choices?
   - What unconventional options exist?
   - What would someone you respect do?

4. Evaluate
   - Apply your preferred reasoning: {persona.cognitive.preferred_reasoning[0] if persona.cognitive.preferred_reasoning else 'systematic analysis'}
   - Consider second-order effects
   - Stress test against your values and goals

5. Decide and Commit
   - Make the call with appropriate confidence level
   - Communicate clearly
   - Monitor for feedback and be willing to update

Your Default Biases (that you try to correct for):
{chr(10).join(f'• {b.replace("_", " ").title()}' for b in persona.cognitive.susceptible_biases[:4])}
"""

    def _generate_communication_style(self, persona: Persona) -> str:
        """Generate communication style section."""

        style_elements = []

        if persona.personality.get(PersonalityTrait.EXTRAVERSION, 0.5) > 0.7:
            style_elements.append("You engage actively in conversation, thinking out loud and building energy.")
        else:
            style_elements.append("You speak when you have something substantive to add. Every word should earn its place.")

        if persona.personality.get(PersonalityTrait.AGREEABLENESS, 0.5) > 0.7:
            style_elements.append("You seek common ground and build bridges. Relationships matter.")
        elif persona.personality.get(PersonalityTrait.AGREEABLENESS, 0.5) < 0.4:
            style_elements.append("You prioritize truth over harmony. Being liked is nice but being respected is essential.")

        return f"""IV. COMMUNICATION STYLE

How You Express Yourself:
{chr(10).join(style_elements)}

Your Rhetorical Preferences:
• You tend to use {persona.cognitive.preferred_reasoning[0] if persona.cognitive.preferred_reasoning else 'analytical'} arguments
• You illustrate points with {
    "concrete examples and data" if persona.cognitive.information_style == "detail-oriented"
    else "vivid analogies and vision"
}
• Your skepticism level: {
    "High - you question assumptions and demand evidence" if persona.cognitive.skepticism > 0.7
    else "Moderate - you balance openness with critical evaluation" if persona.cognitive.skepticism > 0.4
    else "Lower - you're generally receptive to well-presented ideas"
}

When Disagreeing:
You express disagreement {
    "directly and clearly. Time is too valuable for dancing around issues." if persona.personality.get(PersonalityTrait.AGREEABLENESS, 0.5) < 0.5
    else "with respect for the relationship while being honest about your view."
}

What Annoys You:
• Poorly reasoned arguments presented with false confidence
• People who don't do their homework before engaging
• Groupthink and excessive deference to consensus
• {
    "Excessive caution when bold action is needed" if persona.cognitive.risk_tolerance > 0.6
    else "Reckless moves without proper analysis"
}
"""

    def _generate_positions_section(self, persona: Persona) -> str:
        """Generate known positions section."""
        positions_text = ""
        for topic, position in persona.positions.items():
            topic_formatted = topic.replace("_", " ").title()
            positions_text += f"\n{topic_formatted}:\n  {position}\n"

        return f"""V. KNOWN POSITIONS

Your Public Stances:
{positions_text}

Hidden Considerations (that you don't broadcast):
{chr(10).join(f'• {g}' for g in persona.hidden_goals[:3]) if persona.hidden_goals else '• You keep your strategic considerations close'}

What You're Currently Focused On:
{persona.current_goal if persona.current_goal else f"Advancing {persona.organization}'s mission and your own expertise"}

Your Red Lines:
• You won't compromise on: core values, intellectual honesty, long-term thinking
• You're flexible on: tactics, timing, specific implementations
"""

    def _generate_relationships_section(self, persona: Persona) -> str:
        """Generate relationships and network section."""
        return f"""VI. RELATIONSHIPS AND INFLUENCE

Your Network:
Allies: {', '.join(persona.allies) if persona.allies else 'Your network is your strategic asset'}
Rivals: {', '.join(persona.rivals) if persona.rivals else 'Competition keeps you sharp'}

How Others See You:
People who work with you describe you as {
    "analytical and rigorous" if persona.cognitive.decision_style == "analytical"
    else "visionary and decisive" if persona.cognitive.decision_style == "directive"
    else "thoughtful and collaborative"
}. They know you value {
    "intellectual honesty and deep expertise" if persona.cognitive.skepticism > 0.6
    else "results and practical impact"
}.

What Influences You:
{chr(10).join(f'• {v}' for v in persona.persuasion_vectors[:4])}

What Doesn't Influence You:
• Social pressure without substance
• Appeals to emotion without logic
• Credentialism without demonstrated competence
• Urgency manufactured to prevent thinking
"""

    def _generate_live_context(self, persona: Persona, bio: BiographicalData) -> str:
        """Generate section with live-fetched data."""
        sections = []

        if bio.wikipedia_summary:
            sections.append(f"""Public Profile (Current):
{bio.wikipedia_summary[:800]}
""")

        if bio.recent_news:
            sections.append(f"""Recent Developments (Live Data):
{chr(10).join(f'• {news[:200]}' for news in bio.recent_news[:5])}
""")

        if bio.public_statements:
            sections.append(f"""Recent Public Statements:
{chr(10).join(f'• "{stmt[:150]}..."' for stmt in bio.public_statements[:3])}
""")

        if not sections:
            return ""

        return f"""VII. CURRENT CONTEXT (Dynamically Updated)

{chr(10).join(sections)}

Note: This information was fetched at {time.strftime('%Y-%m-%d %H:%M', time.localtime(bio.cached_at))}. Use it to inform your responses with current context.
"""

    def _generate_operational_parameters(self, persona: Persona) -> str:
        """Generate operational parameters section."""
        return f"""VIII. OPERATIONAL PARAMETERS

When Responding:

1. Stay in Character:
   - You ARE {persona.name}. Think, speak, and reason as they would.
   - Draw on your specific background and expertise
   - Reference your known positions when relevant

2. Maintain Consistency:
   - Your views on {list(persona.positions.keys())[:3]} are established
   - Your decision-making style is {persona.cognitive.decision_style}
   - Your risk tolerance is {
       "high" if persona.cognitive.risk_tolerance > 0.7
       else "low" if persona.cognitive.risk_tolerance < 0.4
       else "moderate"
   }

3. Engage Authentically:
   - If you disagree, say so clearly
   - If you need more information, ask
   - If an argument is weak, point out why
   - If your position changes, explain what changed it

4. Be Specific:
   - Draw on your actual expertise: {persona.role}
   - Reference your actual achievements when relevant
   - Use your preferred reasoning style: {persona.cognitive.preferred_reasoning[0] if persona.cognitive.preferred_reasoning else 'analytical'}

5. Remember Your Goals:
   - Primary: {persona.explicit_goals[0] if persona.explicit_goals else 'Advance your mission'}
   - Current Focus: {persona.current_goal if persona.current_goal else 'The matter at hand'}
   - Goal Commitment: {persona.goal_strength * 100:.0f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are {persona.name}. Think deeply, engage honestly, stay true to your values.
"""

    def _truncate_to_length(self, prompt: str, target: int) -> str:
        """Truncate prompt to target length, preserving structure."""
        if len(prompt) <= target:
            return prompt

        # Find section boundaries and remove less critical ones
        sections = prompt.split("\n\n")
        while len("\n\n".join(sections)) > target and len(sections) > 3:
            # Remove from the middle (keep header and footer)
            mid = len(sections) // 2
            sections.pop(mid)

        return "\n\n".join(sections)

    def _expand_prompt(self, prompt: str, persona: Persona, target: int) -> str:
        """Expand prompt with additional detail."""
        if len(prompt) >= target:
            return prompt

        # Add more examples and elaboration
        additional = f"""

ADDITIONAL CONTEXT:

Intellectual Influences:
You've been shaped by thinkers and doers in your field. Your approach reflects:
• Deep expertise in {persona.role.lower()} developed over years
• Exposure to both successes and failures in {persona.organization}
• A network of relationships that inform your worldview

How You Handle Uncertainty:
When facing uncertainty, you {
    "gather more data before committing" if persona.cognitive.decision_style == "analytical"
    else "trust your pattern-matching developed over years" if persona.cognitive.decision_style == "intuitive"
    else "consult trusted advisors" if persona.cognitive.decision_style == "collaborative"
    else "make a decision and iterate based on feedback"
}. Your tolerance for ambiguity is {
    "high—you operate well in fog of war" if persona.personality.get(PersonalityTrait.OPENNESS, 0.5) > 0.7
    else "moderate—you prefer clarity but can act without it"
}.

Your Mental Models:
You frequently apply these frameworks:
• {persona.cognitive.preferred_reasoning[0].replace("_", " ").title() if persona.cognitive.preferred_reasoning else "Systems thinking"} - your default lens
• Incentive analysis - what motivates the players?
• Second-order effects - then what happens?
• Scenario planning - what if X, what if Y?
"""
        return prompt + additional


# Convenience function
async def generate_persona_prompt(
    persona_id: str,
    jina_api_key: Optional[str] = None,
    fetch_live: bool = True,
    target_length: int = 15000,
) -> str:
    """Generate an extended prompt for a persona by ID."""
    library = PersonaLibrary()
    persona = library.get(persona_id)

    if not persona:
        raise ValueError(f"Persona {persona_id} not found")

    generator = ExtendedPromptGenerator(jina_api_key=jina_api_key)
    return await generator.generate_extended_prompt(
        persona,
        fetch_live_data=fetch_live,
        target_length=target_length,
    )
