"""
Personality-Specific Tools for Agent Framework

Each personality archetype gets 2-4 specialized tools that align with their
cognitive style, making them more lifelike and effective.

Based on Qwen-Agent tool framework with support for:
- Function calling with streaming
- JSON Schema definitions
- Async/sync execution
"""
from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Base Tool Framework
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolParameter:
    """A single tool parameter definition."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


class BaseTool(ABC):
    """Base class for all personality tools."""

    def __init__(self):
        self.name: str = self.__class__.__name__.lower()
        self.description: str = ""
        self.parameters: List[ToolParameter] = []

    @abstractmethod
    async def call(self, params: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """Execute the tool with given parameters."""
        pass

    def to_function_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI/Qwen function calling schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Market Analyst Tools (INTJ - Strategic, Analytical)
# ─────────────────────────────────────────────────────────────────────────────


class MarketDataLookup(BaseTool):
    """Look up real-time market data for companies."""

    def __init__(self):
        super().__init__()
        self.name = "market_data_lookup"
        self.description = "Retrieve current market data including stock price, market cap, P/E ratio, and trading volume"
        self.parameters = [
            ToolParameter(
                name="company",
                type="string",
                description="Company name or ticker symbol",
                required=True
            ),
            ToolParameter(
                name="metrics",
                type="array",
                description="List of metrics to retrieve: price, market_cap, pe_ratio, volume, beta",
                required=False,
                default=["price", "market_cap"]
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate market data lookup."""
        company = params.get("company", "")
        metrics = params.get("metrics", ["price", "market_cap"])

        # Simulated data - in production, integrate with real API
        data = {
            "company": company,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {}
        }

        if "price" in metrics:
            data["data"]["price"] = round(random.uniform(50, 500), 2)
        if "market_cap" in metrics:
            data["data"]["market_cap_billions"] = round(random.uniform(10, 3000), 2)
        if "pe_ratio" in metrics:
            data["data"]["pe_ratio"] = round(random.uniform(10, 80), 2)
        if "volume" in metrics:
            data["data"]["volume_millions"] = round(random.uniform(1, 100), 2)
        if "beta" in metrics:
            data["data"]["beta"] = round(random.uniform(0.5, 2.0), 2)

        return data


class ValuationCalculator(BaseTool):
    """Calculate company valuation using multiple methods."""

    def __init__(self):
        super().__init__()
        self.name = "valuation_calculator"
        self.description = "Calculate company valuation using DCF, comparable companies, or precedent transactions"
        self.parameters = [
            ToolParameter(
                name="method",
                type="string",
                description="Valuation method to use",
                required=True,
                enum=["dcf", "comparable", "precedent_transaction"]
            ),
            ToolParameter(
                name="revenue_millions",
                type="number",
                description="Annual revenue in millions USD",
                required=True
            ),
            ToolParameter(
                name="growth_rate",
                type="number",
                description="Expected annual growth rate (0.0 to 1.0)",
                required=False,
                default=0.15
            ),
            ToolParameter(
                name="industry",
                type="string",
                description="Industry sector for comparable multiples",
                required=False
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate valuation."""
        method = params.get("method")
        revenue = params.get("revenue_millions", 0)
        growth = params.get("growth_rate", 0.15)
        industry = params.get("industry", "technology")

        result = {
            "method": method,
            "inputs": params,
            "timestamp": datetime.utcnow().isoformat()
        }

        if method == "dcf":
            # Simplified DCF
            years = 5
            discount_rate = 0.10
            terminal_multiple = 15

            pv_sum = 0
            for year in range(1, years + 1):
                fcf = revenue * (1 + growth) ** year * 0.20  # Assume 20% FCF margin
                pv = fcf / ((1 + discount_rate) ** year)
                pv_sum += pv

            terminal_value = revenue * (1 + growth) ** years * terminal_multiple
            terminal_pv = terminal_value / ((1 + discount_rate) ** years)

            result["valuation_millions"] = round(pv_sum + terminal_pv, 2)
            result["implied_multiple"] = round(result["valuation_millions"] / revenue, 2)

        elif method == "comparable":
            # Industry multiples
            multiples = {
                "technology": {"revenue": 8.5, "ebitda": 15.2},
                "saas": {"revenue": 12.0, "ebitda": 20.0},
                "ai": {"revenue": 15.0, "ebitda": 25.0},
                "hardware": {"revenue": 2.5, "ebitda": 10.0},
            }

            multiple = multiples.get(industry, {"revenue": 5.0, "ebitda": 12.0})
            result["valuation_millions"] = round(revenue * multiple["revenue"], 2)
            result["implied_multiple"] = multiple["revenue"]

        elif method == "precedent_transaction":
            # Recent transaction multiples (usually higher due to control premium)
            premium = 1.35  # 35% control premium
            base_multiple = 10.0
            result["valuation_millions"] = round(revenue * base_multiple * premium, 2)
            result["implied_multiple"] = round(base_multiple * premium, 2)
            result["control_premium"] = "35%"

        return result


class CompetitiveLandscape(BaseTool):
    """Analyze competitive positioning and market dynamics."""

    def __init__(self):
        super().__init__()
        self.name = "competitive_landscape"
        self.description = "Analyze competitive positioning, market share, and strategic threats"
        self.parameters = [
            ToolParameter(
                name="company",
                type="string",
                description="Target company to analyze",
                required=True
            ),
            ToolParameter(
                name="industry",
                type="string",
                description="Industry sector",
                required=True
            ),
            ToolParameter(
                name="analysis_type",
                type="string",
                description="Type of competitive analysis",
                required=False,
                enum=["five_forces", "positioning", "swot"],
                default="positioning"
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive landscape."""
        company = params.get("company", "")
        industry = params.get("industry", "")
        analysis_type = params.get("analysis_type", "positioning")

        result = {
            "company": company,
            "industry": industry,
            "analysis_type": analysis_type,
            "timestamp": datetime.utcnow().isoformat()
        }

        if analysis_type == "positioning":
            result["market_position"] = random.choice(["Leader", "Challenger", "Niche Player", "Emerging"])
            result["estimated_market_share"] = f"{random.randint(5, 45)}%"
            result["key_competitors"] = [
                f"Competitor {i}" for i in range(1, random.randint(3, 6))
            ]
            result["competitive_advantages"] = [
                random.choice([
                    "Technology leadership",
                    "Brand strength",
                    "Network effects",
                    "Cost structure",
                    "Customer lock-in"
                ])
                for _ in range(2)
            ]

        elif analysis_type == "five_forces":
            result["forces"] = {
                "supplier_power": random.choice(["Low", "Medium", "High"]),
                "buyer_power": random.choice(["Low", "Medium", "High"]),
                "competitive_rivalry": random.choice(["Low", "Medium", "High", "Very High"]),
                "threat_of_substitution": random.choice(["Low", "Medium", "High"]),
                "threat_of_new_entry": random.choice(["Low", "Medium", "High"]),
            }
            result["industry_attractiveness"] = random.choice(["Attractive", "Neutral", "Challenging"])

        return result


class TrendAnalyzer(BaseTool):
    """Identify and analyze market trends and momentum."""

    def __init__(self):
        super().__init__()
        self.name = "trend_analyzer"
        self.description = "Analyze market trends, momentum indicators, and sentiment shifts"
        self.parameters = [
            ToolParameter(
                name="sector",
                type="string",
                description="Market sector or company to analyze",
                required=True
            ),
            ToolParameter(
                name="timeframe",
                type="string",
                description="Analysis timeframe",
                required=False,
                enum=["1M", "3M", "6M", "1Y"],
                default="3M"
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends."""
        sector = params.get("sector", "")
        timeframe = params.get("timeframe", "3M")

        return {
            "sector": sector,
            "timeframe": timeframe,
            "trend_direction": random.choice(["Strongly Bullish", "Bullish", "Neutral", "Bearish", "Strongly Bearish"]),
            "momentum": random.choice(["Accelerating", "Steady", "Decelerating"]),
            "sentiment_score": round(random.uniform(-1.0, 1.0), 2),
            "key_drivers": [
                random.choice([
                    "AI/ML adoption",
                    "Regulatory changes",
                    "M&A activity",
                    "Economic conditions",
                    "Technology disruption"
                ])
                for _ in range(2)
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager Tools (Enneagram 6 - Security-focused, Skeptical)
# ─────────────────────────────────────────────────────────────────────────────


class RiskAssessment(BaseTool):
    """Comprehensive risk assessment across multiple dimensions."""

    def __init__(self):
        super().__init__()
        self.name = "risk_assessment"
        self.description = "Assess operational, financial, regulatory, and strategic risks"
        self.parameters = [
            ToolParameter(
                name="company",
                type="string",
                description="Company to assess",
                required=True
            ),
            ToolParameter(
                name="event_type",
                type="string",
                description="Type of corporate event",
                required=False,
                enum=["acquisition", "ipo", "expansion", "product_launch", "general"]
            ),
            ToolParameter(
                name="risk_categories",
                type="array",
                description="Specific risk categories to assess",
                required=False
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk assessment."""
        company = params.get("company", "")
        event_type = params.get("event_type", "general")

        return {
            "company": company,
            "event_type": event_type,
            "overall_risk_score": round(random.uniform(3.0, 8.5), 1),  # 1-10 scale
            "risk_profile": random.choice(["Conservative", "Moderate", "Aggressive", "High Risk"]),
            "risks": {
                "regulatory": {
                    "score": round(random.uniform(2.0, 9.0), 1),
                    "key_concerns": ["Antitrust", "Data privacy", "Industry regulations"],
                    "mitigation": "Compliance program enhancement needed"
                },
                "financial": {
                    "score": round(random.uniform(2.0, 9.0), 1),
                    "key_concerns": ["Debt levels", "Cash flow", "Valuation risk"],
                    "mitigation": "Financial buffer recommended"
                },
                "operational": {
                    "score": round(random.uniform(2.0, 9.0), 1),
                    "key_concerns": ["Integration complexity", "Culture clash", "Talent retention"],
                    "mitigation": "Detailed integration plan required"
                },
                "strategic": {
                    "score": round(random.uniform(2.0, 9.0), 1),
                    "key_concerns": ["Market timing", "Competitive response", "Technology obsolescence"],
                    "mitigation": "Scenario planning recommended"
                }
            },
            "red_flags": random.randint(0, 3),
            "recommended_actions": [
                "Conduct thorough due diligence",
                "Engage regulatory counsel",
                "Stress test financial models"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


class ComplianceChecker(BaseTool):
    """Check regulatory compliance and potential violations."""

    def __init__(self):
        super().__init__()
        self.name = "compliance_checker"
        self.description = "Evaluate compliance with regulations, identify potential violations"
        self.parameters = [
            ToolParameter(
                name="jurisdiction",
                type="string",
                description="Regulatory jurisdiction",
                required=True,
                enum=["US", "EU", "UK", "APAC", "Global"]
            ),
            ToolParameter(
                name="transaction_type",
                type="string",
                description="Type of transaction",
                required=False
            ),
            ToolParameter(
                name="company_size",
                type="string",
                description="Company size category",
                required=False,
                enum=["startup", "mid_market", "enterprise", "public"]
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance."""
        jurisdiction = params.get("jurisdiction", "US")
        transaction_type = params.get("transaction_type", "general")

        regulations = {
            "US": ["SEC regulations", "Hart-Scott-Rodino Act", "FCPA", "CFIUS review"],
            "EU": ["GDPR", "Digital Markets Act", "AI Act", "Competition law"],
            "UK": ["FCA rules", "Competition Act", "Data Protection Act"],
            "APAC": ["Various data localization laws", "Cross-border transfer restrictions"],
            "Global": ["OECD guidelines", "Anti-corruption", "Sanctions"]
        }

        return {
            "jurisdiction": jurisdiction,
            "compliance_status": random.choice(["Compliant", "Requires Review", "Potential Issues", "Non-Compliant"]),
            "applicable_regulations": regulations.get(jurisdiction, []),
            "compliance_score": round(random.uniform(60, 95), 1),
            "issues_identified": random.randint(0, 5),
            "critical_issues": random.randint(0, 2),
            "required_filings": [
                random.choice([
                    "Pre-merger notification",
                    "Foreign investment clearance",
                    "Data protection impact assessment",
                    "Material change disclosure"
                ])
            ],
            "estimated_clearance_time": f"{random.randint(30, 180)} days",
            "recommendations": [
                "Engage specialized counsel",
                "Prepare comprehensive filings",
                "Consider timing implications"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


class HistoricalFailures(BaseTool):
    """Look up historical failures and lessons learned."""

    def __init__(self):
        super().__init__()
        self.name = "historical_failures"
        self.description = "Search for similar past failures, bankruptcies, or failed transactions to identify warning signs"
        self.parameters = [
            ToolParameter(
                name="scenario_type",
                type="string",
                description="Type of scenario to research",
                required=True,
                enum=["acquisition", "ipo", "expansion", "product_launch", "bankruptcy"]
            ),
            ToolParameter(
                name="industry",
                type="string",
                description="Industry sector",
                required=False
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Look up historical failures."""
        scenario_type = params.get("scenario_type", "")
        industry = params.get("industry", "technology")

        failure_examples = {
            "acquisition": [
                "HP + Autonomy: $8.8B writedown due to accounting irregularities",
                "Microsoft + Nokia: Culture clash and market mistiming",
                "AOL + Time Warner: Failed synergy realization"
            ],
            "ipo": [
                "WeWork: Governance concerns and valuation collapse",
                "Theranos: Fraud and technology failure",
                "Pets.com: Unsustainable unit economics"
            ],
            "expansion": [
                "Uber China: Regulatory headwinds and competitive pressure",
                "Target Canada: Supply chain and localization failures"
            ]
        }

        return {
            "scenario_type": scenario_type,
            "industry": industry,
            "failures_found": random.randint(3, 8),
            "case_studies": failure_examples.get(scenario_type, ["Various historical failures"]),
            "common_failure_patterns": [
                random.choice([
                    "Overvaluation",
                    "Poor due diligence",
                    "Integration challenges",
                    "Regulatory obstacles",
                    "Market timing",
                    "Technology risk"
                ])
                for _ in range(3)
            ],
            "warning_signs": [
                "Aggressive growth assumptions",
                "Lack of cultural alignment",
                "Regulatory uncertainty"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Opportunity Scout Tools (ENFP - Enthusiastic, Possibility-focused)
# ─────────────────────────────────────────────────────────────────────────────


class PartnershipFinder(BaseTool):
    """Identify potential strategic partnerships and collaborations."""

    def __init__(self):
        super().__init__()
        self.name = "partnership_finder"
        self.description = "Discover potential strategic partners, acquisition targets, and collaboration opportunities"
        self.parameters = [
            ToolParameter(
                name="company",
                type="string",
                description="Company seeking partnerships",
                required=True
            ),
            ToolParameter(
                name="partnership_type",
                type="string",
                description="Type of partnership to explore",
                required=False,
                enum=["acquisition_target", "strategic_alliance", "joint_venture", "technology_partner", "distribution_partner"]
            ),
            ToolParameter(
                name="criteria",
                type="object",
                description="Partnership criteria (geographic, size, technology, etc.)",
                required=False
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find partnership opportunities."""
        company = params.get("company", "")
        partnership_type = params.get("partnership_type", "strategic_alliance")

        num_opportunities = random.randint(4, 12)

        return {
            "company": company,
            "partnership_type": partnership_type,
            "opportunities_found": num_opportunities,
            "top_candidates": [
                {
                    "name": f"Company {chr(65+i)}",
                    "fit_score": round(random.uniform(60, 95), 1),
                    "synergy_areas": random.sample([
                        "Technology complementarity",
                        "Market access",
                        "Talent acquisition",
                        "Product portfolio expansion",
                        "Geographic expansion"
                    ], 2),
                    "estimated_value_creation": f"${random.randint(50, 500)}M"
                }
                for i in range(min(5, num_opportunities))
            ],
            "recommended_approach": random.choice([
                "Direct outreach",
                "Intermediary introduction",
                "Competitive process",
                "Informal collaboration first"
            ]),
            "timestamp": datetime.utcnow().isoformat()
        }


class InnovationTracker(BaseTool):
    """Track emerging innovations and technology trends."""

    def __init__(self):
        super().__init__()
        self.name = "innovation_tracker"
        self.description = "Monitor emerging technologies, startups, and innovation trends"
        self.parameters = [
            ToolParameter(
                name="sector",
                type="string",
                description="Technology or market sector",
                required=True
            ),
            ToolParameter(
                name="stage",
                type="string",
                description="Innovation maturity stage",
                required=False,
                enum=["emerging", "growth", "mature"],
                default="emerging"
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Track innovations."""
        sector = params.get("sector", "")
        stage = params.get("stage", "emerging")

        return {
            "sector": sector,
            "stage": stage,
            "trending_technologies": [
                random.choice([
                    "Generative AI", "Quantum computing", "Edge AI",
                    "Synthetic biology", "Autonomous systems", "Web3",
                    "Climate tech", "Fusion energy", "Brain-computer interfaces"
                ])
                for _ in range(4)
            ],
            "hot_startups": random.randint(15, 50),
            "total_funding_ytd": f"${random.randint(1, 20)}B",
            "growth_rate": f"{random.randint(20, 200)}% YoY",
            "market_size_2030": f"${random.randint(10, 500)}B",
            "key_players": [f"Innovator {i}" for i in range(1, 6)],
            "adoption_curve": random.choice(["Early adoption", "Rapid growth", "Mainstream"]),
            "investment_opportunities": random.randint(3, 10),
            "timestamp": datetime.utcnow().isoformat()
        }


class SynergyCalculator(BaseTool):
    """Calculate potential synergies from M&A or partnerships."""

    def __init__(self):
        super().__init__()
        self.name = "synergy_calculator"
        self.description = "Estimate revenue and cost synergies from potential combinations"
        self.parameters = [
            ToolParameter(
                name="company_a",
                type="string",
                description="First company",
                required=True
            ),
            ToolParameter(
                name="company_b",
                type="string",
                description="Second company or target",
                required=True
            ),
            ToolParameter(
                name="synergy_type",
                type="string",
                description="Type of synergy to analyze",
                required=False,
                enum=["revenue", "cost", "both"],
                default="both"
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate synergies."""
        company_a = params.get("company_a", "")
        company_b = params.get("company_b", "")
        synergy_type = params.get("synergy_type", "both")

        result = {
            "company_a": company_a,
            "company_b": company_b,
            "analysis_type": synergy_type
        }

        if synergy_type in ["revenue", "both"]:
            result["revenue_synergies"] = {
                "cross_selling": f"${random.randint(50, 300)}M",
                "market_expansion": f"${random.randint(30, 200)}M",
                "product_bundling": f"${random.randint(20, 150)}M",
                "total_annual": f"${random.randint(100, 650)}M",
                "realization_timeline": f"{random.randint(2, 4)} years"
            }

        if synergy_type in ["cost", "both"]:
            result["cost_synergies"] = {
                "headcount_reduction": f"${random.randint(30, 200)}M",
                "facilities_consolidation": f"${random.randint(10, 80)}M",
                "technology_stack": f"${random.randint(15, 100)}M",
                "procurement_savings": f"${random.randint(20, 120)}M",
                "total_annual": f"${random.randint(75, 500)}M",
                "realization_timeline": f"{random.randint(1, 2)} years"
            }

        result["confidence_level"] = random.choice(["High", "Medium", "Low"])
        result["key_assumptions"] = [
            "Customer retention >85%",
            "Successful integration execution",
            "No major regulatory blocks"
        ]
        result["timestamp"] = datetime.utcnow().isoformat()

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Sector Specialist Tools (INTP - Deep Expertise, Theoretical)
# ─────────────────────────────────────────────────────────────────────────────


class TechnicalDeepDive(BaseTool):
    """Perform deep technical analysis of technology and architecture."""

    def __init__(self):
        super().__init__()
        self.name = "technical_deep_dive"
        self.description = "Analyze technical architecture, technology stack, and engineering capabilities"
        self.parameters = [
            ToolParameter(
                name="company",
                type="string",
                description="Company to analyze",
                required=True
            ),
            ToolParameter(
                name="focus_area",
                type="string",
                description="Technical area to focus on",
                required=False,
                enum=["architecture", "infrastructure", "algorithms", "data", "security"]
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical deep dive."""
        company = params.get("company", "")
        focus_area = params.get("focus_area", "architecture")

        return {
            "company": company,
            "focus_area": focus_area,
            "technical_maturity": random.choice(["Leading Edge", "Advanced", "Mature", "Legacy"]),
            "architecture_patterns": random.sample([
                "Microservices", "Event-driven", "Serverless",
                "Distributed systems", "Edge computing", "Mesh architecture"
            ], 3),
            "technology_stack": {
                "core_languages": random.sample(["Python", "Go", "Rust", "TypeScript", "C++"], 2),
                "frameworks": random.sample(["PyTorch", "TensorFlow", "React", "FastAPI"], 2),
                "infrastructure": random.sample(["Kubernetes", "AWS", "GCP", "Azure"], 2)
            },
            "technical_debt_score": round(random.uniform(2.0, 8.0), 1),
            "scalability_rating": random.choice(["Excellent", "Good", "Moderate", "Limited"]),
            "innovation_index": round(random.uniform(60, 95), 1),
            "engineering_quality": random.choice(["Exceptional", "Strong", "Average"]),
            "key_technical_risks": random.sample([
                "Legacy system dependencies",
                "Scalability constraints",
                "Security vulnerabilities",
                "Talent retention",
                "Technical obsolescence"
            ], 2),
            "timestamp": datetime.utcnow().isoformat()
        }


class PatentAnalysis(BaseTool):
    """Analyze patent portfolio and IP strength."""

    def __init__(self):
        super().__init__()
        self.name = "patent_analysis"
        self.description = "Evaluate patent portfolio, IP strength, and innovation output"
        self.parameters = [
            ToolParameter(
                name="company",
                type="string",
                description="Company to analyze",
                required=True
            ),
            ToolParameter(
                name="technology_area",
                type="string",
                description="Specific technology area",
                required=False
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patents."""
        company = params.get("company", "")
        tech_area = params.get("technology_area", "general")

        return {
            "company": company,
            "technology_area": tech_area,
            "total_patents": random.randint(50, 5000),
            "pending_applications": random.randint(10, 500),
            "patent_families": random.randint(20, 1000),
            "geographic_coverage": random.sample(["US", "EU", "China", "Japan", "Korea"], 3),
            "citation_index": round(random.uniform(5.0, 25.0), 1),
            "portfolio_strength": random.choice(["Dominant", "Strong", "Moderate", "Weak"]),
            "key_patent_areas": random.sample([
                "Machine learning", "Natural language processing",
                "Computer vision", "Hardware acceleration",
                "Distributed computing", "Security"
            ], 3),
            "litigation_risk": random.choice(["Low", "Medium", "High"]),
            "licensing_opportunities": random.randint(0, 15),
            "freedom_to_operate": random.choice(["Clear", "Some concerns", "Significant concerns"]),
            "timestamp": datetime.utcnow().isoformat()
        }


class IndustryBenchmark(BaseTool):
    """Compare against industry benchmarks and best practices."""

    def __init__(self):
        super().__init__()
        self.name = "industry_benchmark"
        self.description = "Benchmark performance against industry standards and competitors"
        self.parameters = [
            ToolParameter(
                name="company",
                type="string",
                description="Company to benchmark",
                required=True
            ),
            ToolParameter(
                name="metrics",
                type="array",
                description="Metrics to benchmark",
                required=False
            ),
            ToolParameter(
                name="peer_group",
                type="string",
                description="Peer group for comparison",
                required=False
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform benchmarking."""
        company = params.get("company", "")

        percentiles = {
            metric: random.choice(["Top 10%", "Top 25%", "Median", "Below Median"])
            for metric in ["revenue_growth", "margin", "efficiency", "innovation"]
        }

        return {
            "company": company,
            "peer_group_size": random.randint(10, 50),
            "percentile_rankings": percentiles,
            "performance_summary": {
                "revenue_growth": f"{random.randint(10, 80)}% YoY",
                "gross_margin": f"{random.randint(40, 90)}%",
                "r&d_intensity": f"{random.randint(10, 40)}% of revenue",
                "employee_productivity": f"${random.randint(200, 800)}k per employee"
            },
            "competitive_position": random.choice(["Market Leader", "Strong Performer", "Average", "Laggard"]),
            "gap_analysis": {
                "strengths": random.sample(["Innovation", "Efficiency", "Scale", "Talent"], 2),
                "opportunities": random.sample(["Market expansion", "Cost optimization", "Product development"], 2)
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Devil's Advocate Tools (Enneagram 8 - Challenger, Provocative)
# ─────────────────────────────────────────────────────────────────────────────


class ContrarianAnalysis(BaseTool):
    """Generate contrarian viewpoints and alternative scenarios."""

    def __init__(self):
        super().__init__()
        self.name = "contrarian_analysis"
        self.description = "Develop contrarian perspectives and challenge consensus views"
        self.parameters = [
            ToolParameter(
                name="consensus_view",
                type="string",
                description="The prevailing consensus or popular opinion",
                required=True
            ),
            ToolParameter(
                name="context",
                type="string",
                description="Additional context about the situation",
                required=False
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contrarian analysis."""
        consensus = params.get("consensus_view", "")

        return {
            "consensus_view": consensus,
            "contrarian_stance": random.choice([
                "Strongly Disagree", "Skeptical", "Cautious Disagreement"
            ]),
            "counter_arguments": [
                random.choice([
                    "Market timing concerns overlooked",
                    "Hidden costs not factored",
                    "Overestimating synergies",
                    "Regulatory risks understated",
                    "Competitive response underestimated",
                    "Technology risk ignored"
                ])
                for _ in range(3)
            ],
            "alternative_scenarios": [
                {
                    "scenario": "Bear Case",
                    "probability": f"{random.randint(20, 40)}%",
                    "outcome": "Significant value destruction"
                },
                {
                    "scenario": "Base Case Worse",
                    "probability": f"{random.randint(30, 50)}%",
                    "outcome": "Below expected returns"
                }
            ],
            "overlooked_risks": random.sample([
                "Cultural misalignment",
                "Customer attrition",
                "Technology obsolescence",
                "Regulatory backlash",
                "Management distraction"
            ], 3),
            "devils_advocate_recommendation": random.choice([
                "Proceed with extreme caution",
                "Reconsider the transaction",
                "Significantly reduce scope",
                "Delay until risks clarify"
            ]),
            "timestamp": datetime.utcnow().isoformat()
        }


class AssumptionChallenger(BaseTool):
    """Challenge key assumptions in analysis or business plans."""

    def __init__(self):
        super().__init__()
        self.name = "assumption_challenger"
        self.description = "Identify and stress-test critical assumptions"
        self.parameters = [
            ToolParameter(
                name="plan_or_thesis",
                type="string",
                description="Business plan or investment thesis to examine",
                required=True
            ),
            ToolParameter(
                name="assumptions",
                type="array",
                description="List of key assumptions to challenge",
                required=False
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Challenge assumptions."""
        plan = params.get("plan_or_thesis", "")

        return {
            "plan_analyzed": plan,
            "assumptions_identified": random.randint(5, 12),
            "critical_assumptions": [
                {
                    "assumption": random.choice([
                        "Market will grow at 30% annually",
                        "Customer retention remains >90%",
                        "No new competitors enter",
                        "Regulatory environment stable",
                        "Technology advantage sustainable"
                    ]),
                    "fragility": random.choice(["High", "Medium", "Low"]),
                    "impact_if_wrong": random.choice(["Catastrophic", "Severe", "Moderate"]),
                    "supporting_evidence": random.choice(["Weak", "Moderate", "Strong"]),
                    "alternative_scenario": "Could be 50% lower"
                }
                for _ in range(3)
            ],
            "stress_test_results": {
                "sensitivity_to_growth": "High - 10% change affects outcome 30%",
                "sensitivity_to_competition": "Medium",
                "sensitivity_to_regulation": "High"
            },
            "recommended_derisking": [
                "Develop contingency plans",
                "Build in larger safety margins",
                "Phase implementation",
                "Create early warning indicators"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


class WeaknessDetector(BaseTool):
    """Identify hidden weaknesses and vulnerabilities."""

    def __init__(self):
        super().__init__()
        self.name = "weakness_detector"
        self.description = "Uncover hidden weaknesses, vulnerabilities, and failure modes"
        self.parameters = [
            ToolParameter(
                name="subject",
                type="string",
                description="Company, plan, or strategy to examine",
                required=True
            ),
            ToolParameter(
                name="analysis_depth",
                type="string",
                description="Depth of analysis",
                required=False,
                enum=["surface", "detailed", "comprehensive"],
                default="detailed"
            )
        ]

    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect weaknesses."""
        subject = params.get("subject", "")
        depth = params.get("analysis_depth", "detailed")

        return {
            "subject": subject,
            "analysis_depth": depth,
            "weaknesses_identified": random.randint(4, 15),
            "critical_vulnerabilities": [
                {
                    "area": random.choice([
                        "Business model", "Technology", "Team",
                        "Market position", "Financial structure"
                    ]),
                    "severity": random.choice(["Critical", "High", "Medium"]),
                    "description": "Structural weakness that could undermine strategy",
                    "likelihood_of_impact": f"{random.randint(30, 80)}%"
                }
                for _ in range(3)
            ],
            "single_points_of_failure": random.randint(1, 4),
            "hidden_dependencies": random.sample([
                "Key person risk",
                "Single supplier dependency",
                "Technology platform lock-in",
                "Major customer concentration",
                "Regulatory approval dependency"
            ], 3),
            "early_warning_indicators": [
                "Customer churn increasing",
                "Employee turnover spike",
                "Competitive wins declining"
            ],
            "mitigation_priority": random.choice(["Urgent", "High", "Medium"]),
            "timestamp": datetime.utcnow().isoformat()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────


class PersonalityToolRegistry:
    """Registry mapping personality types to their specialized tools."""

    TOOL_SETS = {
        "market_analyst": {
            "personality": "INTJ",
            "archetype": "mbti_intj",
            "tools": [
                MarketDataLookup(),
                ValuationCalculator(),
                CompetitiveLandscape(),
                TrendAnalyzer()
            ]
        },
        "risk_manager": {
            "personality": "Enneagram 6",
            "archetype": "enneagram_6",
            "tools": [
                RiskAssessment(),
                ComplianceChecker(),
                HistoricalFailures()
            ]
        },
        "opportunity_scout": {
            "personality": "ENFP",
            "archetype": "mbti_enfp",
            "tools": [
                PartnershipFinder(),
                InnovationTracker(),
                SynergyCalculator()
            ]
        },
        "sector_specialist": {
            "personality": "INTP",
            "archetype": "mbti_intp",
            "tools": [
                TechnicalDeepDive(),
                PatentAnalysis(),
                IndustryBenchmark()
            ]
        },
        "contrarian": {
            "personality": "Enneagram 8",
            "archetype": "enneagram_8",
            "tools": [
                ContrarianAnalysis(),
                AssumptionChallenger(),
                WeaknessDetector()
            ]
        }
    }

    @classmethod
    def get_tools_for_role(cls, role_id: str) -> List[BaseTool]:
        """Get tools for a specific personality role."""
        tool_set = cls.TOOL_SETS.get(role_id)
        if tool_set:
            return tool_set["tools"]
        return []

    @classmethod
    def get_tool_schemas_for_role(cls, role_id: str) -> List[Dict[str, Any]]:
        """Get function calling schemas for a role's tools."""
        tools = cls.get_tools_for_role(role_id)
        return [tool.to_function_schema() for tool in tools]

    @classmethod
    def get_tool_by_name(cls, tool_name: str) -> Optional[BaseTool]:
        """Get a specific tool by name across all roles."""
        for role_data in cls.TOOL_SETS.values():
            for tool in role_data["tools"]:
                if tool.name == tool_name:
                    return tool
        return None

    @classmethod
    def list_all_tools(cls) -> Dict[str, List[str]]:
        """List all available tools by role."""
        return {
            role_id: [tool.name for tool in role_data["tools"]]
            for role_id, role_data in cls.TOOL_SETS.items()
        }


async def execute_tool_call(tool_name: str, params: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
    """Execute a tool call by name."""
    tool = PersonalityToolRegistry.get_tool_by_name(tool_name)
    if tool:
        return await tool.call(params)
    return {"error": f"Tool {tool_name} not found"}
