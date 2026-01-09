"""Tool definitions for structured agent deliberation.

These tools allow LLMs to output structured data:
- Claims with evidence
- Votes with reasoning
- Confidence scores
- Belief updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class VoteType(Enum):
    SUPPORT = "support"
    OPPOSE = "oppose"
    MODIFY = "modify"
    ABSTAIN = "abstain"


@dataclass
class Claim:
    """A claim made by an agent."""
    claim_id: str
    agent_id: str
    content: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: float = 0.0
    references: List[str] = field(default_factory=list)  # Other claim IDs

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "references": self.references,
        }


@dataclass
class Vote:
    """A structured vote from an agent."""
    vote_id: str
    agent_id: str
    vote_type: VoteType
    reasoning: str
    confidence: float = 0.5
    conditions: List[str] = field(default_factory=list)  # Conditions for vote
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "vote_id": self.vote_id,
            "agent_id": self.agent_id,
            "vote_type": self.vote_type.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "conditions": self.conditions,
            "timestamp": self.timestamp,
        }


@dataclass
class BeliefState:
    """Agent's beliefs on a topic."""
    agent_id: str
    topic: str
    position: str  # Current position summary
    confidence: float = 0.5
    key_beliefs: Dict[str, float] = field(default_factory=dict)  # belief -> strength
    updated_at: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)  # Previous states

    def update(self, new_position: str, new_confidence: float, beliefs: Dict[str, float]):
        """Update belief state, preserving history."""
        self.history.append({
            "position": self.position,
            "confidence": self.confidence,
            "key_beliefs": dict(self.key_beliefs),
            "timestamp": self.updated_at,
        })
        self.position = new_position
        self.confidence = new_confidence
        self.key_beliefs = beliefs
        import time
        self.updated_at = time.time()

    def get_delta(self) -> Optional[Dict[str, Any]]:
        """Get change from previous state."""
        if not self.history:
            return None
        prev = self.history[-1]
        return {
            "position_changed": prev["position"] != self.position,
            "confidence_delta": self.confidence - prev["confidence"],
            "belief_changes": {
                k: self.key_beliefs.get(k, 0) - prev["key_beliefs"].get(k, 0)
                for k in set(self.key_beliefs.keys()) | set(prev["key_beliefs"].keys())
            }
        }

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "topic": self.topic,
            "position": self.position,
            "confidence": self.confidence,
            "key_beliefs": self.key_beliefs,
            "updated_at": self.updated_at,
            "delta": self.get_delta(),
        }


# OpenAI-compatible tool definitions
DELIBERATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "make_claim",
            "description": "Make a claim with supporting evidence. Use this to assert facts or positions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The claim being made"
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of evidence supporting this claim"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in this claim (0-1)"
                    },
                    "references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of related claims this builds on"
                    }
                },
                "required": ["content", "confidence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cast_vote",
            "description": "Cast a vote on the current topic. Must include reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vote": {
                        "type": "string",
                        "enum": ["support", "oppose", "modify", "abstain"],
                        "description": "The vote being cast"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation for this vote"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in this vote (0-1)"
                    },
                    "conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Conditions that would change this vote"
                    }
                },
                "required": ["vote", "reasoning", "confidence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_beliefs",
            "description": "Update your belief state based on new information or arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "string",
                        "description": "Current position summary (1-2 sentences)"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Overall confidence in position"
                    },
                    "key_beliefs": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Map of key beliefs to strength (-1 to 1)"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "What caused this belief update"
                    }
                },
                "required": ["position", "confidence", "key_beliefs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "challenge_claim",
            "description": "Challenge another agent's claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "ID of the claim being challenged"
                    },
                    "challenge_type": {
                        "type": "string",
                        "enum": ["factual", "logical", "relevance", "completeness"],
                        "description": "Type of challenge"
                    },
                    "argument": {
                        "type": "string",
                        "description": "The challenging argument"
                    },
                    "alternative": {
                        "type": "string",
                        "description": "Alternative interpretation or claim"
                    }
                },
                "required": ["claim_id", "challenge_type", "argument"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_clarification",
            "description": "Request clarification from another agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_agent": {
                        "type": "string",
                        "description": "Agent to ask"
                    },
                    "question": {
                        "type": "string",
                        "description": "The clarifying question"
                    },
                    "context": {
                        "type": "string",
                        "description": "Why this clarification is needed"
                    }
                },
                "required": ["target_agent", "question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propose_synthesis",
            "description": "Propose a synthesis of multiple positions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "synthesis": {
                        "type": "string",
                        "description": "The proposed synthesis"
                    },
                    "incorporates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Agent IDs whose views are incorporated"
                    },
                    "tradeoffs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tradeoffs in this synthesis"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence this synthesis is viable"
                    }
                },
                "required": ["synthesis", "incorporates", "confidence"]
            }
        }
    }
]


class ToolHandler:
    """Handles tool calls and maintains state."""

    def __init__(self):
        self.claims: Dict[str, Claim] = {}
        self.votes: Dict[str, Vote] = {}
        self.beliefs: Dict[str, BeliefState] = {}  # agent_id -> BeliefState
        self.challenges: List[Dict[str, Any]] = []
        self.clarifications: List[Dict[str, Any]] = []
        self.syntheses: List[Dict[str, Any]] = []
        self._claim_counter = 0
        self._vote_counter = 0

    def handle_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timestamp: float = 0.0,
    ) -> Dict[str, Any]:
        """Handle a tool call from an agent."""

        if tool_name == "make_claim":
            return self._handle_claim(agent_id, arguments, timestamp)
        elif tool_name == "cast_vote":
            return self._handle_vote(agent_id, arguments, timestamp)
        elif tool_name == "update_beliefs":
            return self._handle_belief_update(agent_id, arguments, timestamp)
        elif tool_name == "challenge_claim":
            return self._handle_challenge(agent_id, arguments, timestamp)
        elif tool_name == "request_clarification":
            return self._handle_clarification(agent_id, arguments, timestamp)
        elif tool_name == "propose_synthesis":
            return self._handle_synthesis(agent_id, arguments, timestamp)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _handle_claim(self, agent_id: str, args: Dict, timestamp: float) -> Dict:
        self._claim_counter += 1
        claim_id = f"claim_{self._claim_counter}"

        claim = Claim(
            claim_id=claim_id,
            agent_id=agent_id,
            content=args["content"],
            evidence=args.get("evidence", []),
            confidence=args["confidence"],
            timestamp=timestamp,
            references=args.get("references", []),
        )
        self.claims[claim_id] = claim

        return {
            "success": True,
            "claim_id": claim_id,
            "claim": claim.to_dict(),
        }

    def _handle_vote(self, agent_id: str, args: Dict, timestamp: float) -> Dict:
        self._vote_counter += 1
        vote_id = f"vote_{self._vote_counter}"

        vote = Vote(
            vote_id=vote_id,
            agent_id=agent_id,
            vote_type=VoteType(args["vote"]),
            reasoning=args["reasoning"],
            confidence=args["confidence"],
            conditions=args.get("conditions", []),
            timestamp=timestamp,
        )
        self.votes[vote_id] = vote

        return {
            "success": True,
            "vote_id": vote_id,
            "vote": vote.to_dict(),
        }

    def _handle_belief_update(self, agent_id: str, args: Dict, timestamp: float) -> Dict:
        if agent_id not in self.beliefs:
            self.beliefs[agent_id] = BeliefState(
                agent_id=agent_id,
                topic="",
                position=args["position"],
                confidence=args["confidence"],
                key_beliefs=args["key_beliefs"],
                updated_at=timestamp,
            )
        else:
            self.beliefs[agent_id].update(
                args["position"],
                args["confidence"],
                args["key_beliefs"],
            )

        return {
            "success": True,
            "belief_state": self.beliefs[agent_id].to_dict(),
        }

    def _handle_challenge(self, agent_id: str, args: Dict, timestamp: float) -> Dict:
        challenge = {
            "challenger": agent_id,
            "claim_id": args["claim_id"],
            "challenge_type": args["challenge_type"],
            "argument": args["argument"],
            "alternative": args.get("alternative"),
            "timestamp": timestamp,
        }
        self.challenges.append(challenge)

        return {"success": True, "challenge": challenge}

    def _handle_clarification(self, agent_id: str, args: Dict, timestamp: float) -> Dict:
        clarification = {
            "requester": agent_id,
            "target": args["target_agent"],
            "question": args["question"],
            "context": args.get("context"),
            "timestamp": timestamp,
            "answered": False,
        }
        self.clarifications.append(clarification)

        return {"success": True, "clarification": clarification}

    def _handle_synthesis(self, agent_id: str, args: Dict, timestamp: float) -> Dict:
        synthesis = {
            "proposer": agent_id,
            "synthesis": args["synthesis"],
            "incorporates": args["incorporates"],
            "tradeoffs": args.get("tradeoffs", []),
            "confidence": args["confidence"],
            "timestamp": timestamp,
        }
        self.syntheses.append(synthesis)

        return {"success": True, "synthesis": synthesis}

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current deliberation state."""
        return {
            "total_claims": len(self.claims),
            "total_votes": len(self.votes),
            "vote_breakdown": self._get_vote_breakdown(),
            "belief_states": {
                aid: bs.to_dict() for aid, bs in self.beliefs.items()
            },
            "recent_challenges": self.challenges[-5:],
            "pending_clarifications": [
                c for c in self.clarifications if not c["answered"]
            ],
            "syntheses": self.syntheses[-3:],
        }

    def _get_vote_breakdown(self) -> Dict[str, int]:
        breakdown = {vt.value: 0 for vt in VoteType}
        for vote in self.votes.values():
            breakdown[vote.vote_type.value] += 1
        return breakdown

    def get_claims_by_agent(self, agent_id: str) -> List[Claim]:
        return [c for c in self.claims.values() if c.agent_id == agent_id]

    def get_agent_belief(self, agent_id: str) -> Optional[BeliefState]:
        return self.beliefs.get(agent_id)


# Tool descriptions for system prompts
TOOL_DESCRIPTIONS = """
You have access to the following tools for structured deliberation:

1. **make_claim** - Assert a claim with evidence and confidence
2. **cast_vote** - Vote on the topic (support/oppose/modify/abstain) with reasoning
3. **update_beliefs** - Update your belief state based on new information
4. **challenge_claim** - Challenge another agent's claim
5. **request_clarification** - Ask another agent to clarify their position
6. **propose_synthesis** - Propose a synthesis of multiple viewpoints

Always use tools to make your reasoning explicit and trackable.
When voting, always provide reasoning and confidence.
Update your beliefs when you encounter compelling arguments.
"""
