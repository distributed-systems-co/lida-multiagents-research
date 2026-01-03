"""Tests for the messaging system."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.messaging.messages import Message, MessageType, Priority, Envelope
from src.messaging.mailbox import InboundMailbox, OutboundMailbox
from src.messaging.broker import MessageBroker, BrokerConfig
from src.messaging.agent import Agent, AgentConfig, AgentStatus


class TestMessage:
    def test_message_creation(self):
        msg = Message(
            sender_id="agent_1",
            recipient_id="agent_2",
            msg_type=MessageType.DIRECT,
            payload={"key": "value"},
        )
        assert msg.sender_id == "agent_1"
        assert msg.recipient_id == "agent_2"
        assert msg.msg_type == MessageType.DIRECT
        assert msg.payload == {"key": "value"}
        assert msg.msg_id is not None
        assert msg.timestamp is not None

    def test_message_serialization(self):
        msg = Message(
            sender_id="agent_1",
            recipient_id="agent_2",
            msg_type=MessageType.REQUEST,
            payload={"action": "test"},
            priority=Priority.HIGH,
        )

        json_str = msg.to_json()
        restored = Message.from_json(json_str)

        assert restored.sender_id == msg.sender_id
        assert restored.recipient_id == msg.recipient_id
        assert restored.msg_type == msg.msg_type
        assert restored.payload == msg.payload
        assert restored.priority == msg.priority
        assert restored.msg_id == msg.msg_id

    def test_message_reply(self):
        original = Message(
            sender_id="agent_1",
            recipient_id="agent_2",
            msg_type=MessageType.REQUEST,
            payload={"query": "test"},
        )

        reply = original.reply({"answer": "response"})

        assert reply.sender_id == "agent_2"
        assert reply.recipient_id == "agent_1"
        assert reply.msg_type == MessageType.RESPONSE
        assert reply.correlation_id == original.msg_id


class TestEnvelope:
    def test_envelope_creation(self):
        msg = Message(
            sender_id="a",
            recipient_id="b",
            msg_type=MessageType.DIRECT,
            payload={},
        )
        envelope = Envelope(message=msg)

        assert envelope.message == msg
        assert envelope.route == []
        assert envelope.retry_count == 0

    def test_envelope_serialization(self):
        msg = Message(
            sender_id="a",
            recipient_id="b",
            msg_type=MessageType.DIRECT,
            payload={"test": True},
        )
        envelope = Envelope(message=msg, route=["hop1", "hop2"])

        json_str = envelope.to_json()
        restored = Envelope.from_json(json_str)

        assert restored.message.sender_id == msg.sender_id
        assert restored.route == ["hop1", "hop2"]


class TestInboundMailbox:
    @pytest.fixture
    def mailbox(self):
        return InboundMailbox(agent_id="test_agent", max_size=100)

    @pytest.mark.asyncio
    async def test_deliver_message(self, mailbox):
        msg = Message(
            sender_id="sender",
            recipient_id="test_agent",
            msg_type=MessageType.DIRECT,
            payload={"test": True},
        )
        envelope = Envelope(message=msg)

        result = await mailbox.deliver(envelope)

        assert result is True
        assert mailbox.stats.received == 1
        assert mailbox.pending_count() == 1

    @pytest.mark.asyncio
    async def test_filter_wrong_recipient(self, mailbox):
        msg = Message(
            sender_id="sender",
            recipient_id="other_agent",
            msg_type=MessageType.DIRECT,
            payload={},
        )
        envelope = Envelope(message=msg)

        result = await mailbox.deliver(envelope)

        assert result is False
        assert mailbox.pending_count() == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self, mailbox):
        # Deliver low priority first
        low_msg = Message(
            sender_id="s",
            recipient_id="test_agent",
            msg_type=MessageType.DIRECT,
            payload={"priority": "low"},
            priority=Priority.LOW,
        )
        await mailbox.deliver(Envelope(message=low_msg))

        # Then high priority
        high_msg = Message(
            sender_id="s",
            recipient_id="test_agent",
            msg_type=MessageType.DIRECT,
            payload={"priority": "high"},
            priority=Priority.HIGH,
        )
        await mailbox.deliver(Envelope(message=high_msg))

        # High priority should come first
        received = await mailbox.receive(timeout=0.1)
        assert received.message.payload["priority"] == "high"

    @pytest.mark.asyncio
    async def test_custom_filter(self, mailbox):
        # Add filter to reject certain messages
        mailbox.add_filter(lambda msg: msg.payload.get("allowed", False))

        allowed_msg = Message(
            sender_id="s",
            recipient_id="test_agent",
            msg_type=MessageType.DIRECT,
            payload={"allowed": True},
        )
        rejected_msg = Message(
            sender_id="s",
            recipient_id="test_agent",
            msg_type=MessageType.DIRECT,
            payload={"allowed": False},
        )

        result1 = await mailbox.deliver(Envelope(message=allowed_msg))
        result2 = await mailbox.deliver(Envelope(message=rejected_msg))

        assert result1 is True
        assert result2 is False
        assert mailbox.stats.dropped == 1


class TestOutboundMailbox:
    @pytest.mark.asyncio
    async def test_send_message(self):
        mock_broker = MagicMock()
        mock_broker.route = AsyncMock()

        mailbox = OutboundMailbox(
            agent_id="test_agent",
            broker=mock_broker,
            rate_limit=100,
        )

        msg_id = await mailbox.send(
            recipient_id="target",
            msg_type=MessageType.DIRECT,
            payload={"test": True},
        )

        assert msg_id is not None
        assert mailbox.pending_count() == 1


class TestMessageType:
    def test_all_message_types(self):
        # Ensure all message types are valid
        types = list(MessageType)
        assert len(types) > 0
        assert MessageType.DIRECT in types
        assert MessageType.BROADCAST in types
        assert MessageType.REQUEST in types
        assert MessageType.RESPONSE in types


class TestPriority:
    def test_priority_ordering(self):
        assert Priority.LOW < Priority.NORMAL
        assert Priority.NORMAL < Priority.HIGH
        assert Priority.HIGH < Priority.CRITICAL
