from __future__ import annotations
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Any
from .messages import Message, MessageType, Priority, Envelope

logger = logging.getLogger(__name__)


@dataclass
class MailboxStats:
    received: int = 0
    processed: int = 0
    dropped: int = 0
    errors: int = 0
    last_activity: Optional[datetime] = None


class InboundMailbox:
    """
    Receives and queues incoming messages for an agent.
    Supports priority queuing, filtering, and backpressure.
    """

    def __init__(
        self,
        agent_id: str,
        max_size: int = 10000,
        high_priority_ratio: float = 0.3,  # Reserve 30% for high priority
    ):
        self.agent_id = agent_id
        self.max_size = max_size
        self.high_priority_ratio = high_priority_ratio

        # Priority queues (higher priority = processed first)
        self._queues: dict[Priority, deque[Envelope]] = {
            Priority.CRITICAL: deque(maxlen=int(max_size * 0.1)),
            Priority.HIGH: deque(maxlen=int(max_size * 0.2)),
            Priority.NORMAL: deque(maxlen=int(max_size * 0.5)),
            Priority.LOW: deque(maxlen=int(max_size * 0.2)),
        }

        # Filters: functions that return True to accept message
        self._filters: list[Callable[[Message], bool]] = []

        # Handlers by message type
        self._handlers: dict[MessageType, Callable] = {}

        # Stats
        self.stats = MailboxStats()

        # Async event for new messages
        self._new_message = asyncio.Event()

        # Dead letter queue for failed messages
        self._dead_letters: deque[Envelope] = deque(maxlen=1000)

    def add_filter(self, filter_fn: Callable[[Message], bool]):
        """Add a filter function. Message must pass all filters to be accepted."""
        self._filters.append(filter_fn)

    def register_handler(self, msg_type: MessageType, handler: Callable):
        """Register a handler for a specific message type."""
        self._handlers[msg_type] = handler

    def _passes_filters(self, msg: Message) -> bool:
        return all(f(msg) for f in self._filters)

    def _check_ttl(self, msg: Message) -> bool:
        if msg.ttl_seconds is None:
            return True
        created = datetime.fromisoformat(msg.timestamp)
        return datetime.now(timezone.utc) - created < timedelta(seconds=msg.ttl_seconds)

    async def deliver(self, envelope: Envelope) -> bool:
        """
        Deliver a message to this mailbox.
        Returns True if accepted, False if dropped.
        """
        msg = envelope.message

        # Check if message is for us
        if msg.recipient_id != self.agent_id and msg.recipient_id != "*":
            logger.debug(f"Message not for {self.agent_id}, ignoring")
            return False

        # Check TTL
        if not self._check_ttl(msg):
            logger.debug(f"Message {msg.msg_id} expired (TTL)")
            self.stats.dropped += 1
            return False

        # Apply filters
        if not self._passes_filters(msg):
            logger.debug(f"Message {msg.msg_id} filtered out")
            self.stats.dropped += 1
            return False

        # Get appropriate queue
        queue = self._queues[msg.priority]

        # Check capacity (with backpressure for low priority)
        total_size = sum(len(q) for q in self._queues.values())
        if total_size >= self.max_size:
            if msg.priority <= Priority.NORMAL:
                logger.warning(f"Mailbox full, dropping low-priority message {msg.msg_id}")
                self.stats.dropped += 1
                return False

        # Add routing info
        envelope.route.append(self.agent_id)

        # Enqueue
        queue.append(envelope)
        self.stats.received += 1
        self.stats.last_activity = datetime.now(timezone.utc)

        # Signal new message
        self._new_message.set()

        return True

    async def receive(self, timeout: Optional[float] = None) -> Optional[Envelope]:
        """
        Receive the next message, respecting priority.
        Blocks until a message is available or timeout.
        """
        # Check queues in priority order
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            queue = self._queues[priority]
            if queue:
                return queue.popleft()

        # Wait for new message
        try:
            if timeout:
                await asyncio.wait_for(self._new_message.wait(), timeout)
            else:
                await self._new_message.wait()
            self._new_message.clear()

            # Try again
            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                queue = self._queues[priority]
                if queue:
                    return queue.popleft()
        except asyncio.TimeoutError:
            pass

        return None

    async def process_one(self) -> bool:
        """Process a single message using registered handlers."""
        envelope = await self.receive(timeout=0.1)
        if not envelope:
            return False

        msg = envelope.message
        handler = self._handlers.get(msg.msg_type)

        if handler:
            try:
                await handler(msg)
                self.stats.processed += 1
                return True
            except Exception as e:
                logger.error(f"Handler error for {msg.msg_id}: {e}")
                self.stats.errors += 1

                # Retry or dead letter
                envelope.retry_count += 1
                if envelope.retry_count < envelope.max_retries:
                    self._queues[msg.priority].append(envelope)
                else:
                    self._dead_letters.append(envelope)
                return False
        else:
            logger.warning(f"No handler for message type {msg.msg_type}")
            self._dead_letters.append(envelope)
            return False

    def pending_count(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def get_dead_letters(self) -> list[Envelope]:
        return list(self._dead_letters)


class OutboundMailbox:
    """
    Manages outgoing messages from an agent.
    Handles batching, rate limiting, and delivery confirmation.
    """

    def __init__(
        self,
        agent_id: str,
        broker: Any,  # MessageBroker
        rate_limit: int = 1000,  # Messages per second
        batch_size: int = 100,
    ):
        self.agent_id = agent_id
        self.broker = broker
        self.rate_limit = rate_limit
        self.batch_size = batch_size

        # Outbound queue
        self._queue: deque[Envelope] = deque(maxlen=50000)

        # Pending confirmations
        self._pending_acks: dict[str, asyncio.Future] = {}

        # Stats
        self.stats = MailboxStats()

        # Rate limiting
        self._tokens = rate_limit
        self._last_refill = datetime.now(timezone.utc)

        # Background sender task
        self._sender_task: Optional[asyncio.Task] = None
        self._running = False

    def _refill_tokens(self):
        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_refill).total_seconds()
        self._tokens = min(self.rate_limit, self._tokens + int(elapsed * self.rate_limit))
        self._last_refill = now

    async def send(
        self,
        recipient_id: str,
        msg_type: MessageType,
        payload: dict,
        priority: Priority = Priority.NORMAL,
        wait_ack: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """
        Queue a message for sending.
        Returns message ID. If wait_ack=True, waits for acknowledgment.
        """
        msg = Message(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            msg_type=msg_type,
            payload=payload,
            priority=priority,
            **kwargs,
        )

        envelope = Envelope(message=msg)
        self._queue.append(envelope)
        self.stats.received += 1

        if wait_ack:
            future = asyncio.get_event_loop().create_future()
            self._pending_acks[msg.msg_id] = future
            # Trigger immediate send
            await self._send_batch()
            try:
                await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                del self._pending_acks[msg.msg_id]
                raise TimeoutError(f"No ACK received for message {msg.msg_id}")

        return msg.msg_id

    async def broadcast(
        self,
        msg_type: MessageType,
        payload: dict,
        priority: Priority = Priority.NORMAL,
    ) -> str:
        """Broadcast to all agents."""
        return await self.send("*", msg_type, payload, priority)

    async def _send_batch(self):
        """Send a batch of messages."""
        self._refill_tokens()

        batch = []
        while self._queue and len(batch) < self.batch_size and self._tokens > 0:
            envelope = self._queue.popleft()
            batch.append(envelope)
            self._tokens -= 1

        for envelope in batch:
            try:
                await self.broker.route(envelope)
                self.stats.processed += 1
            except Exception as e:
                logger.error(f"Failed to send {envelope.message.msg_id}: {e}")
                self.stats.errors += 1

                # Retry
                envelope.retry_count += 1
                if envelope.retry_count < envelope.max_retries:
                    self._queue.appendleft(envelope)
                else:
                    self.stats.dropped += 1

    def receive_ack(self, msg_id: str):
        """Called when an ACK is received for a message we sent."""
        if msg_id in self._pending_acks:
            self._pending_acks[msg_id].set_result(True)
            del self._pending_acks[msg_id]

    async def start(self):
        """Start background sender."""
        self._running = True
        self._sender_task = asyncio.create_task(self._sender_loop())

    async def stop(self):
        """Stop background sender."""
        self._running = False
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass

    async def _sender_loop(self):
        while self._running:
            if self._queue:
                await self._send_batch()
            await asyncio.sleep(0.01)  # 10ms tick

    def pending_count(self) -> int:
        return len(self._queue)
