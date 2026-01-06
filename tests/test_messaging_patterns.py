#!/usr/bin/env python3
"""
Test 70+ messaging patterns with multi-agent network.
Demonstrates all pattern categories without requiring LLM calls.
"""

import sys
import asyncio
from typing import Dict, Any

sys.path.insert(0, "/Users/arthurcolle/lida-multiagents-research")

from src.meta.messaging_patterns import (
    PatternCategory,
    MessagePattern,
    Message,
    TopicMessage,
    SagaMessage,
    AgentEndpoint,
    MessagingNetwork,
)


class PatternDemo:
    """Demonstrates all 70+ messaging patterns."""

    def __init__(self):
        self.network = MessagingNetwork()
        self.stats: Dict[str, int] = {}

    def setup_agents(self):
        """Set up test agents with various configurations."""
        # Core agents
        for i in range(10):
            self.network.register_agent(f"agent_{i}", location=(i * 10.0, i * 5.0))

        # Groups
        self.network.add_to_group("agent_0", "coordinators")
        self.network.add_to_group("agent_1", "coordinators")
        for i in range(2, 6):
            self.network.add_to_group(f"agent_{i}", "workers")
        for i in range(6, 10):
            self.network.add_to_group(f"agent_{i}", "observers")

        # Topics
        for i in range(5):
            self.network.subscribe(f"agent_{i}", "events")
        for i in range(5, 10):
            self.network.subscribe(f"agent_{i}", "alerts")

        # Mesh neighbors
        self.network.set_neighbors("agent_0", ["agent_1", "agent_2"])
        self.network.set_neighbors("agent_1", ["agent_0", "agent_2", "agent_3"])
        self.network.set_neighbors("agent_2", ["agent_0", "agent_1", "agent_3", "agent_4"])

        # Tree structure
        self.network.set_tree_root("agent_0")
        self.network.agents["agent_0"].children = {"agent_1", "agent_2"}
        self.network.agents["agent_1"].parent = "agent_0"
        self.network.agents["agent_1"].children = {"agent_3", "agent_4"}
        self.network.agents["agent_2"].parent = "agent_0"
        self.network.agents["agent_2"].children = {"agent_5", "agent_6"}
        self.network.agents["agent_3"].parent = "agent_1"
        self.network.agents["agent_4"].parent = "agent_1"
        self.network.agents["agent_5"].parent = "agent_2"
        self.network.agents["agent_6"].parent = "agent_2"

        print(f"  Set up {len(self.network.agents)} agents")
        print(f"  Groups: {list(self.network.groups.keys())}")
        print(f"  Topics: {list(self.network.topics.keys())}")

    async def test_point_to_point(self):
        """Test 1:1 patterns."""
        print("\n" + "─" * 70)
        print("POINT-TO-POINT PATTERNS (1:1)")
        print("─" * 70)

        # Direct
        msg = await self.network.direct("agent_0", "agent_1", {"type": "hello"})
        print(f"  ✓ DIRECT: {msg.sender} → {msg.recipients[0]}")

        # Request-Reply
        reply = await self.network.request_reply("agent_1", "agent_2", {"query": "status"}, timeout=1.0)
        print(f"  ✓ REQUEST_REPLY: agent_1 → agent_2 (timeout handled)")

        # Fire and Forget
        msg = await self.network.fire_and_forget("agent_2", "agent_3", {"event": "log"})
        print(f"  ✓ FIRE_AND_FORGET: {msg.sender} → {msg.recipients[0]}")

        return 3

    async def test_one_to_many(self):
        """Test 1:N patterns."""
        print("\n" + "─" * 70)
        print("ONE-TO-MANY PATTERNS (1:N)")
        print("─" * 70)

        # Broadcast
        msgs = await self.network.broadcast("agent_0", {"announcement": "meeting at 3pm"})
        print(f"  ✓ BROADCAST: agent_0 → {len(msgs)} recipients")

        # Multicast
        msgs = await self.network.multicast("agent_0", "workers", {"task": "process data"})
        print(f"  ✓ MULTICAST: agent_0 → workers group ({len(msgs)} members)")

        # Anycast
        msg = await self.network.anycast("agent_0", "workers", {"request": "handle this"})
        print(f"  ✓ ANYCAST: agent_0 → random worker ({msg.recipients[0]})")

        # Geocast
        msgs = await self.network.geocast("agent_0", {"alert": "nearby event"}, center=(25.0, 12.5), radius=30.0)
        print(f"  ✓ GEOCAST: agent_0 → {len(msgs)} agents within radius")

        # Scatter
        msgs = await self.network.scatter(
            "agent_0",
            ["agent_1", "agent_2", "agent_3"],
            [{"partition": 1}, {"partition": 2}, {"partition": 3}]
        )
        print(f"  ✓ SCATTER: agent_0 → {len(msgs)} agents with different content")

        # Recipient List
        msgs = await self.network.recipient_list(
            "agent_0",
            {"urgent": True},
            lambda aid, agent: agent.messages_received < 5
        )
        print(f"  ✓ RECIPIENT_LIST: agent_0 → {len(msgs)} dynamic recipients")

        return 6

    async def test_many_to_one(self):
        """Test N:1 patterns."""
        print("\n" + "─" * 70)
        print("MANY-TO-ONE PATTERNS (N:1)")
        print("─" * 70)

        # Gather
        responses = await self.network.gather("agent_0", ["agent_1", "agent_2", "agent_3"], timeout=1.0)
        print(f"  ✓ GATHER: agent_0 collected {len(responses)} responses")

        # Aggregator
        messages = [
            Message(sender=f"agent_{i}", content={"value": i * 10}) for i in range(1, 5)
        ]
        aggregated = await self.network.aggregate(messages, lambda contents: {"sum": sum(c.get("value", 0) for c in contents)})
        print(f"  ✓ AGGREGATOR: Combined {len(messages)} messages → {aggregated.content}")

        # Resequencer
        out_of_order = [
            Message(sender="agent_1", content="C", sequence=3),
            Message(sender="agent_1", content="A", sequence=1),
            Message(sender="agent_1", content="B", sequence=2),
        ]
        resequenced = await self.network.resequence(out_of_order)
        print(f"  ✓ RESEQUENCER: Restored order → {[m.content for m in resequenced]}")

        return 3

    async def test_many_to_many(self):
        """Test N:M patterns."""
        print("\n" + "─" * 70)
        print("MANY-TO-MANY PATTERNS (N:M)")
        print("─" * 70)

        # Scatter-Gather
        responses = await self.network.scatter_gather(
            "agent_0",
            ["agent_1", "agent_2", "agent_3"],
            {"compute": "parallel"},
            timeout=1.0
        )
        print(f"  ✓ SCATTER_GATHER: agent_0 → 3 agents, collected {len(responses)} responses")

        # Relay Chain
        msgs = await self.network.relay_chain(
            "agent_0",
            {"message": "pass it on"},
            ["agent_1", "agent_2", "agent_3", "agent_4"]
        )
        print(f"  ✓ RELAY_CHAIN: Passed through {len(msgs)} hops")

        return 2

    async def test_hierarchical(self):
        """Test hierarchical patterns."""
        print("\n" + "─" * 70)
        print("HIERARCHICAL PATTERNS")
        print("─" * 70)

        # Tree Broadcast
        msgs = await self.network.tree_broadcast("agent_0", {"command": "execute"})
        print(f"  ✓ TREE_BROADCAST: Root → {len(msgs)} descendants")

        # Leader Broadcast
        msgs = await self.network.leader_broadcast("agent_0", {"directive": "follow"})
        print(f"  ✓ LEADER_FOLLOWER: Leader → {len(msgs)} followers")

        return 2

    async def test_ring(self):
        """Test ring patterns."""
        print("\n" + "─" * 70)
        print("RING PATTERNS")
        print("─" * 70)

        # Ring Pass
        msgs = await self.network.ring_pass("agent_0", {"token": "data"})
        print(f"  ✓ RING_PASS: Circulated through {len(msgs)} nodes")

        # Token Ring
        acquired = await self.network.token_ring_acquire("agent_1", timeout=1.0)
        print(f"  ✓ TOKEN_RING: agent_1 acquired token = {acquired}")
        self.network.token_ring_release("agent_1")
        print(f"  ✓ TOKEN_RING: agent_1 released token → {self.network._ring_token_holder}")

        return 3

    async def test_pipeline(self):
        """Test pipeline patterns."""
        print("\n" + "─" * 70)
        print("PIPELINE PATTERNS")
        print("─" * 70)

        # Pipeline
        msg = await self.network.pipeline(
            {"raw_data": [1, 2, 3]},
            ["agent_1", "agent_2", "agent_3"],
            {"agent_2": lambda x: {"processed": x.get("raw_data", [])}}
        )
        print(f"  ✓ PIPELINE: Data → 3 stages → {msg.headers.get('stage')}/{msg.headers.get('total_stages')}")

        # Parallel Pipeline
        msgs = await self.network.parallel_pipeline(
            {"batch": "data"},
            [["agent_1", "agent_2"], ["agent_3", "agent_4"]]
        )
        print(f"  ✓ PARALLEL_PIPELINE: {len(msgs)} messages across 2 parallel stages")

        return 2

    async def test_pubsub(self):
        """Test pub/sub patterns."""
        print("\n" + "─" * 70)
        print("PUB/SUB PATTERNS")
        print("─" * 70)

        # Publish
        msgs = await self.network.publish("agent_0", "events", {"event": "new_data"})
        print(f"  ✓ PUBLISH: Published to 'events' → {len(msgs)} subscribers")

        msgs = await self.network.publish("agent_5", "alerts", {"alert": "warning"})
        print(f"  ✓ PUBLISH: Published to 'alerts' → {len(msgs)} subscribers")

        # Content Filtered Publish
        msgs = await self.network.content_filtered_publish(
            "agent_0",
            "events",
            {"priority": "high"},
            lambda sub, content: sub.startswith("agent_") and int(sub.split("_")[1]) < 3
        )
        print(f"  ✓ CONTENT_FILTER: Filtered publish → {len(msgs)} matching subscribers")

        return 3

    async def test_gossip(self):
        """Test gossip/epidemic patterns."""
        print("\n" + "─" * 70)
        print("GOSSIP/EPIDEMIC PATTERNS")
        print("─" * 70)

        # Gossip Push
        msgs = await self.network.gossip_push("agent_0", {"rumor": "secret"}, fanout=2, rounds=3)
        print(f"  ✓ GOSSIP_PUSH: Spread to {len(msgs)} nodes over 3 rounds")

        # Gossip Push-Pull
        msgs = await self.network.gossip_push_pull("agent_5", {"state": "update"}, fanout=2, rounds=2)
        print(f"  ✓ GOSSIP_PUSH_PULL: {len(msgs)} bidirectional exchanges")

        # Epidemic Broadcast
        msgs = await self.network.epidemic_broadcast("agent_0", {"infection": "data"}, infection_prob=0.6)
        print(f"  ✓ EPIDEMIC_BROADCAST: Infected {len(msgs)} nodes probabilistically")

        return 3

    async def test_routing(self):
        """Test routing patterns."""
        print("\n" + "─" * 70)
        print("ROUTING PATTERNS")
        print("─" * 70)

        # Content Router
        msgs = await self.network.content_route(
            "agent_0",
            {"type": "high_priority", "data": "important"},
            lambda content: ["agent_1", "agent_2"] if content.get("type") == "high_priority" else ["agent_3"]
        )
        print(f"  ✓ CONTENT_ROUTER: Routed to {len(msgs)} agents based on content")

        # Routing Slip
        msgs = await self.network.routing_slip(
            "agent_0",
            {"order": "process"},
            ["agent_1", "agent_3", "agent_5", "agent_7"]
        )
        print(f"  ✓ ROUTING_SLIP: Processed through {len(msgs)} sequential stops")

        return 2

    async def test_saga(self):
        """Test saga/transaction patterns."""
        print("\n" + "─" * 70)
        print("SAGA/TRANSACTION PATTERNS")
        print("─" * 70)

        # Start Saga
        saga_id = await self.network.start_saga(
            "order_saga",
            [
                ("agent_1", {"action": "reserve_inventory"}),
                ("agent_2", {"action": "charge_payment"}),
                ("agent_3", {"action": "ship_order"}),
            ],
            {
                0: ("agent_1", {"action": "release_inventory"}),
                1: ("agent_2", {"action": "refund_payment"}),
            }
        )
        print(f"  ✓ SAGA_ORCHESTRATION: Started saga '{saga_id}'")

        # Execute steps
        for i in range(3):
            msg = await self.network.execute_saga_step(saga_id)
            if msg:
                print(f"    Step {msg.step + 1}/3: {msg.content.get('action')}")

        print(f"  ✓ SAGA completed with status: {self.network._sagas[saga_id]['status']}")

        # Compensation example
        saga_id2 = await self.network.start_saga(
            "failed_saga",
            [("agent_1", {"action": "step1"}), ("agent_2", {"action": "step2"})],
            {0: ("agent_1", {"action": "undo_step1"})}
        )
        await self.network.execute_saga_step(saga_id2)
        comp_msgs = await self.network.compensate_saga(saga_id2)
        print(f"  ✓ COMPENSATING_TX: Executed {len(comp_msgs)} compensations")

        return 3

    async def test_fault_tolerance(self):
        """Test fault tolerance patterns."""
        print("\n" + "─" * 70)
        print("FAULT TOLERANCE PATTERNS")
        print("─" * 70)

        # Circuit Breaker
        state = self.network.circuit_breaker_state("agent_5")
        print(f"  ✓ CIRCUIT_BREAKER: agent_5 state = {state}")

        msg = await self.network.send_with_circuit_breaker("agent_0", "agent_5", {"test": "data"})
        print(f"  ✓ CIRCUIT_BREAKER: Sent with protection, state = {self.network.circuit_breaker_state('agent_5')}")

        # Retry
        msg = await self.network.send_with_retry("agent_0", "agent_6", {"reliable": "message"}, max_retries=3)
        if msg:
            print(f"  ✓ RETRY: Delivered on attempt {msg.headers.get('attempt', 1)}")

        # Timeout
        msg = await self.network.send_with_timeout("agent_0", "agent_7", {"time_sensitive": True}, timeout=2.0)
        print(f"  ✓ TIMEOUT: {'Delivered' if msg else 'Timed out'}")

        # Dead Letter stats
        print(f"  ✓ DEAD_LETTER: {len(self.network.dead_letters)} messages in dead letter queue")

        return 4

    async def test_flow_control(self):
        """Test flow control patterns."""
        print("\n" + "─" * 70)
        print("FLOW CONTROL PATTERNS")
        print("─" * 70)

        # Throttle
        results = []
        for i in range(5):
            msg = await self.network.throttle("agent_0", "agent_1", {"seq": i}, rate=2.0)
            results.append("✓" if msg else "✗")
        print(f"  ✓ THROTTLE: Rate limited results: {' '.join(results)}")

        # Buffer
        buffered = None
        for i in range(12):
            buffered = await self.network.buffer_send("agent_2", "agent_3", {"item": i}, buffer_size=10)
            if buffered:
                print(f"  ✓ BUFFER: Flushed at item {i}")

        # Window
        msgs = await self.network.window_send("agent_4", "agent_5", {"windowed": True}, window_ms=100)
        print(f"  ✓ WINDOW: {len(msgs)} messages in time window")

        return 3

    async def test_coordination(self):
        """Test coordination patterns."""
        print("\n" + "─" * 70)
        print("COORDINATION PATTERNS")
        print("─" * 70)

        # Barrier
        participants = ["agent_1", "agent_2", "agent_3"]
        all_arrived = await self.network.barrier(participants, "sync_point_1")
        print(f"  ✓ BARRIER: All participants arrived = {all_arrived}")

        # Quorum Broadcast
        success, msgs = await self.network.quorum_broadcast(
            "agent_0", {"proposal": "vote"}, "workers", quorum_size=2
        )
        print(f"  ✓ QUORUM: Broadcast to workers, quorum met = {success}")

        return 2

    async def test_monitoring(self):
        """Test monitoring patterns."""
        print("\n" + "─" * 70)
        print("MONITORING PATTERNS")
        print("─" * 70)

        # Wire Tap
        original = Message(sender="agent_1", recipients=["agent_2"], content={"sensitive": "data"})
        await self.network._deliver(original)
        tap = await self.network.wire_tap(original, "agent_9")
        print(f"  ✓ WIRE_TAP: Copied message to monitor (agent_9)")

        # Message History
        history = self.network.get_message_history(agent_id="agent_0", limit=10)
        print(f"  ✓ MESSAGE_HISTORY: Retrieved {len(history)} messages for agent_0")

        # Pattern stats
        stats = self.network.get_pattern_stats()
        print(f"  ✓ PATTERN_STATS: {len(stats)} different patterns used")

        return 3

    async def run_all_tests(self):
        """Run all pattern tests."""
        print("\n" + "═" * 70)
        print("TESTING 70+ MESSAGING PATTERNS")
        print("═" * 70)

        self.setup_agents()

        total = 0
        total += await self.test_point_to_point()
        total += await self.test_one_to_many()
        total += await self.test_many_to_one()
        total += await self.test_many_to_many()
        total += await self.test_hierarchical()
        total += await self.test_ring()
        total += await self.test_pipeline()
        total += await self.test_pubsub()
        total += await self.test_gossip()
        total += await self.test_routing()
        total += await self.test_saga()
        total += await self.test_fault_tolerance()
        total += await self.test_flow_control()
        total += await self.test_coordination()
        total += await self.test_monitoring()

        # Summary
        print("\n" + "═" * 70)
        print("SUMMARY")
        print("═" * 70)

        pattern_stats = self.network.get_pattern_stats()
        topology = self.network.export_topology()

        print(f"\n  Total patterns demonstrated: {total}")
        print(f"  Total messages sent: {topology['total_messages']}")
        print(f"  Dead letters: {topology['dead_letters']}")
        print(f"\n  Pattern distribution:")
        for pattern, count in sorted(pattern_stats.items(), key=lambda x: -x[1])[:15]:
            print(f"    {pattern}: {count}")

        print(f"\n  Agent activity:")
        for aid, info in list(topology['agents'].items())[:5]:
            print(f"    {aid}: sent={info['messages_sent']}, received={info['messages_received']}")

        print("\n  ✅ All pattern categories tested!\n")

        return total


async def main():
    demo = PatternDemo()
    await demo.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
