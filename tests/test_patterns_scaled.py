#!/usr/bin/env python3
"""
Test 70+ messaging patterns at scale with larger agent groups.
Scales from 15 → 50 agents across multiple configurations.
"""

import sys
import asyncio
import time
from typing import Dict, List, Any

sys.path.insert(0, "/Users/arthurcolle/lida-multiagents-research")

from src.meta.messaging_patterns import (
    PatternCategory,
    MessagePattern,
    Message,
    MessagingNetwork,
)


class ScaledPatternDemo:
    """Demonstrates patterns at scale with larger groups."""

    def __init__(self, num_agents: int):
        self.network = MessagingNetwork()
        self.num_agents = num_agents
        self.stats: Dict[str, Any] = {
            "messages_sent": 0,
            "patterns_used": set(),
            "timing": {},
        }

    def setup_scaled_network(self):
        """Set up a larger network with multiple group configurations."""
        print(f"\n  Setting up {self.num_agents} agents...")

        # Register all agents with locations spread across a grid
        grid_size = int(self.num_agents ** 0.5) + 1
        for i in range(self.num_agents):
            x = (i % grid_size) * 10.0
            y = (i // grid_size) * 10.0
            self.network.register_agent(f"agent_{i}", location=(x, y))

        # Create hierarchical groups
        # Leadership (top 5%)
        leader_count = max(2, self.num_agents // 20)
        for i in range(leader_count):
            self.network.add_to_group(f"agent_{i}", "leaders")

        # Managers (next 15%)
        manager_count = max(3, self.num_agents // 7)
        for i in range(leader_count, leader_count + manager_count):
            self.network.add_to_group(f"agent_{i}", "managers")

        # Workers (60%)
        worker_start = leader_count + manager_count
        worker_count = int(self.num_agents * 0.6)
        for i in range(worker_start, worker_start + worker_count):
            self.network.add_to_group(f"agent_{i}", "workers")

        # Observers (remaining)
        for i in range(worker_start + worker_count, self.num_agents):
            self.network.add_to_group(f"agent_{i}", "observers")

        # Create functional teams (overlapping with hierarchy)
        team_size = max(4, self.num_agents // 8)
        teams = ["alpha", "beta", "gamma", "delta", "epsilon"]
        for t_idx, team in enumerate(teams):
            start = (t_idx * team_size) % self.num_agents
            for i in range(team_size):
                agent_idx = (start + i) % self.num_agents
                self.network.add_to_group(f"agent_{agent_idx}", f"team_{team}")

        # Create regional groups based on location
        regions = [
            ("north", lambda x, y: y > 30),
            ("south", lambda x, y: y <= 30),
            ("east", lambda x, y: x > 30),
            ("west", lambda x, y: x <= 30),
            ("central", lambda x, y: 20 <= x <= 40 and 20 <= y <= 40),
        ]
        for region_name, region_filter in regions:
            for aid, agent in self.network.agents.items():
                if region_filter(agent.location[0], agent.location[1]):
                    self.network.add_to_group(aid, f"region_{region_name}")

        # Set up topics
        topics = ["events", "alerts", "metrics", "commands", "logs"]
        for topic in topics:
            # Subscribe subset of agents to each topic
            for i in range(self.num_agents):
                if hash(f"{topic}_{i}") % 3 == 0:  # ~33% subscribe to each
                    self.network.subscribe(f"agent_{i}", topic)

        # Set up tree structure (balanced)
        self.network.set_tree_root("agent_0")
        self._build_tree(0, 0, min(4, self.num_agents // 10 + 1))

        # Set up mesh neighbors (each agent connected to 3-5 neighbors)
        for i in range(self.num_agents):
            neighbors = []
            for j in range(1, 6):
                neighbor_idx = (i + j) % self.num_agents
                if neighbor_idx != i:
                    neighbors.append(f"agent_{neighbor_idx}")
            self.network.set_neighbors(f"agent_{i}", neighbors[:4])

        # Print setup summary
        print(f"  Groups: {len(self.network.groups)}")
        for gname, members in self.network.groups.items():
            print(f"    {gname}: {len(members)} members")
        print(f"  Topics: {len(self.network.topics)}")
        for tname, subs in self.network.topics.items():
            print(f"    {tname}: {len(subs)} subscribers")

    def _build_tree(self, parent_idx: int, depth: int, max_children: int):
        """Recursively build tree structure."""
        if depth > 4:  # Max depth
            return

        parent = f"agent_{parent_idx}"
        children_added = 0
        child_idx = parent_idx + 1

        while children_added < max_children and child_idx < self.num_agents:
            child = f"agent_{child_idx}"
            if child not in [f"agent_{i}" for i in range(parent_idx + 1)]:
                self.network.agents[parent].children.add(child)
                self.network.agents[child].parent = parent
                self._build_tree(child_idx, depth + 1, max(2, max_children - 1))
                children_added += 1
            child_idx += 1

    async def timed_test(self, name: str, coro):
        """Run a test with timing."""
        start = time.perf_counter()
        result = await coro
        elapsed = time.perf_counter() - start
        self.stats["timing"][name] = elapsed
        return result, elapsed

    async def test_broadcast_patterns(self):
        """Test broadcast patterns at scale."""
        print("\n" + "─" * 70)
        print(f"BROADCAST PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Full broadcast
        msgs, elapsed = await self.timed_test(
            "broadcast",
            self.network.broadcast("agent_0", {"type": "announcement"})
        )
        print(f"  ✓ BROADCAST: 1 → {len(msgs)} agents ({elapsed*1000:.1f}ms)")

        # Multicast to each group
        for group in ["leaders", "managers", "workers", "team_alpha"]:
            if group in self.network.groups:
                msgs, elapsed = await self.timed_test(
                    f"multicast_{group}",
                    self.network.multicast("agent_0", group, {"directive": "execute"})
                )
                print(f"  ✓ MULTICAST: 1 → {group} ({len(msgs)} members, {elapsed*1000:.1f}ms)")

        # Geocast to regions
        msgs, elapsed = await self.timed_test(
            "geocast",
            self.network.geocast("agent_0", {"alert": "local"}, center=(25.0, 25.0), radius=20.0)
        )
        print(f"  ✓ GEOCAST: 1 → {len(msgs)} agents in radius ({elapsed*1000:.1f}ms)")

        return len(msgs)

    async def test_scatter_gather_patterns(self):
        """Test scatter-gather at scale."""
        print("\n" + "─" * 70)
        print(f"SCATTER-GATHER PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Scatter to workers
        workers = list(self.network.groups.get("workers", []))[:10]
        if workers:
            contents = [{"partition": i} for i in range(len(workers))]
            msgs, elapsed = await self.timed_test(
                "scatter",
                self.network.scatter("agent_0", workers, contents)
            )
            print(f"  ✓ SCATTER: 1 → {len(msgs)} workers with unique content ({elapsed*1000:.1f}ms)")

        # Scatter-gather
        targets = [f"agent_{i}" for i in range(1, min(15, self.num_agents))]
        responses, elapsed = await self.timed_test(
            "scatter_gather",
            self.network.scatter_gather("agent_0", targets, {"query": "status"}, timeout=0.5)
        )
        print(f"  ✓ SCATTER_GATHER: 1 → {len(targets)} → collected {len(responses)} ({elapsed*1000:.1f}ms)")

        # Gather from managers
        managers = list(self.network.groups.get("managers", []))[:8]
        if managers:
            responses, elapsed = await self.timed_test(
                "gather",
                self.network.gather("agent_0", managers, timeout=0.5)
            )
            print(f"  ✓ GATHER: Collected from {len(managers)} managers ({elapsed*1000:.1f}ms)")

        return len(targets)

    async def test_hierarchical_patterns(self):
        """Test hierarchical patterns at scale."""
        print("\n" + "─" * 70)
        print(f"HIERARCHICAL PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Tree broadcast from root
        msgs, elapsed = await self.timed_test(
            "tree_broadcast",
            self.network.tree_broadcast("agent_0", {"cascade": "down"})
        )
        print(f"  ✓ TREE_BROADCAST: Root → {len(msgs)} descendants ({elapsed*1000:.1f}ms)")

        # Leader to all followers
        leader = list(self.network.groups.get("leaders", ["agent_0"]))[0]
        msgs, elapsed = await self.timed_test(
            "leader_broadcast",
            self.network.leader_broadcast(leader, {"command": "follow"})
        )
        print(f"  ✓ LEADER_FOLLOWER: {leader} → {len(msgs)} followers ({elapsed*1000:.1f}ms)")

        return len(msgs)

    async def test_ring_patterns(self):
        """Test ring patterns at scale."""
        print("\n" + "─" * 70)
        print(f"RING PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Ring pass (limit to prevent too many messages)
        msgs, elapsed = await self.timed_test(
            "ring_pass",
            self.network.ring_pass("agent_0", {"token": "circulating"})
        )
        print(f"  ✓ RING_PASS: Circulated through {len(msgs)} nodes ({elapsed*1000:.1f}ms)")

        # Token ring - multiple acquisitions
        acquired_count = 0
        for i in range(5):
            if await self.network.token_ring_acquire(f"agent_{i}", timeout=0.2):
                acquired_count += 1
                self.network.token_ring_release(f"agent_{i}")
        print(f"  ✓ TOKEN_RING: {acquired_count}/5 sequential acquisitions")

        return len(msgs)

    async def test_pubsub_patterns(self):
        """Test pub/sub patterns at scale."""
        print("\n" + "─" * 70)
        print(f"PUB/SUB PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        total_delivered = 0
        for topic in ["events", "alerts", "metrics"]:
            if topic in self.network.topics:
                msgs, elapsed = await self.timed_test(
                    f"publish_{topic}",
                    self.network.publish("agent_0", topic, {"data": f"{topic}_update"})
                )
                print(f"  ✓ PUBLISH: '{topic}' → {len(msgs)} subscribers ({elapsed*1000:.1f}ms)")
                total_delivered += len(msgs)

        # Content filtered publish
        msgs, elapsed = await self.timed_test(
            "content_filter",
            self.network.content_filtered_publish(
                "agent_0", "events",
                {"priority": "high"},
                lambda sub, content: int(sub.split("_")[1]) < self.num_agents // 2
            )
        )
        print(f"  ✓ CONTENT_FILTER: Filtered → {len(msgs)} matching ({elapsed*1000:.1f}ms)")

        return total_delivered

    async def test_gossip_patterns(self):
        """Test gossip/epidemic patterns at scale."""
        print("\n" + "─" * 70)
        print(f"GOSSIP/EPIDEMIC PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Gossip push with higher fanout for larger networks
        fanout = min(5, max(2, self.num_agents // 10))
        rounds = min(6, max(3, int((self.num_agents ** 0.5))))

        msgs, elapsed = await self.timed_test(
            "gossip_push",
            self.network.gossip_push("agent_0", {"rumor": "spread"}, fanout=fanout, rounds=rounds)
        )
        print(f"  ✓ GOSSIP_PUSH: Spread to {len(msgs)} nodes (fanout={fanout}, rounds={rounds}, {elapsed*1000:.1f}ms)")

        # Epidemic broadcast
        msgs, elapsed = await self.timed_test(
            "epidemic",
            self.network.epidemic_broadcast("agent_5", {"virus": "data"}, infection_prob=0.4)
        )
        print(f"  ✓ EPIDEMIC_BROADCAST: Infected {len(msgs)} nodes ({elapsed*1000:.1f}ms)")

        return len(msgs)

    async def test_pipeline_patterns(self):
        """Test pipeline patterns at scale."""
        print("\n" + "─" * 70)
        print(f"PIPELINE PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Long pipeline
        pipeline_length = min(10, self.num_agents // 3)
        stages = [f"agent_{i}" for i in range(1, pipeline_length + 1)]
        msg, elapsed = await self.timed_test(
            "pipeline",
            self.network.pipeline({"data": "process"}, stages)
        )
        print(f"  ✓ PIPELINE: {pipeline_length} stages ({elapsed*1000:.1f}ms)")

        # Parallel pipeline with multiple agents per stage
        stage_groups = []
        agents_per_stage = max(2, self.num_agents // 15)
        for s in range(3):
            stage_agents = [f"agent_{s * agents_per_stage + i}" for i in range(agents_per_stage)
                          if s * agents_per_stage + i < self.num_agents]
            if stage_agents:
                stage_groups.append(stage_agents)

        if stage_groups:
            msgs, elapsed = await self.timed_test(
                "parallel_pipeline",
                self.network.parallel_pipeline({"batch": "data"}, stage_groups)
            )
            print(f"  ✓ PARALLEL_PIPELINE: {len(stage_groups)} stages × {agents_per_stage} agents = {len(msgs)} msgs ({elapsed*1000:.1f}ms)")

        return pipeline_length

    async def test_saga_patterns(self):
        """Test saga patterns at scale."""
        print("\n" + "─" * 70)
        print(f"SAGA PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Multi-step saga with compensations
        saga_steps = min(8, self.num_agents // 4)
        steps = [(f"agent_{i}", {"action": f"step_{i}"}) for i in range(1, saga_steps + 1)]
        compensations = {i: (f"agent_{i+1}", {"action": f"undo_step_{i}"}) for i in range(saga_steps - 1)}

        saga_id = await self.network.start_saga("large_saga", steps, compensations)
        print(f"  ✓ SAGA_ORCHESTRATION: Started with {saga_steps} steps")

        executed = 0
        for _ in range(saga_steps):
            msg = await self.network.execute_saga_step(saga_id)
            if msg:
                executed += 1
        print(f"  ✓ SAGA: Executed {executed}/{saga_steps} steps")

        # Compensation
        comp_msgs = await self.network.compensate_saga(saga_id)
        print(f"  ✓ COMPENSATING_TX: {len(comp_msgs)} compensations executed")

        return executed

    async def test_fault_tolerance_patterns(self):
        """Test fault tolerance patterns at scale."""
        print("\n" + "─" * 70)
        print(f"FAULT TOLERANCE PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Circuit breaker across multiple agents
        success_count = 0
        for i in range(10):
            target = f"agent_{(i * 3) % self.num_agents}"
            msg = await self.network.send_with_circuit_breaker(
                "agent_0", target, {"test": i}
            )
            if msg:
                success_count += 1
        print(f"  ✓ CIRCUIT_BREAKER: {success_count}/10 messages delivered")

        # Retry pattern
        retry_results = 0
        for i in range(5):
            msg = await self.network.send_with_retry(
                "agent_0", f"agent_{i + 1}", {"reliable": True}, max_retries=2
            )
            if msg:
                retry_results += 1
        print(f"  ✓ RETRY: {retry_results}/5 delivered with retry")

        # Timeout pattern
        timeout_results = 0
        for i in range(5):
            msg = await self.network.send_with_timeout(
                "agent_0", f"agent_{i + 5}", {"urgent": True}, timeout=1.0
            )
            if msg:
                timeout_results += 1
        print(f"  ✓ TIMEOUT: {timeout_results}/5 delivered within timeout")

        return success_count + retry_results + timeout_results

    async def test_flow_control_patterns(self):
        """Test flow control patterns at scale."""
        print("\n" + "─" * 70)
        print(f"FLOW CONTROL PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Throttle - burst of messages
        throttle_passed = 0
        for i in range(20):
            msg = await self.network.throttle(
                "agent_0", "agent_1", {"burst": i}, rate=5.0
            )
            if msg:
                throttle_passed += 1
        print(f"  ✓ THROTTLE: {throttle_passed}/20 messages passed rate limit")

        # Buffer - collect and flush
        buffer_flushes = 0
        for i in range(25):
            msg = await self.network.buffer_send(
                "agent_2", "agent_3", {"item": i}, buffer_size=8
            )
            if msg:
                buffer_flushes += 1
        print(f"  ✓ BUFFER: {buffer_flushes} buffer flushes (8-item batches)")

        return throttle_passed

    async def test_coordination_patterns(self):
        """Test coordination patterns at scale."""
        print("\n" + "─" * 70)
        print(f"COORDINATION PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Barrier with subset of agents
        barrier_size = min(10, self.num_agents // 3)
        participants = [f"agent_{i}" for i in range(barrier_size)]
        all_arrived = await self.network.barrier(participants, "sync_all")
        print(f"  ✓ BARRIER: {barrier_size} participants synchronized = {all_arrived}")

        # Quorum broadcast to workers
        if "workers" in self.network.groups:
            worker_count = len(self.network.groups["workers"])
            quorum_size = max(2, worker_count // 2)
            success, msgs = await self.network.quorum_broadcast(
                "agent_0", {"proposal": "vote"}, "workers", quorum_size=quorum_size
            )
            print(f"  ✓ QUORUM: Broadcast to workers, quorum {quorum_size}/{worker_count} met = {success}")

        return barrier_size

    async def test_relay_chains(self):
        """Test relay chain patterns at scale."""
        print("\n" + "─" * 70)
        print(f"RELAY CHAIN PATTERNS ({self.num_agents} agents)")
        print("─" * 70)

        # Short relay through leaders
        leaders = list(self.network.groups.get("leaders", []))
        if len(leaders) >= 2:
            msgs, elapsed = await self.timed_test(
                "relay_leaders",
                self.network.relay_chain("agent_0", {"priority": "high"}, leaders)
            )
            print(f"  ✓ RELAY_CHAIN (leaders): {len(msgs)} hops ({elapsed*1000:.1f}ms)")

        # Long relay chain
        chain_length = min(15, self.num_agents // 2)
        chain = [f"agent_{i}" for i in range(1, chain_length + 1)]
        msgs, elapsed = await self.timed_test(
            "relay_long",
            self.network.relay_chain("agent_0", {"message": "pass_along"}, chain)
        )
        print(f"  ✓ RELAY_CHAIN (long): {len(msgs)} hops ({elapsed*1000:.1f}ms)")

        # Cross-team relay
        teams = ["team_alpha", "team_beta", "team_gamma"]
        cross_team_chain = []
        for team in teams:
            if team in self.network.groups:
                members = list(self.network.groups[team])
                if members:
                    cross_team_chain.append(members[0])

        if len(cross_team_chain) >= 2:
            msgs, elapsed = await self.timed_test(
                "relay_cross_team",
                self.network.relay_chain("agent_0", {"cross_team": True}, cross_team_chain)
            )
            print(f"  ✓ RELAY_CHAIN (cross-team): {len(msgs)} hops ({elapsed*1000:.1f}ms)")

        return chain_length

    async def run_all_tests(self):
        """Run all pattern tests at this scale."""
        print("\n" + "═" * 70)
        print(f"TESTING PATTERNS AT SCALE: {self.num_agents} AGENTS")
        print("═" * 70)

        self.setup_scaled_network()

        total_ops = 0
        total_ops += await self.test_broadcast_patterns()
        total_ops += await self.test_scatter_gather_patterns()
        total_ops += await self.test_hierarchical_patterns()
        total_ops += await self.test_ring_patterns()
        total_ops += await self.test_pubsub_patterns()
        total_ops += await self.test_gossip_patterns()
        total_ops += await self.test_pipeline_patterns()
        total_ops += await self.test_saga_patterns()
        total_ops += await self.test_fault_tolerance_patterns()
        total_ops += await self.test_flow_control_patterns()
        total_ops += await self.test_coordination_patterns()
        total_ops += await self.test_relay_chains()

        # Summary
        print("\n" + "─" * 70)
        print("SCALE SUMMARY")
        print("─" * 70)

        topology = self.network.export_topology()
        pattern_stats = self.network.get_pattern_stats()

        print(f"  Agents: {self.num_agents}")
        print(f"  Groups: {len(self.network.groups)}")
        print(f"  Topics: {len(self.network.topics)}")
        print(f"  Total messages: {topology['total_messages']}")
        print(f"  Dead letters: {topology['dead_letters']}")
        print(f"  Patterns used: {len(pattern_stats)}")

        # Top patterns by usage
        print(f"\n  Top patterns:")
        for pattern, count in sorted(pattern_stats.items(), key=lambda x: -x[1])[:10]:
            print(f"    {pattern}: {count}")

        # Timing summary
        print(f"\n  Timing (slowest operations):")
        for op, time_sec in sorted(self.stats["timing"].items(), key=lambda x: -x[1])[:8]:
            print(f"    {op}: {time_sec*1000:.1f}ms")

        return topology['total_messages']


async def main():
    """Run scaled tests at multiple sizes."""
    print("\n" + "🔬" * 35)
    print("   MESSAGING PATTERNS AT SCALE")
    print("   15 → 25 → 40 → 50 Agents")
    print("🔬" * 35)

    scales = [15, 25, 40, 50]
    results = {}

    for num_agents in scales:
        demo = ScaledPatternDemo(num_agents)
        msg_count = await demo.run_all_tests()
        results[num_agents] = msg_count
        print(f"\n  ✅ {num_agents} agents: {msg_count} total messages\n")

    # Final comparison
    print("\n" + "═" * 70)
    print("FINAL COMPARISON")
    print("═" * 70)
    print(f"\n  Scale progression:")
    for agents, messages in results.items():
        bar = "█" * (messages // 50)
        print(f"    {agents:3d} agents: {messages:5d} messages {bar}")

    print("\n  ✅ All scale tests complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
