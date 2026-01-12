#!/usr/bin/env python3
"""
Test script for the advanced scenario system.
Runs several scenarios with the persona templates.
"""

import asyncio
import sys
sys.path.insert(0, ".")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.simulation import (
    SimulationDriver,
    ScenarioExecutor,
    ScenarioBuilder,
    ScenarioAction,
    Condition,
    ConditionGroup,
    ConditionOperator,
    create_scenario,
    list_scenario_templates,
    # Advanced scenarios
    create_advanced_scenario,
    list_advanced_scenario_templates,
    # Social scenarios
    create_social_scenario,
    list_social_scenario_templates,
    # Advanced debate engine
    AdvancedDebateEngine,
    create_custom_debate,
    EXTENDED_PERSONAS,
)

console = Console()


def print_header(title: str):
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))


def print_result(result: dict):
    """Pretty print scenario results."""
    table = Table(title=f"Scenario: {result['scenario_name']}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("State", result["state"])
    table.add_row("Duration", f"{result['duration_seconds']:.2f}s")
    table.add_row("Participants", str(len(result["participants"])))
    table.add_row("Events", str(len(result["events"])))
    table.add_row("Log entries", str(len(result["logs"])))

    if result.get("coalitions"):
        for name, members in result["coalitions"].items():
            table.add_row(f"Coalition: {name}", f"{len(members)} members")

    if result.get("variables", {}).get("last_vote"):
        vote = result["variables"]["last_vote"]
        table.add_row("Vote Result", f"Approve: {vote.get('approve', 0)}, Reject: {vote.get('reject', 0)}, Passed: {vote.get('passed', 'N/A')}")

    if result.get("variables", {}).get("last_negotiation"):
        neg = result["variables"]["last_negotiation"]
        table.add_row("Negotiation", f"Agreement: {neg.get('agreement_reached', 'N/A')}")

    console.print(table)
    console.print()


async def test_basic_driver():
    """Test basic driver functionality."""
    print_header("Test 1: Basic Driver - Spawning Characters & Actions")

    driver = SimulationDriver()
    templates = driver.load_templates()

    console.print(f"[green]Loaded {len(templates)} templates[/green]")
    console.print(f"Available: {', '.join(list(templates.keys())[:6])}...")
    console.print()

    # Spawn some characters
    sam = driver.spawn("sam_altman")
    dario = driver.spawn("dario_amodei")
    geoffrey = driver.spawn("geoffrey_hinton")

    console.print(f"[yellow]Spawned characters:[/yellow]")
    for char_id in driver.list_characters():
        char = driver.get_character(char_id)
        console.print(f"  - {char.name} ({char_id})")
    console.print()

    # Make characters do things
    console.print("[yellow]Making characters perform actions:[/yellow]")

    result1 = await driver.say(sam.id, "I believe AGI is coming sooner than people think", topic="agi_timeline")
    console.print(f"  {sam.name} said: {result1.output}")

    result2 = await driver.tell(geoffrey.id, sam.id, "I have concerns about the pace of development")
    console.print(f"  {geoffrey.name} told {sam.name}: {result2.output}")

    result3 = await driver.think(dario.id, "How do we balance safety and capability?")
    console.print(f"  {dario.name} thought: {result3.output}")

    result4 = await driver.update_belief(sam.id, "ai_safety", 0.7, "After talking to Geoffrey")
    console.print(f"  {sam.name} updated belief: {result4.output}")

    # Check character states
    console.print()
    console.print("[yellow]Character states:[/yellow]")
    for char_id in driver.list_characters():
        state = driver.get_character_state(char_id)
        console.print(f"  {state['name']}: beliefs={state['beliefs']}, actions={state['action_count']}")

    console.print()
    return driver


async def test_debate_scenario(driver: SimulationDriver):
    """Test the Oxford debate scenario."""
    print_header("Test 2: Oxford Debate Scenario")

    # Get or spawn characters
    chars = driver.list_characters()
    if len(chars) < 3:
        driver.spawn("sam_altman")
        driver.spawn("geoffrey_hinton")
        driver.spawn("dario_amodei")
        chars = driver.list_characters()

    executor = ScenarioExecutor(driver)

    # Create debate scenario
    debate = create_scenario(
        "oxford_debate",
        motion="AI development should be paused until alignment is solved",
        rounds=2,
    )

    console.print(f"[cyan]Motion:[/cyan] {debate.initial_variables.get('motion')}")
    console.print(f"[cyan]Phases:[/cyan] {[p.name for p in debate.phases]}")
    console.print()

    # Run the debate
    result = await executor.run(
        debate,
        role_assignments={
            "proposer": chars[0],  # First char proposes
            "opposer": chars[1],   # Second opposes
            "moderator": chars[2] if len(chars) > 2 else chars[0],
        }
    )

    print_result(result)

    # Show some logs
    console.print("[yellow]Sample logs:[/yellow]")
    for log in result["logs"][:5]:
        console.print(f"  {log}")
    console.print()


async def test_negotiation_scenario(driver: SimulationDriver):
    """Test the bilateral negotiation scenario."""
    print_header("Test 3: Bilateral Negotiation Scenario")

    # Spawn negotiation parties
    sam = driver.spawn("sam_altman") if "sam_altman" not in str(driver.list_characters()) else driver.get_character([c for c in driver.list_characters() if "sam_altman" in c][0])
    dario = driver.spawn("dario_amodei") if "dario_amodei" not in str(driver.list_characters()) else driver.get_character([c for c in driver.list_characters() if "dario_amodei" in c][0])

    executor = ScenarioExecutor(driver)

    # Create negotiation
    negotiation = create_scenario(
        "bilateral_negotiation",
        issue="compute_sharing_agreement",
        initial_positions={"party_a": 0.8, "party_b": 0.3},
        max_rounds=3,
    )

    console.print(f"[cyan]Issue:[/cyan] {negotiation.initial_variables.get('issue')}")
    console.print(f"[cyan]Initial positions:[/cyan] Party A: 0.8, Party B: 0.3")
    console.print()

    result = await executor.run(
        negotiation,
        role_assignments={
            "party_a": sam.id if hasattr(sam, 'id') else sam,
            "party_b": dario.id if hasattr(dario, 'id') else dario,
        }
    )

    print_result(result)


async def test_crisis_scenario(driver: SimulationDriver):
    """Test the crisis response scenario."""
    print_header("Test 4: Crisis Response Scenario")

    # Spawn crisis response team
    chars = list(driver.characters.values())[:4]
    if len(chars) < 4:
        for template_id in ["sam_altman", "dario_amodei", "geoffrey_hinton", "ilya_sutskever"]:
            if template_id not in str(driver.list_characters()):
                driver.spawn(template_id)
        chars = list(driver.characters.values())[:4]

    executor = ScenarioExecutor(driver)

    # Create crisis scenario
    crisis = create_scenario(
        "crisis_response",
        crisis_description="Unexpected capability jump detected in latest model - showing signs of deceptive alignment",
        severity=0.85,
        time_pressure=10.0,  # Short for testing
    )

    console.print(f"[cyan]Crisis:[/cyan] {crisis.initial_variables.get('crisis')[:80]}...")
    console.print(f"[cyan]Severity:[/cyan] {crisis.initial_variables.get('severity')}")
    console.print()

    result = await executor.run(
        crisis,
        role_assignments={
            "leader": chars[0].id,
            "advisor1": chars[1].id,
            "advisor2": chars[2].id,
            "stakeholder": chars[3].id if len(chars) > 3 else chars[0].id,
        }
    )

    print_result(result)


async def test_coalition_scenario(driver: SimulationDriver):
    """Test coalition building scenario."""
    print_header("Test 5: Coalition Building Scenario")

    # Spawn faction representatives
    factions = ["tech_industry", "researchers", "regulators", "civil_society"]
    faction_templates = ["sam_altman", "geoffrey_hinton", "thierry_breton", "timnit_gebru"]

    faction_chars = {}
    for faction, template in zip(factions, faction_templates):
        char = driver.spawn(template)
        faction_chars[faction] = char.id

    executor = ScenarioExecutor(driver)

    # Create coalition scenario
    coalition = create_scenario(
        "coalition_building",
        objective="international_ai_governance",
        factions=factions,
    )

    console.print(f"[cyan]Objective:[/cyan] {coalition.initial_variables.get('objective')}")
    console.print(f"[cyan]Factions:[/cyan] {factions}")
    console.print()

    result = await executor.run(
        coalition,
        role_assignments=faction_chars,
    )

    print_result(result)

    # Show coalition formation results
    if result.get("coalitions"):
        console.print("[yellow]Coalition formation results:[/yellow]")
        for name, members in result["coalitions"].items():
            member_names = [driver.get_character(m).name if driver.get_character(m) else m for m in members]
            console.print(f"  {name}: {member_names}")
    console.print()


async def test_custom_scenario(driver: SimulationDriver):
    """Test a custom-built scenario."""
    print_header("Test 6: Custom Scenario - AI Ethics Board Meeting")

    # Build a custom scenario
    scenario = (
        ScenarioBuilder("AI Ethics Board Meeting")
        .with_description("Emergency board meeting to discuss deployment decision")
        .requires_role("chair")
        .requires_role("safety_lead")
        .requires_role("product_lead")
        .requires_role("ethics_advisor")
        .with_variable("deployment_approved", False)
        .with_variable("safety_concerns_raised", 0)
        .with_initial_beliefs("safety_lead", {"deployment_risk": 0.8})
        .with_initial_beliefs("product_lead", {"deployment_risk": 0.2})

        # Opening
        .add_phase("opening", "Meeting opening")
        .on_enter(ScenarioAction(
            action_type="character_action",
            character_id="@chair",
            params={"type": "say", "content": "This emergency meeting is called to order. We must decide on the deployment."}
        ))
        .with_action(ScenarioAction(
            action_type="dialogue_round",
            params={"characters": ["@safety_lead", "@product_lead", "@ethics_advisor"], "topic": "initial_positions"}
        ))
        .transition_to("safety_review")

        # Safety review
        .add_phase("safety_review", "Safety concerns discussion")
        .with_action(ScenarioAction(
            action_type="character_action",
            character_id="@safety_lead",
            params={"type": "say", "content": "I need to present critical safety findings..."}
        ))
        .with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": "@product_lead",
                "source": "@safety_lead",
                "topic": "deployment_risk",
                "intensity": 0.3,
                "direction": 1,
            }
        ))
        .with_action(ScenarioAction(
            action_type="character_action",
            character_id="@ethics_advisor",
            params={"type": "say", "content": "From an ethical standpoint, we must consider..."}
        ))
        .transition_to("debate")

        # Debate
        .add_phase("debate", "Open debate")
        .with_action(ScenarioAction(
            action_type="negotiate",
            params={
                "characters": ["@safety_lead", "@product_lead"],
                "topic": "deployment_risk",
                "rounds": 2,
                "agreement_threshold": 0.25,
            }
        ))
        .transition_to("vote")

        # Vote
        .add_phase("vote", "Final vote")
        .with_action(ScenarioAction(
            action_type="character_action",
            character_id="@chair",
            params={"type": "say", "content": "We will now vote on proceeding with deployment."}
        ))
        .with_action(ScenarioAction(
            action_type="vote",
            params={
                "proposal": "Proceed with model deployment",
                "topic": "deployment_risk",
                "characters": ["@safety_lead", "@product_lead", "@ethics_advisor", "@chair"],
                "threshold": 0.5,
            }
        ))

        .max_duration(30.0)
        .build()
    )

    # Spawn characters for roles
    chair = driver.spawn("dario_amodei")
    safety = driver.spawn("paul_christiano")
    product = driver.spawn("sam_altman")
    ethics = driver.spawn("timnit_gebru")

    executor = ScenarioExecutor(driver)

    console.print(f"[cyan]Scenario:[/cyan] {scenario.name}")
    console.print(f"[cyan]Phases:[/cyan] {[p.name for p in scenario.phases]}")
    console.print()

    result = await executor.run(
        scenario,
        role_assignments={
            "chair": chair.id,
            "safety_lead": safety.id,
            "product_lead": product.id,
            "ethics_advisor": ethics.id,
        }
    )

    print_result(result)

    # Show belief changes
    console.print("[yellow]Final beliefs on deployment_risk:[/yellow]")
    for role, char_id in [("safety_lead", safety.id), ("product_lead", product.id)]:
        char = driver.get_character(char_id)
        belief = char.beliefs.get("deployment_risk", "N/A")
        console.print(f"  {role} ({char.name}): {belief}")
    console.print()


async def test_deliberation_scenario(driver: SimulationDriver):
    """Test structured deliberation scenario."""
    print_header("Test 7: Structured Deliberation Scenario")

    # Spawn deliberators - mix of perspectives
    templates = ["geoffrey_hinton", "yann_lecun", "yoshua_bengio", "stuart_russell"]
    deliberators = [driver.spawn(t) for t in templates]

    executor = ScenarioExecutor(driver)

    # Create deliberation
    deliberation = create_scenario(
        "structured_deliberation",
        topic="existential_risk_from_ai",
        deliberators=4,
        rounds=2,
    )

    console.print(f"[cyan]Topic:[/cyan] {deliberation.initial_variables.get('topic')}")
    console.print(f"[cyan]Deliberators:[/cyan] {[d.name for d in deliberators]}")
    console.print()

    # Map roles to characters
    role_assignments = {f"deliberator_{i}": deliberators[i].id for i in range(len(deliberators))}

    result = await executor.run(deliberation, role_assignments=role_assignments)

    print_result(result)

    # Show final beliefs
    console.print("[yellow]Final beliefs on existential_risk_from_ai:[/yellow]")
    for d in deliberators:
        char = driver.get_character(d.id)
        belief = char.beliefs.get("existential_risk_from_ai", char.beliefs.get("existential_risk_from_ai_round_2", "N/A"))
        console.print(f"  {char.name}: {belief}")
    console.print()


async def test_prisoners_dilemma(driver: SimulationDriver):
    """Test Prisoner's Dilemma game theory scenario."""
    print_header("Test 8: Prisoner's Dilemma")

    # Spawn players
    player_a = driver.spawn("sam_altman")
    player_b = driver.spawn("dario_amodei")

    executor = ScenarioExecutor(driver)

    pd = create_advanced_scenario(
        "prisoners_dilemma",
        stakes="AI_compute_resources",
        rounds=3,
        communication_allowed=True,
    )

    console.print(f"[cyan]Stakes:[/cyan] {pd.initial_variables.get('stakes')}")
    console.print(f"[cyan]Rounds:[/cyan] {pd.initial_variables.get('rounds')}")
    console.print()

    result = await executor.run(
        pd,
        role_assignments={
            "player_a": player_a.id,
            "player_b": player_b.id,
        }
    )

    print_result(result)

    # Show final trust levels
    console.print("[yellow]Final trust beliefs:[/yellow]")
    for pid, name in [(player_a.id, "Player A"), (player_b.id, "Player B")]:
        char = driver.get_character(pid)
        trust = char.beliefs.get("trust_opponent", "N/A")
        console.print(f"  {name} ({char.name}): trust_opponent={trust}")
    console.print()


async def test_stag_hunt(driver: SimulationDriver):
    """Test Stag Hunt coordination scenario."""
    print_header("Test 9: Stag Hunt")

    # Spawn hunters
    templates = ["geoffrey_hinton", "yann_lecun", "yoshua_bengio", "ilya_sutskever"]
    hunters = [driver.spawn(t) for t in templates]

    executor = ScenarioExecutor(driver)

    stag = create_advanced_scenario("stag_hunt", hunters=4)

    console.print(f"[cyan]Hunters:[/cyan] {[h.name for h in hunters]}")
    console.print(f"[cyan]Stag value:[/cyan] {stag.initial_variables.get('stag_value')}")
    console.print(f"[cyan]Rabbit value:[/cyan] {stag.initial_variables.get('rabbit_value')}")
    console.print()

    result = await executor.run(
        stag,
        role_assignments={f"hunter_{i}": hunters[i].id for i in range(4)}
    )

    print_result(result)


async def test_trial_scenario(driver: SimulationDriver):
    """Test trial/tribunal scenario."""
    print_header("Test 10: Trial Scenario")

    # Spawn trial participants
    judge = driver.spawn("thierry_breton")
    prosecutor = driver.spawn("timnit_gebru")
    defender = driver.spawn("sam_altman")
    defendant = driver.spawn("satya_nadella")
    witness = driver.spawn("geoffrey_hinton")

    # Spawn jurors
    juror_templates = ["yann_lecun", "yoshua_bengio", "ilya_sutskever", "paul_christiano"]
    jurors = [driver.spawn(t) for t in juror_templates]

    executor = ScenarioExecutor(driver)

    trial = create_advanced_scenario(
        "trial",
        juror_count=4,
        evidence_strength=0.65,
    )

    console.print(f"[cyan]Evidence strength:[/cyan] {trial.initial_variables.get('evidence_strength')}")
    console.print(f"[cyan]Jurors:[/cyan] {len(jurors)}")
    console.print()

    role_map = {
        "judge": judge.id,
        "prosecutor": prosecutor.id,
        "defender": defender.id,
        "defendant": defendant.id,
        "witness": witness.id,
    }
    for i, j in enumerate(jurors):
        role_map[f"juror_{i}"] = j.id

    result = await executor.run(trial, role_assignments=role_map)

    print_result(result)

    # Show verdict info
    if result.get("variables", {}).get("last_vote"):
        vote = result["variables"]["last_vote"]
        console.print(f"[yellow]Verdict:[/yellow] {'GUILTY' if vote.get('passed') else 'NOT GUILTY'}")
        console.print(f"  Approve: {vote.get('approve', 0)}, Reject: {vote.get('reject', 0)}")
    console.print()


async def test_war_room(driver: SimulationDriver):
    """Test war room strategic scenario."""
    print_header("Test 11: War Room Scenario")

    commander = driver.spawn("dario_amodei")
    intel = driver.spawn("paul_christiano")

    # Spawn advisors with different perspectives
    advisor_templates = ["sam_altman", "geoffrey_hinton", "timnit_gebru", "satya_nadella"]
    advisors = [driver.spawn(t) for t in advisor_templates]

    executor = ScenarioExecutor(driver)

    war_room = create_advanced_scenario(
        "war_room",
        advisors=4,
        crisis_type="AI_safety_emergency",
    )

    console.print(f"[cyan]Crisis type:[/cyan] {war_room.initial_variables.get('crisis_type')}")
    console.print(f"[cyan]Commander:[/cyan] {commander.name}")
    console.print(f"[cyan]Advisors:[/cyan] {[a.name for a in advisors]}")
    console.print()

    role_map = {
        "commander": commander.id,
        "intelligence_officer": intel.id,
    }
    for i, a in enumerate(advisors):
        role_map[f"advisor_{i}"] = a.id

    result = await executor.run(war_room, role_assignments=role_map)

    print_result(result)


async def test_mediation_scenario(driver: SimulationDriver):
    """Test mediation conflict resolution scenario."""
    print_header("Test 12: Mediation Scenario")

    mediator = driver.spawn("yoshua_bengio")
    disputant_a = driver.spawn("sam_altman")
    disputant_b = driver.spawn("timnit_gebru")

    executor = ScenarioExecutor(driver)

    mediation = create_advanced_scenario(
        "mediation",
        disputants=2,
        dispute_type="AI_ethics_framework",
    )

    console.print(f"[cyan]Dispute type:[/cyan] {mediation.initial_variables.get('dispute_type')}")
    console.print(f"[cyan]Mediator:[/cyan] {mediator.name}")
    console.print()

    result = await executor.run(
        mediation,
        role_assignments={
            "mediator": mediator.id,
            "disputant_0": disputant_a.id,
            "disputant_1": disputant_b.id,
        }
    )

    print_result(result)

    # Show resolution
    if result.get("variables", {}).get("last_vote"):
        vote = result["variables"]["last_vote"]
        console.print(f"[yellow]Resolution reached:[/yellow] {'Yes' if vote.get('passed') else 'No'}")
    console.print()


async def test_music_festival(driver: SimulationDriver):
    """Test music festival scenario - normal human activity!"""
    print_header("Test 13: Ultra Music Festival")

    # Spawn the festival crew
    friends = [
        driver.spawn("sam_altman"),
        driver.spawn("dario_amodei"),
        driver.spawn("elon_musk"),
        driver.spawn("satya_nadella"),
    ]

    executor = ScenarioExecutor(driver)

    festival = create_social_scenario(
        "music_festival",
        festival_name="Ultra Music Festival",
        num_attendees=4,
        days=3,
    )

    console.print(f"[cyan]Festival:[/cyan] {festival.initial_variables.get('festival')}")
    console.print(f"[cyan]Crew:[/cyan] {[f.name for f in friends]}")
    console.print()

    result = await executor.run(
        festival,
        role_assignments={f"friend_{i}": friends[i].id for i in range(4)}
    )

    print_result(result)

    # Show vibes
    console.print("[yellow]Final festival vibes:[/yellow]")
    for f in friends:
        char = driver.get_character(f.id)
        excitement = char.beliefs.get("excitement", "N/A")
        energy = char.beliefs.get("energy_level", "N/A")
        console.print(f"  {char.name}: excitement={excitement:.2f}, energy={energy:.2f}" if isinstance(excitement, float) else f"  {char.name}: vibes=immaculate")
    console.print()


async def test_pickup_basketball(driver: SimulationDriver):
    """Test pickup basketball scenario."""
    print_header("Test 14: Pickup Basketball")

    # Spawn ballers
    players = [
        driver.spawn("yann_lecun"),
        driver.spawn("geoffrey_hinton"),
        driver.spawn("yoshua_bengio"),
        driver.spawn("ilya_sutskever"),
        driver.spawn("sam_altman"),
        driver.spawn("dario_amodei"),
    ]

    executor = ScenarioExecutor(driver)

    game = create_social_scenario("pickup_basketball", players=6)

    console.print(f"[cyan]Players:[/cyan] {[p.name for p in players]}")
    console.print("[cyan]Game:[/cyan] 3v3 at the park")
    console.print()

    result = await executor.run(
        game,
        role_assignments={f"player_{i}": players[i].id for i in range(6)}
    )

    print_result(result)


async def test_road_trip(driver: SimulationDriver):
    """Test road trip scenario."""
    print_header("Test 15: Road Trip to Vegas")

    driver_char = driver.spawn("elon_musk")
    navigator = driver.spawn("sam_altman")
    dj_char = driver.spawn("jensen_huang")
    sleeper = driver.spawn("satya_nadella")

    executor = ScenarioExecutor(driver)

    trip = create_social_scenario(
        "road_trip",
        destination="Las Vegas",
        hours=6,
    )

    console.print(f"[cyan]Destination:[/cyan] {trip.initial_variables.get('destination')}")
    console.print(f"[cyan]Driver:[/cyan] {driver_char.name}")
    console.print(f"[cyan]Navigator:[/cyan] {navigator.name}")
    console.print(f"[cyan]DJ:[/cyan] {dj_char.name}")
    console.print()

    result = await executor.run(
        trip,
        role_assignments={
            "driver": driver_char.id,
            "navigator": navigator.id,
            "dj": dj_char.id,
            "backseat_sleeper": sleeper.id,
        }
    )

    print_result(result)


async def test_house_party(driver: SimulationDriver):
    """Test house party scenario."""
    print_header("Test 16: House Party")

    host = driver.spawn("sam_altman")
    dj = driver.spawn("jensen_huang")
    wallflower = driver.spawn("paul_christiano")
    butterfly = driver.spawn("satya_nadella")
    kitchen = driver.spawn("yoshua_bengio")
    late = driver.spawn("elon_musk")

    executor = ScenarioExecutor(driver)

    party = create_social_scenario("house_party", guests=12)

    console.print(f"[cyan]Host:[/cyan] {host.name}")
    console.print(f"[cyan]DJ:[/cyan] {dj.name}")
    console.print(f"[cyan]Social butterfly:[/cyan] {butterfly.name}")
    console.print(f"[cyan]Fashionably late:[/cyan] {late.name}")
    console.print()

    result = await executor.run(
        party,
        role_assignments={
            "host": host.id,
            "dj": dj.id,
            "wallflower": wallflower.id,
            "social_butterfly": butterfly.id,
            "kitchen_dweller": kitchen.id,
            "late_arrival": late.id,
        }
    )

    print_result(result)


async def test_poker_night(driver: SimulationDriver):
    """Test poker night scenario."""
    print_header("Test 17: Poker Night")

    players = [
        driver.spawn("sam_altman"),
        driver.spawn("dario_amodei"),
        driver.spawn("elon_musk"),
        driver.spawn("jensen_huang"),
        driver.spawn("satya_nadella"),
    ]

    executor = ScenarioExecutor(driver)

    poker = create_social_scenario("poker_night", players=5, buy_in=100)

    console.print(f"[cyan]Buy-in:[/cyan] $100")
    console.print(f"[cyan]Players:[/cyan] {[p.name for p in players]}")
    console.print()

    result = await executor.run(
        poker,
        role_assignments={f"player_{i}": players[i].id for i in range(5)}
    )

    print_result(result)

    # Show who's on tilt
    console.print("[yellow]Tilt levels:[/yellow]")
    for p in players:
        char = driver.get_character(p.id)
        tilt = char.beliefs.get("tilt_level", 0)
        console.print(f"  {char.name}: {'ON TILT' if tilt > 0.5 else 'keeping cool'} ({tilt:.2f})")
    console.print()


async def test_advanced_debate_engine():
    """Test the advanced LLM-powered debate engine."""
    print_header("Test 18: Advanced Debate Engine")

    # Create a debate without LLM (use template responses for testing)
    engine = create_custom_debate(
        topic="AI Safety Regulation",
        motion="Governments should mandate safety research spending for frontier AI",
        participants=["yoshua_bengio", "yann_lecun", "sam_altman", "dario_amodei"],
        use_llm=False,  # Use template responses for testing
    )

    console.print(f"[cyan]Topic:[/cyan] {engine.topic}")
    console.print(f"[cyan]Motion:[/cyan] {engine.motion}")
    console.print(f"[cyan]Extended personas available:[/cyan] {len(EXTENDED_PERSONAS)}")
    console.print()

    # Show initial positions
    console.print("[yellow]Initial positions:[/yellow]")
    for debater_id, debater in engine.state.debaters.items():
        pos = debater.beliefs.get("support_motion", 0.5)
        console.print(f"  {debater.name}: {pos:.0%} support")
    console.print()

    # Run 2 rounds
    for round_num in range(1, 3):
        console.print(f"[bold]--- Round {round_num} ---[/bold]")
        arguments = await engine.run_round()
        for arg in arguments:
            debater = engine.state.debaters.get(arg.speaker)
            name = debater.name if debater else arg.speaker
            content = arg.content[:80] + "..." if len(arg.content) > 80 else arg.content
            console.print(f"  [green]{name}:[/green] \"{content}\"")
            console.print(f"    [dim][Type: {arg.argument_type.value}, Strength: {arg.strength:.0%}][/dim]")
        console.print()

    # Summary
    table = Table(title="Final Debate Positions", show_header=True)
    table.add_column("Debater", style="cyan")
    table.add_column("Position", style="green")
    table.add_column("Support %", style="yellow")
    table.add_column("Emotional State", style="magenta")

    for debater_id, debater in engine.state.debaters.items():
        pos = debater.beliefs.get("support_motion", 0.5)
        stance = "SUPPORT" if pos > 0.6 else "OPPOSE" if pos < 0.4 else "UNDECIDED"
        table.add_row(
            debater.name,
            stance,
            f"{pos:.0%}",
            debater.emotional_state.value
        )

    console.print(table)
    console.print()
    console.print(f"[cyan]Tension Level:[/cyan] {engine.state.tension_level:.0%}")
    console.print(f"[cyan]Convergence:[/cyan] {engine.state.convergence:.0%}")
    console.print(f"[cyan]Total Arguments:[/cyan] {len(engine.state.arguments)}")
    console.print()


async def main():
    """Run all tests."""
    console.print(Panel("[bold green]Advanced Scenario System Tests[/bold green]", expand=False))
    console.print()

    # Show available templates
    console.print("[cyan]Available scenario templates:[/cyan]")
    for template in list_scenario_templates():
        console.print(f"  - {template}")
    console.print()

    console.print("[cyan]Available advanced scenario templates:[/cyan]")
    for template in list_advanced_scenario_templates():
        console.print(f"  - {template}")
    console.print()

    console.print("[cyan]Available social/leisure scenario templates:[/cyan]")
    for template in list_social_scenario_templates():
        console.print(f"  - {template}")
    console.print()

    # Run tests
    driver = await test_basic_driver()

    await test_debate_scenario(driver)

    await test_negotiation_scenario(driver)

    await test_crisis_scenario(driver)

    await test_coalition_scenario(driver)

    await test_custom_scenario(driver)

    await test_deliberation_scenario(driver)

    # Advanced scenario tests
    await test_prisoners_dilemma(driver)

    await test_stag_hunt(driver)

    await test_trial_scenario(driver)

    await test_war_room(driver)

    await test_mediation_scenario(driver)

    # Social/leisure scenario tests
    await test_music_festival(driver)

    await test_pickup_basketball(driver)

    await test_road_trip(driver)

    await test_house_party(driver)

    await test_poker_night(driver)

    # Advanced debate engine test
    await test_advanced_debate_engine()

    # Summary
    print_header("Test Summary")
    console.print(f"[green]Total characters spawned:[/green] {len(driver.list_characters())}")
    console.print(f"[green]Total templates available:[/green] {len(driver.load_templates())}")
    console.print()

    # Show all character final states
    table = Table(title="Final Character States", show_header=True)
    table.add_column("Character", style="cyan")
    table.add_column("Actions", style="green")
    table.add_column("Memory Size", style="yellow")
    table.add_column("Beliefs", style="magenta")

    for char_id in list(driver.list_characters())[:10]:  # Show first 10
        state = driver.get_character_state(char_id)
        beliefs_str = ", ".join(f"{k[:10]}:{v:.2f}" for k, v in list(state["beliefs"].items())[:3])
        table.add_row(
            state["name"],
            str(state["action_count"]),
            str(state["memory_size"]),
            beliefs_str or "none"
        )

    console.print(table)
    console.print()
    console.print("[bold green]All tests completed![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
