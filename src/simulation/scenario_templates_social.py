"""
Social/Leisure Scenario Templates

Normal human activities and social situations:
- Festivals and concerts
- Sports and games
- Dining and parties
- Travel and adventures
- Everyday life situations
"""

from __future__ import annotations

import random
from typing import List, Any

from .scenarios import (
    ScenarioDefinition,
    ScenarioBuilder,
    ScenarioAction,
    Condition,
    ConditionGroup,
    ConditionOperator,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _action(action_type: str, **params) -> ScenarioAction:
    return ScenarioAction(action_type=action_type, params=params)


def _char_action(char_id: str, action_type: str, content: str, **kwargs) -> ScenarioAction:
    return ScenarioAction(
        action_type="character_action",
        character_id=char_id,
        params={"type": action_type, "content": content, **kwargs},
    )


# =============================================================================
# FESTIVALS & CONCERTS
# =============================================================================

def create_music_festival(
    festival_name: str = "Ultra Music Festival",
    num_attendees: int = 4,
    days: int = 3,
) -> ScenarioDefinition:
    """Create a music festival scenario.

    Friends attend a multi-day festival together.
    Tests group dynamics, decision-making, and having fun.
    """
    builder = ScenarioBuilder(f"{festival_name} Weekend")
    builder.with_description(f"{num_attendees} friends at {festival_name}")

    for i in range(num_attendees):
        builder.requires_role(f"friend_{i}")
        builder.with_initial_beliefs(f"friend_{i}", {
            "energy_level": 0.9,
            "excitement": 0.8 + 0.2 * random.random(),
            "hydration": 1.0,
            "group_cohesion": 0.7,
        })

    builder.with_variable("festival", festival_name)
    builder.with_variable("day", 1)
    builder.with_variable("memorable_moments", [])
    builder.with_variable("stages_visited", [])

    # Arrival
    builder.add_phase("arrival", "Arriving at the festival")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": f"Welcome to {festival_name}! The bass is already dropping!", "topic": "festival"}
    ))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@friend_{i}" for i in range(num_attendees)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "LET'S GOOO! This is going to be amazing!"}
            }
        }
    ))
    builder.transition_to("mainstage")

    # Main stage experience
    builder.add_phase("mainstage", "Main stage headliner")
    builder.with_action(_char_action("@friend_0", "say", "The main stage is INSANE! Can you feel that bass?"))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@friend_{i}" for i in range(num_attendees)], "topic": "music"}
    ))
    # Everyone's excitement builds
    for i in range(num_attendees):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@friend_{i}",
                "source": f"@friend_{(i+1) % num_attendees}",
                "topic": "excitement",
                "intensity": 0.15,
                "direction": 1,
            }
        ))
    builder.transition_to("split_decision")

    # Group splits up
    builder.add_phase("split_decision", "Where to next?")
    builder.with_action(_char_action("@friend_0", "say", "Should we check out the techno stage or get food?"))
    builder.with_action(_char_action("@friend_1", "say", "I'm vibing here but also starving..."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Stay at main stage",
            "topic": "excitement",
            "characters": [f"@friend_{i}" for i in range(num_attendees)],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("food_break")

    # Food and hydration
    builder.add_phase("food_break", "Festival food experience")
    builder.with_action(_char_action("@friend_0", "say", "$15 for a slice of pizza?! Festival prices are wild."))
    builder.with_action(_char_action("@friend_1", "say", "But have you tried the tacos? Life changing."))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@friend_{i}" for i in range(num_attendees)],
            "action": {
                "action_type": "apply_pressure",
                "params": {
                    "target": "$current_character",
                    "source": "$current_character",
                    "topic": "hydration",
                    "intensity": 0.3,
                    "direction": 1,
                }
            }
        }
    ))
    builder.transition_to("sunset_set")

    # Sunset set - peak experience
    builder.add_phase("sunset_set", "Sunset set magic")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "The sun is setting and the DJ just dropped that track everyone was waiting for...", "topic": "music"}
    ))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@friend_{i}" for i in range(num_attendees)], "topic": "peak_experience"}
    ))
    # Group cohesion peaks
    for i in range(num_attendees):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@friend_{i}",
                "source": f"@friend_{(i+1) % num_attendees}",
                "topic": "group_cohesion",
                "intensity": 0.2,
                "direction": 1,
            }
        ))
    builder.transition_to("late_night")

    # Late night
    builder.add_phase("late_night", "Late night adventures")
    builder.with_action(_char_action("@friend_0", "say", "It's 3 AM and I've never felt more alive!"))
    builder.with_action(_char_action("@friend_1", "say", "My feet hurt but I can't stop dancing!"))
    # Energy drops
    for i in range(num_attendees):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@friend_{i}",
                "source": f"@friend_{i}",
                "topic": "energy_level",
                "intensity": 0.2,
                "direction": -1,
            }
        ))
    builder.transition_to("afterparty_decision")

    # After party?
    builder.add_phase("afterparty_decision", "After party or sleep?")
    builder.with_action(_char_action("@friend_0", "say", "There's an afterparty at the hotel... we should go right?"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Go to the afterparty",
            "topic": "energy_level",
            "characters": [f"@friend_{i}" for i in range(num_attendees)],
            "threshold": 0.4,
        }
    ))
    builder.transition_to("morning_after")

    # Morning after
    builder.add_phase("morning_after", "The morning after")
    builder.with_action(_char_action("@friend_0", "say", "I can still hear the ringing in my ears... totally worth it."))
    builder.with_action(_char_action("@friend_1", "say", "Best weekend ever. Same time next year?"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Come back next year",
            "topic": "excitement",
            "characters": [f"@friend_{i}" for i in range(num_attendees)],
            "threshold": 0.5,
        }
    ))

    return builder.build()


def create_concert_experience(
    artist: str = "Taylor Swift",
    venue: str = "stadium",
) -> ScenarioDefinition:
    """Create a concert experience scenario."""
    builder = ScenarioBuilder(f"{artist} Concert")
    builder.with_description(f"Attending {artist} at the {venue}")

    builder.requires_role("superfan")
    builder.requires_role("casual_fan")
    builder.requires_role("brought_along")
    builder.requires_role("skeptic")

    builder.with_initial_beliefs("superfan", {"excitement": 0.99, "knowledge": 0.95})
    builder.with_initial_beliefs("casual_fan", {"excitement": 0.7, "knowledge": 0.5})
    builder.with_initial_beliefs("brought_along", {"excitement": 0.4, "knowledge": 0.1})
    builder.with_initial_beliefs("skeptic", {"excitement": 0.2, "knowledge": 0.3})

    builder.with_variable("artist", artist)
    builder.with_variable("songs_played", [])

    # Pre-show
    builder.add_phase("preshow", "Waiting for the show")
    builder.with_action(_char_action("@superfan", "say", f"I've been waiting 3 YEARS for this! I know every song!"))
    builder.with_action(_char_action("@skeptic", "say", "I don't really get the hype but okay..."))
    builder.with_action(_char_action("@brought_along", "say", "Who are we seeing again?"))
    builder.with_action(_char_action("@casual_fan", "say", "This is going to be fun! I love their hits."))
    builder.transition_to("opening")

    # Opening
    builder.add_phase("opening", "The lights go down")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": f"The crowd roars as {artist} takes the stage!", "topic": "concert"}
    ))
    builder.with_action(_char_action("@superfan", "say", "*screaming incoherently*"))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@skeptic", "source": "@superfan", "topic": "excitement", "intensity": 0.3, "direction": 1}
    ))
    builder.transition_to("deep_cuts")

    # Deep cuts - superfan shines
    builder.add_phase("deep_cuts", "Playing the deep cuts")
    builder.with_action(_char_action("@superfan", "say", "OH MY GOD THEY'RE PLAYING THE B-SIDE FROM 2012!"))
    builder.with_action(_char_action("@casual_fan", "say", "I don't know this one..."))
    builder.with_action(_char_action("@brought_along", "say", "Is this the bathroom break song?"))
    builder.with_action(_char_action("@superfan", "say", "NO! This is a MASTERPIECE! Listen to the bridge!"))
    builder.transition_to("hits")

    # The hits
    builder.add_phase("hits", "Playing all the hits")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "The opening notes of the biggest hit play...", "topic": "concert"}
    ))
    builder.with_action(_char_action("@casual_fan", "say", "YESSS I KNOW THIS ONE!"))
    builder.with_action(_char_action("@brought_along", "say", "Wait, I actually know this song!"))
    builder.with_action(_char_action("@skeptic", "say", "Okay... this is actually pretty good."))
    for role in ["casual_fan", "brought_along", "skeptic"]:
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@{role}", "source": "@superfan", "topic": "excitement", "intensity": 0.25, "direction": 1}
        ))
    builder.transition_to("encore")

    # Encore
    builder.add_phase("encore", "The encore")
    builder.with_action(_char_action("@superfan", "say", "ENCORE! ENCORE!"))
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": f"{artist} returns for one final song...", "topic": "concert"}
    ))
    builder.with_action(_char_action("@skeptic", "say", "Alright, I'll admit it... that was incredible."))
    builder.transition_to("afterglow")

    # Post-show
    builder.add_phase("afterglow", "After the show")
    builder.with_action(_char_action("@superfan", "say", "I'm never washing this hand. They looked RIGHT AT ME!"))
    builder.with_action(_char_action("@skeptic", "say", "When's their next tour? Asking for a friend."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "That was the best concert ever",
            "topic": "excitement",
            "characters": ["@superfan", "@casual_fan", "@brought_along", "@skeptic"],
            "threshold": 0.6,
        }
    ))

    return builder.build()


# =============================================================================
# SPORTS & GAMES
# =============================================================================

def create_pickup_basketball(
    players: int = 6,
) -> ScenarioDefinition:
    """Create a pickup basketball game scenario."""
    builder = ScenarioBuilder("Pickup Basketball")
    builder.with_description(f"{players} players, 3v3 at the park")

    for i in range(players):
        builder.requires_role(f"player_{i}")
        builder.with_initial_beliefs(f"player_{i}", {
            "confidence": 0.4 + 0.5 * random.random(),
            "fatigue": 0.0,
            "competitiveness": 0.5 + 0.4 * random.random(),
            "sportsmanship": 0.6 + 0.3 * random.random(),
        })

    builder.with_variable("score_team_a", 0)
    builder.with_variable("score_team_b", 0)
    builder.with_variable("game_to", 21)

    # Picking teams
    builder.add_phase("picking_teams", "Picking teams")
    builder.with_action(_char_action("@player_0", "say", "Shoot for teams? Make it take it?"))
    builder.with_action(_char_action("@player_1", "say", "I got next. Let's run it."))
    builder.with_action(ScenarioAction(
        action_type="form_coalition",
        params={
            "characters": [f"@player_{i}" for i in range(players)],
            "topic": "confidence",
            "similarity_threshold": 0.3,
            "coalition_name": "team_a",
        }
    ))
    builder.transition_to("early_game")

    # Early game
    builder.add_phase("early_game", "First few points")
    builder.with_action(_char_action("@player_0", "say", "Ball don't lie! That's AND ONE!"))
    builder.with_action(_char_action("@player_3", "say", "That was a foul! You hacked me!"))
    builder.with_action(_char_action("@player_1", "say", "Play through it, no blood no foul."))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@player_{i}" for i in range(players)], "topic": "trash_talk"}
    ))
    builder.transition_to("heating_up")

    # Game heats up
    builder.add_phase("heating_up", "Getting competitive")
    builder.with_action(_char_action("@player_0", "say", "I'm COOKING right now! Can't guard me!"))
    builder.with_action(_char_action("@player_3", "say", "That's cause you're traveling every play!"))
    for i in range(players):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@player_{i}",
                "source": f"@player_{(i+1) % players}",
                "topic": "competitiveness",
                "intensity": 0.15,
                "direction": 1,
            }
        ))
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={
                "target": f"@player_{i}",
                "source": f"@player_{i}",
                "topic": "fatigue",
                "intensity": 0.2,
                "direction": 1,
            }
        ))
    builder.transition_to("clutch_time")

    # Clutch time
    builder.add_phase("clutch_time", "Game point")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "Game point. Next bucket wins.", "topic": "basketball"}
    ))
    builder.with_action(_char_action("@player_0", "say", "Give me the ball. I got this."))
    builder.with_action(_char_action("@player_1", "say", "You been bricking all game, pass it!"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Player 0 takes the last shot",
            "topic": "confidence",
            "characters": ["@player_0", "@player_1", "@player_2"],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("post_game")

    # Post game
    builder.add_phase("post_game", "After the game")
    builder.with_action(_char_action("@player_0", "say", "Good game, good game. Run it back?"))
    builder.with_action(_char_action("@player_3", "say", "I need water first. Y'all got me dying out here."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Play another game",
            "topic": "fatigue",
            "characters": [f"@player_{i}" for i in range(players)],
            "threshold": 0.3,  # Low threshold = more likely to say no when tired
        }
    ))

    return builder.build()


def create_poker_night(
    players: int = 5,
    buy_in: int = 50,
) -> ScenarioDefinition:
    """Create a poker night scenario."""
    builder = ScenarioBuilder("Poker Night")
    builder.with_description(f"{players} friends, ${buy_in} buy-in")

    for i in range(players):
        builder.requires_role(f"player_{i}")
        builder.with_initial_beliefs(f"player_{i}", {
            "chip_stack": 1.0,
            "tilt_level": 0.0,
            "bluffing_confidence": 0.3 + 0.5 * random.random(),
            "read_on_others": 0.3 + 0.4 * random.random(),
        })

    builder.with_variable("pot", 0)
    builder.with_variable("hands_played", 0)

    # Setting up
    builder.add_phase("setup", "Setting up the game")
    builder.with_action(_char_action("@player_0", "say", f"Alright, ${buy_in} buy-in. Everyone got cash?"))
    builder.with_action(_char_action("@player_1", "say", "Venmo works too. Let's shuffle up and deal!"))
    builder.with_action(_char_action("@player_2", "say", "I'm feeling lucky tonight. Who wants to donate?"))
    builder.transition_to("early_hands")

    # Early hands
    builder.add_phase("early_hands", "Feeling each other out")
    builder.with_action(_char_action("@player_0", "say", "I'll raise. Let's see who's here to play."))
    builder.with_action(_char_action("@player_3", "say", "You always raise. I call."))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@player_{i}" for i in range(players)], "topic": "poker_banter"}
    ))
    builder.transition_to("big_hand")

    # A big hand develops
    builder.add_phase("big_hand", "Big pot brewing")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "The pot is getting huge. Three players still in.", "topic": "poker"}
    ))
    builder.with_action(_char_action("@player_0", "say", "All in."))
    builder.with_action(_char_action("@player_1", "say", "*stares intensely* You're bluffing. I can see it."))
    builder.with_action(_char_action("@player_2", "say", "This is too rich for me. I fold."))

    # Bluffing dynamics
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@player_1", "source": "@player_0", "topic": "read_on_others", "intensity": 0.2, "direction": -1}
    ))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Call the all-in bet",
            "topic": "bluffing_confidence",
            "characters": ["@player_1"],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("showdown")

    # Showdown
    builder.add_phase("showdown", "Cards revealed")
    builder.with_action(_char_action("@player_0", "say", "Flip 'em."))
    builder.with_action(ScenarioAction(
        action_type="conditional",
        params={
            "condition": {"conditions": [{"field": "variables.last_vote.passed", "operator": "eq", "value": True}], "mode": "all"},
            "then": {"action_type": "character_action", "character_id": "@player_1", "params": {"type": "say", "content": "CALLED IT! Read you like a book!"}},
            "else": {"action_type": "character_action", "character_id": "@player_0", "params": {"type": "say", "content": "Take it down. Should've called."}}
        }
    ))
    builder.transition_to("tilt_management")

    # Someone's on tilt
    builder.add_phase("tilt_management", "Emotions running high")
    builder.with_action(_char_action("@player_3", "say", "I can't catch a card! This is ridiculous!"))
    builder.with_action(_char_action("@player_4", "say", "That's poker, baby. Variance is a beast."))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@player_3", "source": "@player_3", "topic": "tilt_level", "intensity": 0.3, "direction": 1}
    ))
    builder.transition_to("late_night")

    # Late night
    builder.add_phase("late_night", "Late night decisions")
    builder.with_action(_char_action("@player_0", "say", "It's 2 AM. One more orbit or cash out?"))
    builder.with_action(_char_action("@player_3", "say", "I need to win my money back!"))
    builder.with_action(_char_action("@player_1", "say", "Classic tilt talk. You should quit while you're... behind."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Keep playing",
            "topic": "tilt_level",
            "characters": [f"@player_{i}" for i in range(players)],
            "threshold": 0.4,
        }
    ))

    return builder.build()


def create_golf_outing(
    players: int = 4,
) -> ScenarioDefinition:
    """Create a golf outing scenario."""
    builder = ScenarioBuilder("Golf Outing")
    builder.with_description(f"{players} friends on the course")

    for i in range(players):
        builder.requires_role(f"golfer_{i}")
        builder.with_initial_beliefs(f"golfer_{i}", {
            "skill_level": 0.3 + 0.5 * random.random(),
            "frustration": 0.0,
            "beer_count": 0.0,
            "enjoyment": 0.7,
        })

    builder.with_variable("holes_played", 0)
    builder.with_variable("best_shot", None)
    builder.with_variable("worst_shot", None)

    # First tee
    builder.add_phase("first_tee", "First tee jitters")
    builder.with_action(_char_action("@golfer_0", "say", "Alright, honors goes to... whoever wants it."))
    builder.with_action(_char_action("@golfer_1", "say", "I haven't played in months. Lower your expectations."))
    builder.with_action(_char_action("@golfer_2", "say", "What are we playing for? Gotta have some stakes."))
    builder.with_action(_char_action("@golfer_3", "say", "Dollar a hole, automatic press?"))
    builder.transition_to("front_nine")

    # Front nine
    builder.add_phase("front_nine", "Front nine action")
    builder.with_action(_char_action("@golfer_0", "say", "FORE! ...sorry about your windshield."))
    builder.with_action(_char_action("@golfer_1", "say", "That's out of bounds. Take a drop."))
    builder.with_action(_char_action("@golfer_2", "say", "I'm taking a breakfast ball. That didn't count."))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@golfer_{i}" for i in range(players)], "topic": "golf_excuses"}
    ))
    # Beer cart comes by
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "The beer cart approaches...", "topic": "golf"}
    ))
    for i in range(players):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@golfer_{i}", "source": f"@golfer_{i}", "topic": "beer_count", "intensity": 0.3, "direction": 1}
        ))
    builder.transition_to("turn")

    # The turn
    builder.add_phase("turn", "Making the turn")
    builder.with_action(_char_action("@golfer_0", "say", "I shot 52 on the front. That's basically par for me."))
    builder.with_action(_char_action("@golfer_3", "say", "You took like 8 mulligans though."))
    builder.with_action(_char_action("@golfer_1", "say", "We don't count those. Gentleman's rules."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Hot dogs at the turn",
            "topic": "enjoyment",
            "characters": [f"@golfer_{i}" for i in range(players)],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("back_nine")

    # Back nine
    builder.add_phase("back_nine", "Back nine drama")
    builder.with_action(_char_action("@golfer_2", "say", "I'm pressing. Double or nothing on the back."))
    builder.with_action(_char_action("@golfer_0", "say", "You're already down $15!"))
    builder.with_action(_char_action("@golfer_2", "say", "I'm due for a good hole!"))
    # Frustration builds
    for i in range(players):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@golfer_{i}", "source": f"@golfer_{i}", "topic": "frustration", "intensity": 0.2, "direction": 1}
        ))
    builder.transition_to("eighteenth")

    # 18th hole
    builder.add_phase("eighteenth", "Final hole")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "18th hole. All bets are on the line.", "topic": "golf"}
    ))
    builder.with_action(_char_action("@golfer_0", "say", "I just need a bogey to win the match."))
    builder.with_action(_char_action("@golfer_3", "say", "*skulls chip shot* ...I hate this game."))
    builder.with_action(_char_action("@golfer_1", "say", "See you next week? Same time?"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Play again next week",
            "topic": "enjoyment",
            "characters": [f"@golfer_{i}" for i in range(players)],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("nineteenth_hole")

    # 19th hole (bar)
    builder.add_phase("nineteenth_hole", "Post-round drinks")
    builder.with_action(_char_action("@golfer_0", "say", "First round's on the loser. That's you."))
    builder.with_action(_char_action("@golfer_3", "say", "I'm never playing this stupid game again."))
    builder.with_action(_char_action("@golfer_1", "say", "You said that last week."))
    builder.with_action(_char_action("@golfer_3", "say", "...Okay but THIS time I mean it."))

    return builder.build()


# =============================================================================
# DINING & PARTIES
# =============================================================================

def create_dinner_party(
    guests: int = 6,
    theme: str = "casual dinner",
) -> ScenarioDefinition:
    """Create a dinner party scenario."""
    builder = ScenarioBuilder(f"Dinner Party")
    builder.with_description(f"{guests} guests at a {theme}")

    builder.requires_role("host")
    for i in range(guests - 1):
        builder.requires_role(f"guest_{i}")
        builder.with_initial_beliefs(f"guest_{i}", {
            "social_energy": 0.6 + 0.3 * random.random(),
            "wine_consumption": 0.0,
            "enjoyment": 0.7,
            "conversational_momentum": 0.5,
        })

    builder.with_initial_beliefs("host", {"stress_level": 0.4, "hosting_satisfaction": 0.7})
    builder.with_variable("courses_served", 0)
    builder.with_variable("topics_discussed", [])

    # Arrivals
    builder.add_phase("arrivals", "Guests arriving")
    builder.with_action(_char_action("@host", "say", "Welcome! Come in, come in! Can I get you a drink?"))
    builder.with_action(ScenarioAction(
        action_type="foreach",
        params={
            "characters": [f"@guest_{i}" for i in range(guests - 1)],
            "action": {
                "action_type": "character_action",
                "character_id": "$current_character",
                "params": {"type": "say", "content": "This place looks amazing! Thanks for having us!"}
            }
        }
    ))
    builder.transition_to("cocktail_hour")

    # Cocktail hour
    builder.add_phase("cocktail_hour", "Pre-dinner drinks")
    builder.with_action(_char_action("@guest_0", "say", "So, how do you all know each other?"))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@guest_{i}" for i in range(min(4, guests - 1))], "topic": "small_talk"}
    ))
    for i in range(guests - 1):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@guest_{i}", "source": f"@guest_{i}", "topic": "wine_consumption", "intensity": 0.2, "direction": 1}
        ))
    builder.transition_to("dinner_served")

    # Dinner is served
    builder.add_phase("dinner_served", "Main course")
    builder.with_action(_char_action("@host", "say", "Dinner is served! Please, sit anywhere."))
    builder.with_action(_char_action("@guest_0", "say", "This looks INCREDIBLE! You made all this?"))
    builder.with_action(_char_action("@host", "say", "The secret ingredient is anxiety and wine."))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": ["@host"] + [f"@guest_{i}" for i in range(guests - 1)], "topic": "food_appreciation"}
    ))
    builder.transition_to("deep_conversation")

    # Conversation gets deep
    builder.add_phase("deep_conversation", "Real talk begins")
    builder.with_action(_char_action("@guest_1", "say", "Okay but what do you all REALLY think about..."))
    builder.with_action(ScenarioAction(
        action_type="negotiate",
        params={
            "characters": [f"@guest_{i}" for i in range(min(4, guests - 1))],
            "topic": "conversational_momentum",
            "rounds": 2,
            "agreement_threshold": 0.3,
        }
    ))
    builder.transition_to("controversial_topic")

    # Someone brings up a controversial topic
    builder.add_phase("controversial_topic", "Things get spicy")
    builder.with_action(_char_action("@guest_2", "say", "I have a hot take... *everyone leans in*"))
    builder.with_action(_char_action("@guest_0", "say", "Oh no, here we go..."))
    builder.with_action(_char_action("@host", "say", "More wine, anyone? Let's keep it civil!"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Agree to disagree and change the subject",
            "topic": "social_energy",
            "characters": [f"@guest_{i}" for i in range(guests - 1)],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("dessert")

    # Dessert
    builder.add_phase("dessert", "Sweet ending")
    builder.with_action(_char_action("@host", "say", "Who saved room for dessert?"))
    builder.with_action(_char_action("@guest_0", "say", "I always have room for dessert."))
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "A beautiful homemade dessert appears...", "topic": "dinner"}
    ))
    builder.transition_to("winding_down")

    # Winding down
    builder.add_phase("winding_down", "Evening wraps up")
    builder.with_action(_char_action("@guest_0", "say", "This was so lovely. We have to do this more often."))
    builder.with_action(_char_action("@host", "say", "Thank you all for coming. It means the world."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Rate this dinner party 10/10",
            "topic": "enjoyment",
            "characters": [f"@guest_{i}" for i in range(guests - 1)],
            "threshold": 0.7,
        }
    ))

    return builder.build()


def create_house_party(
    guests: int = 12,
) -> ScenarioDefinition:
    """Create a house party scenario."""
    builder = ScenarioBuilder("House Party")
    builder.with_description(f"Classic house party with {guests} people")

    builder.requires_role("host")
    builder.requires_role("dj")
    builder.requires_role("wallflower")
    builder.requires_role("social_butterfly")
    builder.requires_role("kitchen_dweller")
    builder.requires_role("late_arrival")

    builder.with_initial_beliefs("host", {"party_stress": 0.5, "neighbor_anxiety": 0.3})
    builder.with_initial_beliefs("dj", {"music_control": 0.9, "crowd_reading": 0.7})
    builder.with_initial_beliefs("wallflower", {"social_comfort": 0.3, "drink_count": 0.0})
    builder.with_initial_beliefs("social_butterfly", {"energy": 0.95, "connections_made": 0.0})
    builder.with_initial_beliefs("kitchen_dweller", {"snack_proximity": 1.0, "conversation_depth": 0.6})
    builder.with_initial_beliefs("late_arrival", {"fomo_level": 0.8, "fashionably_late_score": 0.9})

    builder.with_variable("party_vibe", 0.5)
    builder.with_variable("noise_complaints", 0)

    # Party starts
    builder.add_phase("early_party", "Party getting started")
    builder.with_action(_char_action("@host", "say", "Welcome! Drinks are in the kitchen, DJ's got the aux."))
    builder.with_action(_char_action("@dj", "say", "Starting with some chill vibes. Trust the process."))
    builder.with_action(_char_action("@wallflower", "say", "*stands awkwardly by the wall with a drink*"))
    builder.with_action(_char_action("@social_butterfly", "say", "HEYYY! Oh my god, do you know everyone here? Let me introduce you!"))
    builder.transition_to("kitchen_hangout")

    # Kitchen crew
    builder.add_phase("kitchen_hangout", "Kitchen conversations")
    builder.with_action(_char_action("@kitchen_dweller", "say", "The real party is always in the kitchen."))
    builder.with_action(_char_action("@social_butterfly", "say", "What are we talking about? I need to know everything!"))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": ["@kitchen_dweller", "@social_butterfly", "@wallflower"], "topic": "deep_kitchen_talk"}
    ))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@wallflower", "source": "@social_butterfly", "topic": "social_comfort", "intensity": 0.2, "direction": 1}
    ))
    builder.transition_to("peak_party")

    # Peak party
    builder.add_phase("peak_party", "Party peaks")
    builder.with_action(_char_action("@dj", "say", "Okay we're going UP now. This song is a certified banger."))
    builder.with_action(_char_action("@social_butterfly", "say", "EVERYONE TO THE LIVING ROOM!"))
    builder.with_action(_char_action("@late_arrival", "say", "*walks in* Did I miss anything?"))
    builder.with_action(_char_action("@host", "say", "You missed the first two hours but you're right on time for chaos."))
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "The party energy hits maximum levels...", "topic": "party"}
    ))
    builder.transition_to("noise_complaint")

    # Noise complaint
    builder.add_phase("noise_complaint", "The neighbors...")
    builder.with_action(_char_action("@host", "say", "Everyone SHHHH! I think someone's at the door..."))
    builder.with_action(_char_action("@dj", "say", "*turns music down*"))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@host", "source": "@host", "topic": "neighbor_anxiety", "intensity": 0.3, "direction": 1}
    ))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Keep the party going",
            "topic": "party_stress",
            "characters": ["@host", "@dj", "@social_butterfly"],
            "threshold": 0.4,
        }
    ))
    builder.transition_to("late_night_vibes")

    # Late night
    builder.add_phase("late_night_vibes", "Late night mode")
    builder.with_action(_char_action("@dj", "say", "Switching to late night vibes. We're chilling now."))
    builder.with_action(_char_action("@kitchen_dweller", "say", "The deep conversations happen after midnight."))
    builder.with_action(_char_action("@wallflower", "say", "This is actually fun. Why don't I do this more often?"))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": ["@wallflower", "@kitchen_dweller", "@late_arrival"], "topic": "life_philosophy"}
    ))
    builder.transition_to("party_ends")

    # Party ends
    builder.add_phase("party_ends", "Party winding down")
    builder.with_action(_char_action("@host", "say", "Alright everyone, you don't have to go home but you can't stay here..."))
    builder.with_action(_char_action("@social_butterfly", "say", "BEST. PARTY. EVER. When's the next one?"))
    builder.with_action(_char_action("@host", "say", "*looks at the mess* ...ask me in a month."))

    return builder.build()


# =============================================================================
# TRAVEL & ADVENTURES
# =============================================================================

def create_road_trip(
    travelers: int = 4,
    destination: str = "Vegas",
    hours: int = 6,
) -> ScenarioDefinition:
    """Create a road trip scenario."""
    builder = ScenarioBuilder(f"Road Trip to {destination}")
    builder.with_description(f"{travelers} friends, {hours} hour drive")

    builder.requires_role("driver")
    builder.requires_role("navigator")
    builder.requires_role("dj")
    builder.requires_role("backseat_sleeper")

    builder.with_initial_beliefs("driver", {"road_rage": 0.0, "fatigue": 0.0, "snack_dependency": 0.5})
    builder.with_initial_beliefs("navigator", {"directional_confidence": 0.6, "phone_battery": 1.0})
    builder.with_initial_beliefs("dj", {"aux_power": 1.0, "playlist_quality": 0.7})
    builder.with_initial_beliefs("backseat_sleeper", {"consciousness": 0.5, "neck_pain": 0.0})

    builder.with_variable("destination", destination)
    builder.with_variable("miles_driven", 0)
    builder.with_variable("stops_made", 0)

    # Departure
    builder.add_phase("departure", "Hitting the road")
    builder.with_action(_char_action("@driver", "say", f"Alright, {destination} here we come! Everyone buckled?"))
    builder.with_action(_char_action("@dj", "say", "I made a playlist for this. It's perfect. Trust me."))
    builder.with_action(_char_action("@backseat_sleeper", "say", "Wake me up when we get there."))
    builder.with_action(_char_action("@navigator", "say", "GPS says 6 hours but I know a shortcut."))
    builder.transition_to("music_debate")

    # Music debate
    builder.add_phase("music_debate", "The aux cord debate")
    builder.with_action(_char_action("@dj", "say", "*plays obscure song* This one's a deep cut, you'll love it."))
    builder.with_action(_char_action("@driver", "say", "What IS this? Play something everyone knows."))
    builder.with_action(_char_action("@navigator", "say", "Driver picks the music, that's the rule."))
    builder.with_action(_char_action("@dj", "say", "Fine, but you're all missing out on CULTURE."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "DJ keeps the aux",
            "topic": "aux_power",
            "characters": ["@driver", "@navigator", "@backseat_sleeper"],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("snack_stop")

    # Snack stop
    builder.add_phase("snack_stop", "Gas station stop")
    builder.with_action(_char_action("@driver", "say", "I need gas. And coffee. And snacks. 5 minutes."))
    builder.with_action(_char_action("@navigator", "say", "Get me a Red Bull and some gummy worms."))
    builder.with_action(_char_action("@backseat_sleeper", "say", "*wakes up* Where are we? Are we there?"))
    builder.with_action(_char_action("@dj", "say", "I'm getting enough snacks for the apocalypse."))
    builder.transition_to("highway_drama")

    # Highway drama
    builder.add_phase("highway_drama", "Highway adventures")
    builder.with_action(_char_action("@driver", "say", "THIS GUY! Use your turn signal!"))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@driver", "source": "@driver", "topic": "road_rage", "intensity": 0.2, "direction": 1}
    ))
    builder.with_action(_char_action("@navigator", "say", "Whoa, calm down. We're not in a rush."))
    builder.with_action(_char_action("@backseat_sleeper", "say", "Can you guys argue quieter? Some of us are trying to sleep."))
    builder.transition_to("getting_lost")

    # Getting lost
    builder.add_phase("getting_lost", "Navigational challenges")
    builder.with_action(_char_action("@navigator", "say", "Okay so the 'shortcut' might have been a mistake."))
    builder.with_action(_char_action("@driver", "say", "I KNEW we should've just stayed on the highway!"))
    builder.with_action(_char_action("@navigator", "say", "My phone died. Does anyone have a charger?"))
    builder.with_action(_char_action("@dj", "say", "This is fine. It's an adventure. Scenic route."))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@navigator", "source": "@driver", "topic": "directional_confidence", "intensity": 0.3, "direction": -1}
    ))
    builder.transition_to("back_on_track")

    # Back on track
    builder.add_phase("back_on_track", "Finding our way")
    builder.with_action(_char_action("@backseat_sleeper", "say", "Wait, I know where we are. Turn left up here."))
    builder.with_action(_char_action("@driver", "say", "Since when do you know anything? You've been asleep!"))
    builder.with_action(_char_action("@backseat_sleeper", "say", "I was listening. Also, my family's from around here."))
    builder.with_action(_char_action("@navigator", "say", "...why didn't you say that 30 minutes ago?"))
    builder.transition_to("arrival")

    # Arrival
    builder.add_phase("arrival", "We made it!")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": f"The {destination} skyline comes into view...", "topic": "road_trip"}
    ))
    builder.with_action(_char_action("@driver", "say", f"{destination}, baby! We made it!"))
    builder.with_action(_char_action("@dj", "say", "Cue the arrival song!"))
    builder.with_action(_char_action("@backseat_sleeper", "say", "Great, I'm awake just in time."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Best road trip ever",
            "topic": "road_rage",  # Inverse - low road rage = good trip
            "characters": ["@driver", "@navigator", "@dj", "@backseat_sleeper"],
            "threshold": 0.3,
        }
    ))

    return builder.build()


def create_beach_day(
    group_size: int = 5,
) -> ScenarioDefinition:
    """Create a beach day scenario."""
    builder = ScenarioBuilder("Beach Day")
    builder.with_description(f"Perfect beach day with {group_size} friends")

    for i in range(group_size):
        builder.requires_role(f"beachgoer_{i}")
        builder.with_initial_beliefs(f"beachgoer_{i}", {
            "sunburn_risk": 0.0,
            "hydration": 1.0,
            "relaxation": 0.3,
            "ocean_enthusiasm": 0.5 + 0.4 * random.random(),
        })

    builder.with_variable("umbrella_claimed", False)
    builder.with_variable("sandcastle_built", False)

    # Setup
    builder.add_phase("setup", "Setting up on the beach")
    builder.with_action(_char_action("@beachgoer_0", "say", "I call this spot! Perfect view, close to the water."))
    builder.with_action(_char_action("@beachgoer_1", "say", "Did anyone bring sunscreen? I forgot mine."))
    builder.with_action(_char_action("@beachgoer_2", "say", "SPF 50. You're welcome."))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@beachgoer_{i}" for i in range(group_size)], "topic": "setup_logistics"}
    ))
    builder.transition_to("water_time")

    # Getting in the water
    builder.add_phase("water_time", "Ocean time")
    builder.with_action(_char_action("@beachgoer_0", "say", "Last one in is a rotten egg!"))
    builder.with_action(_char_action("@beachgoer_1", "say", "The water is SO COLD! I need to go slow."))
    builder.with_action(_char_action("@beachgoer_3", "say", "*already swimming* What? Just get in!"))
    builder.with_action(_char_action("@beachgoer_4", "say", "I'll just... stay here and guard the stuff."))
    for i in range(group_size):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@beachgoer_{i}", "source": f"@beachgoer_{i}", "topic": "sunburn_risk", "intensity": 0.15, "direction": 1}
        ))
    builder.transition_to("beach_activities")

    # Beach activities
    builder.add_phase("beach_activities", "Fun in the sun")
    builder.with_action(_char_action("@beachgoer_0", "say", "Volleyball? Frisbee? Sandcastle competition?"))
    builder.with_action(_char_action("@beachgoer_2", "say", "Sandcastle competition, winner doesn't buy lunch."))
    builder.with_action(_char_action("@beachgoer_1", "say", "I'm just going to lay here and exist. Don't mind me."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Sandcastle competition",
            "topic": "ocean_enthusiasm",
            "characters": [f"@beachgoer_{i}" for i in range(group_size)],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("peak_sun")

    # Peak sun
    builder.add_phase("peak_sun", "Midday sun")
    builder.with_action(_char_action("@beachgoer_0", "say", "It is HOT out here. Did everyone reapply sunscreen?"))
    builder.with_action(_char_action("@beachgoer_3", "say", "I'm starting to look like a lobster..."))
    builder.with_action(_char_action("@beachgoer_4", "say", "That's because you've been in the sun for 3 hours straight."))
    for i in range(group_size):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@beachgoer_{i}", "source": f"@beachgoer_{i}", "topic": "hydration", "intensity": 0.2, "direction": -1}
        ))
    builder.transition_to("food_run")

    # Food run
    builder.add_phase("food_run", "Beach snacks")
    builder.with_action(_char_action("@beachgoer_2", "say", "Who wants to make a snack bar run?"))
    builder.with_action(_char_action("@beachgoer_0", "say", "Get me a frozen lemonade and some fries."))
    builder.with_action(_char_action("@beachgoer_1", "say", "Nachos. And more water. Lots of water."))
    builder.transition_to("golden_hour")

    # Golden hour
    builder.add_phase("golden_hour", "Sunset approaching")
    builder.with_action(ScenarioAction(
        action_type="broadcast",
        params={"message": "The sun begins to set, painting the sky orange and pink...", "topic": "beach"}
    ))
    builder.with_action(_char_action("@beachgoer_0", "say", "This view though. This is why we live here."))
    builder.with_action(_char_action("@beachgoer_1", "say", "I'm never leaving. You'll have to carry me out."))
    for i in range(group_size):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@beachgoer_{i}", "source": f"@beachgoer_{i}", "topic": "relaxation", "intensity": 0.3, "direction": 1}
        ))
    builder.transition_to("packing_up")

    # Packing up
    builder.add_phase("packing_up", "Heading home")
    builder.with_action(_char_action("@beachgoer_0", "say", "Did we get everything? Count the towels."))
    builder.with_action(_char_action("@beachgoer_3", "say", "I have sand in places I didn't know existed."))
    builder.with_action(_char_action("@beachgoer_2", "say", "Perfect beach day. Same time next week?"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Perfect 10/10 beach day",
            "topic": "relaxation",
            "characters": [f"@beachgoer_{i}" for i in range(group_size)],
            "threshold": 0.7,
        }
    ))

    return builder.build()


# =============================================================================
# EVERYDAY LIFE
# =============================================================================

def create_group_chat_drama(
    members: int = 6,
) -> ScenarioDefinition:
    """Create a group chat drama scenario."""
    builder = ScenarioBuilder("Group Chat Drama")
    builder.with_description(f"{members} friends in the group chat")

    for i in range(members):
        builder.requires_role(f"member_{i}")
        builder.with_initial_beliefs(f"member_{i}", {
            "drama_involvement": 0.3 + 0.4 * random.random(),
            "typing_indicator": 0.0,
            "read_receipts_anxiety": 0.5,
            "emoji_usage": 0.4 + 0.4 * random.random(),
        })

    builder.with_variable("messages_sent", 0)
    builder.with_variable("drama_level", 0)

    # Normal day
    builder.add_phase("normal_chat", "Just a normal day")
    builder.with_action(_char_action("@member_0", "say", "gm everyone"))
    builder.with_action(_char_action("@member_1", "say", "morning! ☕"))
    builder.with_action(_char_action("@member_2", "say", "anyone else see that sunset last night? 🌅"))
    builder.transition_to("making_plans")

    # Making plans
    builder.add_phase("making_plans", "Trying to make plans")
    builder.with_action(_char_action("@member_0", "say", "dinner friday? where should we go"))
    builder.with_action(_char_action("@member_1", "say", "I'm down for whatever"))
    builder.with_action(_char_action("@member_2", "say", "anywhere but that Italian place"))
    builder.with_action(_char_action("@member_3", "say", "wait why? the Italian place is great"))
    builder.with_action(_char_action("@member_2", "say", "...never mind. anywhere is fine."))
    builder.with_action(ScenarioAction(
        action_type="dialogue_round",
        params={"characters": [f"@member_{i}" for i in range(members)], "topic": "restaurant_debate"}
    ))
    builder.transition_to("drama_starts")

    # Drama starts
    builder.add_phase("drama_starts", "Someone stirs the pot")
    builder.with_action(_char_action("@member_4", "say", "so... did everyone hear about what happened at the party?"))
    builder.with_action(_char_action("@member_0", "say", "wait what?? spill"))
    builder.with_action(_char_action("@member_4", "say", "👀👀👀"))
    builder.with_action(_char_action("@member_5", "say", "*typing...*"))
    for i in range(members):
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@member_{i}", "source": "@member_4", "topic": "drama_involvement", "intensity": 0.2, "direction": 1}
        ))
    builder.transition_to("drama_peaks")

    # Drama peaks
    builder.add_phase("drama_peaks", "Drama escalates")
    builder.with_action(_char_action("@member_5", "say", "I can't believe this is even being discussed here"))
    builder.with_action(_char_action("@member_3", "say", "can someone just say what happened?!"))
    builder.with_action(_char_action("@member_1", "say", "should we take this offline?"))
    builder.with_action(_char_action("@member_0", "say", "no no we need to address this"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Keep discussing in the group chat",
            "topic": "drama_involvement",
            "characters": [f"@member_{i}" for i in range(members)],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("resolution")

    # Resolution
    builder.add_phase("resolution", "Cooling down")
    builder.with_action(_char_action("@member_1", "say", "okay everyone let's just calm down"))
    builder.with_action(_char_action("@member_0", "say", "agreed. sorry if things got heated"))
    builder.with_action(_char_action("@member_5", "say", "it's fine. we're good. ❤️"))
    builder.with_action(_char_action("@member_4", "say", "so... we still on for Friday?"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Still friends after this",
            "topic": "drama_involvement",
            "characters": [f"@member_{i}" for i in range(members)],
            "threshold": 0.3,
        }
    ))

    return builder.build()


def create_gym_session(
    gym_goers: int = 4,
) -> ScenarioDefinition:
    """Create a gym session scenario."""
    builder = ScenarioBuilder("Gym Session")
    builder.with_description(f"{gym_goers} people at the gym")

    builder.requires_role("regular")
    builder.requires_role("newbie")
    builder.requires_role("equipment_hogger")
    builder.requires_role("mirror_flexer")

    builder.with_initial_beliefs("regular", {"workout_focus": 0.8, "helpfulness": 0.6})
    builder.with_initial_beliefs("newbie", {"confidence": 0.3, "form_knowledge": 0.2, "intimidation": 0.7})
    builder.with_initial_beliefs("equipment_hogger", {"sets_remaining": 99.0, "awareness": 0.1})
    builder.with_initial_beliefs("mirror_flexer", {"self_admiration": 0.95, "actual_lifting": 0.3})

    builder.with_variable("sets_completed", 0)
    builder.with_variable("gym_etiquette_violations", 0)

    # Arrival
    builder.add_phase("arrival", "Walking in")
    builder.with_action(_char_action("@regular", "say", "Time to hit legs. Let's see if the squat rack is free."))
    builder.with_action(_char_action("@newbie", "say", "*looks around nervously* Okay, where do I even start?"))
    builder.with_action(_char_action("@equipment_hogger", "say", "*sitting on bench, scrolling phone, weights unused*"))
    builder.with_action(_char_action("@mirror_flexer", "say", "*adjusts tank top, checks reflection* Looking good."))
    builder.transition_to("equipment_drama")

    # Equipment drama
    builder.add_phase("equipment_drama", "The equipment situation")
    builder.with_action(_char_action("@regular", "say", "Excuse me, are you using this?"))
    builder.with_action(_char_action("@equipment_hogger", "say", "Yeah I got like 6 more sets. Maybe 7."))
    builder.with_action(_char_action("@regular", "say", "You've been on your phone for 10 minutes..."))
    builder.with_action(_char_action("@equipment_hogger", "say", "Resting is part of the workout, bro."))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@regular", "source": "@equipment_hogger", "topic": "workout_focus", "intensity": 0.2, "direction": -1}
    ))
    builder.transition_to("helping_newbie")

    # Helping the newbie
    builder.add_phase("helping_newbie", "Newbie needs help")
    builder.with_action(_char_action("@newbie", "say", "*struggling with form* Is this right?"))
    builder.with_action(_char_action("@regular", "say", "Hey, want some tips? That form could hurt your back."))
    builder.with_action(_char_action("@newbie", "say", "Oh thank god, yes please. I have no idea what I'm doing."))
    builder.with_action(_char_action("@regular", "say", "No worries, we all started somewhere. Here, like this..."))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@newbie", "source": "@regular", "topic": "confidence", "intensity": 0.3, "direction": 1}
    ))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@newbie", "source": "@regular", "topic": "form_knowledge", "intensity": 0.3, "direction": 1}
    ))
    builder.transition_to("mirror_moment")

    # Mirror moment
    builder.add_phase("mirror_moment", "Mirror confrontation")
    builder.with_action(_char_action("@mirror_flexer", "say", "*flexing in front of dumbbells others need*"))
    builder.with_action(_char_action("@regular", "say", "Can I grab those 30s behind you?"))
    builder.with_action(_char_action("@mirror_flexer", "say", "Sure, let me just finish this set... *continues flexing*"))
    builder.with_action(_char_action("@newbie", "say", "Is... is that person actually working out?"))
    builder.transition_to("cardio_debate")

    # Cardio vs weights
    builder.add_phase("cardio_debate", "The eternal debate")
    builder.with_action(_char_action("@regular", "say", "Alright, cardio to finish up."))
    builder.with_action(_char_action("@newbie", "say", "How much cardio should I be doing?"))
    builder.with_action(_char_action("@mirror_flexer", "say", "Cardio? Cardio kills gains, bro."))
    builder.with_action(_char_action("@regular", "say", "That's... that's not how any of this works."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Cardio is important",
            "topic": "workout_focus",
            "characters": ["@regular", "@newbie", "@mirror_flexer"],
            "threshold": 0.5,
        }
    ))
    builder.transition_to("cool_down")

    # Cool down
    builder.add_phase("cool_down", "Wrapping up")
    builder.with_action(_char_action("@regular", "say", "Good session. See you next time!"))
    builder.with_action(_char_action("@newbie", "say", "Thanks for all the help! This wasn't as scary as I thought."))
    builder.with_action(_char_action("@equipment_hogger", "say", "*still on the same bench, phone in hand*"))
    builder.with_action(_char_action("@mirror_flexer", "say", "*one last flex, takes gym selfie*"))

    return builder.build()


def create_coffee_shop_work(
    patrons: int = 4,
) -> ScenarioDefinition:
    """Create a coffee shop work session scenario."""
    builder = ScenarioBuilder("Coffee Shop Work Session")
    builder.with_description(f"{patrons} people working at the coffee shop")

    builder.requires_role("remote_worker")
    builder.requires_role("student")
    builder.requires_role("loud_phone_talker")
    builder.requires_role("writer")

    builder.with_initial_beliefs("remote_worker", {"productivity": 0.5, "caffeine_level": 0.0, "meeting_anxiety": 0.3})
    builder.with_initial_beliefs("student", {"study_focus": 0.4, "exam_panic": 0.7, "procrastination": 0.6})
    builder.with_initial_beliefs("loud_phone_talker", {"self_awareness": 0.1, "volume": 0.9})
    builder.with_initial_beliefs("writer", {"inspiration": 0.4, "blank_page_fear": 0.6, "people_watching": 0.8})

    builder.with_variable("coffees_consumed", 0)
    builder.with_variable("words_written", 0)

    # Setting up
    builder.add_phase("setup", "Finding a spot")
    builder.with_action(_char_action("@remote_worker", "say", "*claims table with laptop* Perfect, near an outlet."))
    builder.with_action(_char_action("@student", "say", "*spreads textbooks everywhere* I have SO much to study."))
    builder.with_action(_char_action("@writer", "say", "*opens laptop, stares at blank document*"))
    builder.with_action(_char_action("@loud_phone_talker", "say", "*phone rings loudly* HELLO? YEAH I CAN TALK!"))
    builder.transition_to("ordering")

    # Ordering
    builder.add_phase("ordering", "Getting coffee")
    builder.with_action(_char_action("@remote_worker", "say", "Large cold brew, please. It's going to be a long day."))
    builder.with_action(_char_action("@student", "say", "Quad shot espresso. Finals week."))
    builder.with_action(_char_action("@writer", "say", "Something moody... pour-over, single origin?"))
    for role in ["remote_worker", "student", "writer"]:
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@{role}", "source": f"@{role}", "topic": "caffeine_level" if role == "remote_worker" else "study_focus" if role == "student" else "inspiration", "intensity": 0.3, "direction": 1}
        ))
    builder.transition_to("work_begins")

    # Work begins
    builder.add_phase("work_begins", "Getting into the zone")
    builder.with_action(_char_action("@remote_worker", "say", "*puts on headphones, opens Slack*"))
    builder.with_action(_char_action("@student", "say", "*highlighter in hand, immediately opens TikTok*"))
    builder.with_action(_char_action("@writer", "say", "*types sentence, deletes it, types same sentence again*"))
    builder.with_action(_char_action("@loud_phone_talker", "say", "YEAH NO I'M AT A COFFEE SHOP. WORKING. VERY PRODUCTIVE."))
    builder.with_action(ScenarioAction(
        action_type="apply_pressure",
        params={"target": "@remote_worker", "source": "@loud_phone_talker", "topic": "productivity", "intensity": 0.3, "direction": -1}
    ))
    builder.transition_to("distraction")

    # Distraction
    builder.add_phase("distraction", "Focus breaking down")
    builder.with_action(_char_action("@remote_worker", "say", "*glares at loud phone talker*"))
    builder.with_action(_char_action("@student", "say", "I've been on my phone for 45 minutes... exam is tomorrow."))
    builder.with_action(_char_action("@writer", "say", "*observes room for 'research', writes nothing*"))
    builder.with_action(_char_action("@loud_phone_talker", "say", "CAN YOU HEAR ME? THE CONNECTION IS BAD!"))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "Say something to the loud phone talker",
            "topic": "productivity",
            "characters": ["@remote_worker", "@student", "@writer"],
            "threshold": 0.4,
        }
    ))
    builder.transition_to("refill")

    # Coffee refill
    builder.add_phase("refill", "Second coffee")
    builder.with_action(_char_action("@remote_worker", "say", "One more coffee. I have a 3pm meeting I'm dreading."))
    builder.with_action(_char_action("@student", "say", "Maybe I need more caffeine to focus..."))
    builder.with_action(_char_action("@writer", "say", "This is my third coffee. I've written 47 words."))
    builder.transition_to("breakthrough")

    # Breakthrough
    builder.add_phase("breakthrough", "Finally productive")
    builder.with_action(_char_action("@loud_phone_talker", "say", "*finally leaves*"))
    builder.with_action(_char_action("@remote_worker", "say", "*visible relief* Oh thank god."))
    builder.with_action(_char_action("@student", "say", "Wait, I actually understand this chapter now!"))
    builder.with_action(_char_action("@writer", "say", "*fingers flying over keyboard* The words are FLOWING."))
    for role in ["remote_worker", "student", "writer"]:
        builder.with_action(ScenarioAction(
            action_type="apply_pressure",
            params={"target": f"@{role}", "source": f"@{role}", "topic": "productivity" if role == "remote_worker" else "study_focus" if role == "student" else "inspiration", "intensity": 0.4, "direction": 1}
        ))
    builder.transition_to("packing_up")

    # Packing up
    builder.add_phase("packing_up", "Time to go")
    builder.with_action(_char_action("@remote_worker", "say", "Survived the meeting. Time to head home."))
    builder.with_action(_char_action("@student", "say", "I studied for 4 hours! (2 hours of actual studying)"))
    builder.with_action(_char_action("@writer", "say", "500 words! That's basically a novel."))
    builder.with_action(ScenarioAction(
        action_type="vote",
        params={
            "proposal": "This was a productive session",
            "topic": "productivity",
            "characters": ["@remote_worker", "@student", "@writer"],
            "threshold": 0.5,
        }
    ))

    return builder.build()


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

SOCIAL_SCENARIO_TEMPLATES = {
    # Festivals & Concerts
    "music_festival": create_music_festival,
    "concert": create_concert_experience,

    # Sports & Games
    "pickup_basketball": create_pickup_basketball,
    "poker_night": create_poker_night,
    "golf_outing": create_golf_outing,

    # Dining & Parties
    "dinner_party": create_dinner_party,
    "house_party": create_house_party,

    # Travel & Adventures
    "road_trip": create_road_trip,
    "beach_day": create_beach_day,

    # Everyday Life
    "group_chat": create_group_chat_drama,
    "gym_session": create_gym_session,
    "coffee_shop": create_coffee_shop_work,
}


def list_social_scenario_templates() -> List[str]:
    """List available social scenario templates."""
    return list(SOCIAL_SCENARIO_TEMPLATES.keys())


def create_social_scenario(template_name: str, **kwargs) -> ScenarioDefinition:
    """Create a scenario from a social template."""
    if template_name not in SOCIAL_SCENARIO_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list_social_scenario_templates()}")
    return SOCIAL_SCENARIO_TEMPLATES[template_name](**kwargs)
