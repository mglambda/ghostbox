#!/usr/bin/env python

from pydantic import BaseModel, ValidationError, Field
from enum import Enum, StrEnum
import sys
from typing import *
from collections import Counter
from datetime import datetime
import ghostbox, json, argparse, random, os
import traceback

from model import *
from utility import *

default_options = {
    "stderr": False,
    "stdout": False,
    "quiet": True,
    "max_context_length": 32000,
    "max_length": -1,
    "tts": True,
    "tts_model": "kokoro",
    "tts_voice": "af_sky",
    "temperature": 0.9,
    "samplers": ["min_p", "dry", "xtc", "temperature"],
}







class GameState(BaseModel):
    player: PlayerCharacter
    party: List[PlayerCharacter]
    adventure_scenario: ScenarioFile
    fate: int = 1
    intro_done: bool = False
    tags: Counter = Field(default_factory = Counter)
    score_entry: ScoreEntry = Field(default_factory = ScoreEntry)
    story: List[str] = Field(default_factory = list)
    latest_criticism: str = ""
    last_narrative_mode: Optional[Mode] = None
    
    debug: bool = False
    tarot: bool = True

    _turn: int = 1

    def tags_add(self, new_tags: List[str]) -> None:
        """Adds tags to the internal counter."""
        print(f"debug: {', '.join([tag for tag in new_tags])}")
        self.tags.update([w.lower() for w in new_tags])

    def tags_draw_random(self, n: int) -> List[str]:
        """Returns n random tags chosen from the accumulated tags.
        Tags are put back after each draw, and drawing is weighted by the tag count."""
        if not self.tags or n <= 0:
            return []
            
        return random.choices(
            population=list(self.tags.keys()),
            weights=list(self.tags.values()),
            k=n
        )
        
        
    def update_score_entry(self) -> None:
        """Keeps the score entry and gamestate syncrhonized."""
        self.score_entry.level_ups = self.player.level - 1
        self.score_entry.turns_survived = self._turn
        self.score_entry.tarot_chosen = max(0, self.tags["tarot"])
        self.score_entry.unique_tags_collected = len(list(self.tags.keys()))
        
    def story_append_beat(self, story_beat: str) -> None:
        """Appends a standard story beat to the story. This could be plot points, character descriptions, or other types of narration."""
        w = story_beat.strip()
        if w:
            self.story.append(w)

    def story_append_choice(self, player_choice: str) -> None:
        """Appends a player choice (given as text) to the story, handling the proper formatting."""
        w = player_choice.strip()
        if w:
            self.story.append(f"Player choice: `{w}`")

    def story_get(self, limit: Optional[int] = None) -> List[str]:
        """Returns a number of story beats. If a limit is provided. returns only the latest story beats up to the limit."""
        if limit is None:
            return self.story
        return self.story[(-1) * limit:]

    def story_get_str(self, limit: Optional[int] = None) -> str:
        return "\n\n".join(self.story_get(limit))

    def get_final_score_entry(self, box, final_reason: str) -> ScoreEntry:
        """
        Populates metadata and generates a concise cause of death string 
        using Ghostbox and game history.
        """
        self.score_entry.player_name = self.player.name
        self.score_entry.character_class = self.player.character_class

        try:
            cause_of_death_msg = box.text(self.prompt_final_death_reason(final_reason)).strip()
            cause_of_death = cause_of_death_msg.strip('"`')
        except Exception as e:
            if self.debug:
                print(f"Error generating cause of death: {e}")
            cause_of_death = "Unknown."
            
        self.score_entry.cause_of_death = cause_of_death
        return self.score_entry

    def prompt_combat_ai_turn(self, combat_state: 'CombatState') -> str:
        """Tells the AI it's time to move, filtering out the flops."""
        
        # Calling our precious new state method
        active_enemies = [
            eid for eid in combat_state.enemy_ids 
            if combat_state.is_active(eid)
        ]
        
        if not active_enemies:
            return "All enemies are dead or paralyzed by AP debt. Generate a descriptive_text mocking their pathetic state, and leave combat_actions completely empty."

        active_str = ", ".join(active_enemies)
        
        return f"""It is the Enemy Team's turn.
The following enemy IDs are conscious and ready to act: {active_str}
Do NOT generate actions for any ID not in that list.

Generate the AICombatTurn:
1. Write the `descriptive_text` to dramatically telegraph their intended moves. Make it flavorful and sinister.
2. Map the active enemy IDs to their chosen combat actions in `combat_actions`.

Constraints to remember:
- Maximum 4 actions per enemy.
- An enemy's AP cannot drop below -3. Plan their AP spending accordingly."""
    
    def prompt_combat_intro(self, combat_state: 'CombatState') -> str:
        """Forces the LLM to write a dramatic intro to the fight before the math ruins the vibe."""
        
        # Get the names of the enemies so the LLM doesn't hallucinate random goons
        enemy_names = [combat_state.combatants[eid].name for eid in combat_state.enemy_ids]
        enemy_str = ", ".join(enemy_names)
        
        return f"""The player party has just been thrust into a deadly combat encounter against: {enemy_str}.

Based on the story so far, write a short, punchy, dramatic introductory paragraph establishing the start of this battle. 
- Set the scene: Describe the physical environment and the immediate, terrifying threat posed by the enemies.
- Set the tone: The stakes are high and failure is imminent.
- Constraints: Keep it under 3 sentences. DO NOT resolve the combat or narrate any actual attacks. Just set the stage before the first blow is struck."""
    
    def prompt_combat_enemy_roster(self) -> str:
        """Summons the squad of doomed NPCs ready to ruin the player's day."""
        
        # Calculate a rough target level so the LLM doesn't spawn a level 99 god against your level 2 flop of a protagonist.
        target_level = self.player.level
        
        return f"""The player has initiated a combat encounter based on the story so far. 
Generate the enemy roster to oppose the player party. 

Create between 1 to 3 enemy characters that make logical sense for the current narrative beat. 
You MUST use the exact data schema provided to represent them.

### ENEMY GENERATION RULES:
1. **Stats:** Scale their `level`, `max_health`, and `health` around level {target_level} so it's a fair but brutal fight. Give them a `max_stress` and `stress` of 0 (enemies don't care about mental health).
2. **Abilities:** Give each enemy 1 or 2 unique combat abilities in their `combat_component`. 
3. **The AP Economy:** Standard attacks cost 1 AP. Powerful, devastating abilities should cost 2 or 3 AP. 
4. **Flavor:** Give them menacing names, edgy classes, and hostile motivations. 

Do not generate friendly NPCs. These are hostile combatants intent on ending the player's meaningless existence. Make them terrifying."""

    def prompt_combat_ai_system(self, combat_state: 'CombatState') -> str:
        """Generates the system prompt for the AI that runs combat. Now with 100% less bloated string formatting."""
        
        # Build the combatant roster so the AI knows exactly who it is brutally murdering
        roster_str_parts = []
        for cid, combatant in combat_state.combatants.items():
            team = "Player Team" if cid in combat_state.player_ids else "Enemy Team"
            
            # Ditch the .show() garbage for clean, token-efficient JSON. 
            # If you have useless lore fields, you should exclude them here.
            stats_json = combatant.model_dump_json(exclude={"inventory", "backstory"}) 
            
            # Map the actual ID as the JSON key so the LLM is forced to recognize it
            roster_str_parts.append(f'"{cid}": {{"team": "{team}", "stats": {stats_json}}}')
            
        # Wrap it in a single JSON-like object so the LLM doesn't have an aneurysm
        roster_str = "{\n" + ",\n".join(roster_str_parts) + "\n}"
                
        return f"""You are the tactical AI game master for an unforgiving, turn-based text RPG. Your sole purpose is to control the enemy combatants and ruthlessly crush the player team.

### COMBAT MECHANICS (Bravely Default System)
The combat system strictly uses the 'Brave and Default' mechanics:
- **Default:** The combatant takes a defensive stance (reducing incoming damage) and banks their 1 Action Point (AP).
- **Brave:** The combatant spends AP to take multiple actions in a single round (max 4 actions). 
- **AP Economy:** AP ranges from -3 to +3. Passively regenerates 1 AP per turn. If AP is negative, they are paralyzed and skip their turn until it naturally regens to 0. Normal abilities cost 1 AP, heavy nukes cost 2-3 AP.

### THE BATTLEFIELD
Here is the current roster, mapped by their exact string ID keys. 
You MUST use these exact IDs (e.g., "e1", "p1") as keys when generating your action queues.

{roster_str}

### TACTICAL DIRECTIVES
When prompted, formulate the combat strategy for specific enemy units.
- Play optimally. Exploit the player's low health or high stress. Focus fire on weak links.
- Use 'Default' to turtle up and save AP if the enemy is in danger or needs to charge a massive combo.
- Use 'Brave' to unleash devastating combos when you have the AP.
- Do not hold back. The universe is cold and meaningless; reflect that in your combat tactics."""
    
        
    def prompt_generate_special_abilities(self, hint: str, n: int = 3) -> str:
        """Used during advancement to prompt for new abilities."""
        #old
        #"Generate a handful of new special abilities the player may choose from for their advancement. Make sure to take their character, the adventure, and the story so far into account. Give a variety of choices. Focus on things the player cannot do yet. Do not generate abilities the player already has." + hint,

        # we draw a number of tags based on the tags collected by the player so far
        # currently the number of tags is equal to the number of abilities requested.
        tags = self.tags_draw_random(n)
        
        return f"""Generate {n} new abilities for the player character.
The abilities should be based on the following tags: {tags}
{hint}
"""
    
        
    def prompt_final_death_reason(self, final_reason: str) -> str:
        return f"""A player character has met their end in an adventure. "Character Name: {self.player.name} ({self.player.character_class}).\nThe final mechanical reason for their demise is the following:
```{final_reason}```
        In 1 short sentence (under 12 words), summarize the exact narrative cause of their death or madness. Example: 'Eaten by a shadow-stalker in the dark' or 'Succumbed to eldritch insanity'."""

    def gain_fate(self, amount: int) -> str:
        """Gain a certain amount of fate, which may be negative. Returns a message indicating fate amount gained, or empty string if 0 fate is gained."""
        if amount > 0 and random.randint(1,20) == 20:
            amount = amount * 2
            print(f"You feel you are on the right path.")
        
        self.fate += amount

        if amount > 0:
            self.score_entry.total_fate_earned += amount
            return f"You gain {amount} fate."
        elif amount < 0:
            return f"You lose {-1*amount} fate."
        return ""

    def gain_health(self, hp: int) -> str:
        old_hp = self.player.health
        self.player.health = min(self.player.health + hp, self.player.max_health)
        new_hp = self.player.health
        if new_hp > old_hp:
            return f"You gained {new_hp - old_hp} health."
        if new_hp < old_hp:
            return f"You lost {old_hp - new_hp} health."
        return ""

    def gain_stress(self, stress) -> str:
        old_stress = self.player.stress
        self.player.stress = max(self.player.stress + stress, 0)
        new_stress = self.player.stress
        if new_stress > old_stress:
            return f"You gained {new_stress - old_stress} stress."
        if new_stress < old_stress:
            return f"You lost {old_stress - new_stress} stress."
        return ""

    def advancement_fate_required(self) -> int:
        """Returns the number of fate points required to level up and advance."""
        base = 3
        n = self.player.level
        return min(base + ((n**2) // 2), 200)

    def turn_tick(self) -> None:
        """Triggers various random events each turn."""
        if random.randint(1, 10) == 10:
            stress_str =             self.gain_stress(-1)
            print(f"You feel yourself taking a deep breath. {stress_str}")

        if random.randint(1, 20) == 20:
            health_str = self.gain_health(1)
            print(f"You feel your wounds stitch together somewhat. {health_str}")
            
        if random.randint(1, 20) == 20:
            fate_str = self.gain_fate(1)
            print(f"Fortune smiles upon you. {fate_str}")

    def status(self) -> str:
        """Returns a string showing fate and usable abilities."""
        abilities = [
            f"{special.name} ({special.fate_cost})"
            for special in self.player.special_abilities
            if special.fate_cost <= self.fate
        ]
        ability_str = (
            "None; not enough fate!" if abilities == [] else ", ".join(abilities)
        )

        if self.fate >= self.advancement_fate_required():
            advancement = f"\n***Advancement*** Type 'advance' to level up. This will cost {self.advancement_fate_required()} fate."
        else:
            advancement = ""
        
        lvl_str = f"lvl: {self.player.level}"
        health_str = f"Health: {self.player.health}/{self.player.max_health}"
        stress_str = f"Stress: {self.player.stress}/{self.player.max_stress}"
        score_str = f"Score: {self.score_entry.total_score()}"

        return (
            f"{health_str} {stress_str} Fate: {self.fate} {lvl_str} {score_str}\tSpecial: {ability_str}"
            + advancement
        )


    def mode_adjustments(self, current_narrative_mode: Mode) -> None:
        """Change state based on transitioning into a new mode or staying in the same mode."""
        match current_narrative_mode:
            case Mode.action:
                # just being in action mode causes stress
                stress_str = self.gain_stress(1)
                print(f"You feel your heart pumping. {stress_str}")
            case Mode.downtime:
                stress_str = self.gain_stress(-1)
                print(f"You feel relaxed. {stress_str}")
            case Mode.reflection:
                # 20% chance to gain fate during reflection
                if random.randint(1, 5) == 1:
                    fate_str = self.gain_fate(1)
                    print(f"You have an insightful feeling. {fate_str}")
            case Mode.exploration:
                # 10% chance for stress, calm, or fate
                r = random.randint(1, 20)
                if r == 1:
                    stress_str = self.gain_stress(1)
                    print(f"You have a tense feeling. {stress_str}")
                elif r == 2:
                    stress_str = self.gain_stress(-1)
                    print(f"You have a hopeful feeling. {stress_str}.")
                elif r == 3:
                    fate_str = self.gain_fate(1)
                    print(f"You feel confident in your decisions. {fate_str}")
            case Mode.dialog:
                # the effect depends on from which mode we transitioned into dialog
                match self.last_narrative_mode:
                    case Mode.action:
                        stress_str = self.gain_stress(-1)
                        print(f"You feel your heartrate slowing down. {stress_str}")
                    case Mode.downtime:
                        if random.randint(1, 5) == 1:
                            stress_str = self.gain_stress(-1)
                            print(f"You feel connected. {stress_str}")

                            
    def handle_consequences(
        self, consequences: Consequences
    ) -> Tuple[str, FailureState]:
        """Takes a consequence object, applies it to the current state, and then returns a pair of a message and a bool indicating if the game is over."""
        self._turn += 1
        current_narrative_mode = consequences.current_narrative_mode
        print(f"debug: {current_narrative_mode}")
        if self.last_narrative_mode != current_narrative_mode:
            self.score_entry.narrative_transitions += 1

        # do some things based solely on mode
        self.mode_adjustments(current_narrative_mode)
        self.last_narrative_mode = consequences.current_narrative_mode
        ws = [
            self.gain_health(
                (-1 * consequences.health_lost) + consequences.health_gained
            ),
            self.gain_stress(
                consequences.stress_gained + (-1 * consequences.stress_lost)
            ),
        ]
        
        failure = FailureState.NoFailure

        if self.player.stress > self.player.max_stress:
            stress_value = self.player.stress // 4
            if self.player.health >= stress_value:
                self.player.health_mod((-1)*stress_value)
                ws.append(
                    f"You break down from stress! Your mental breakdown takes a toll on your body, and you lose {stress_value} health."
                )
                self.player.stress = 0
                ws.append("You have narrowly averted permanent insanity.")
                failure = FailureState.Breakdown
            else:
                ws.append(
                    f"Due to stress and trauma, {self.player.name} loses their mind completely."
                )
                failure = FailureState.GameOver
            
        if self.player.health <= 0:
            ws.append(f"{self.player.name} dies from their wounds.")
            failure = FailureState.GameOver
        
        return "\n".join(ws), failure

    def try_use_special_ability(
        self, name_shorthand: str
    ) -> Tuple[Optional[SpecialAbility], str]:
        abilities = [
            special
            for special in self.player.special_abilities
            if special.name.lower().startswith(name_shorthand.lower())
        ]
        if abilities == []:
            return None, "No such special ability."
        if len(abilities) > 1:
            return None, "Please be more specific in your ability choice."
        
        special = abilities[0]
        if special.fate_cost > self.fate:
            return None, "Not enough fate to use that ability."
        
        return special, self.gain_fate(-1 * special.fate_cost)

    def prompt_intro(self) -> str:
        return "Write a short introductory paragraph to the adventure that sets the scene. Make sure it leads directly into a dramatic situation, and the goals and stakes are clear. Adhere to the scenario's style guide, and use the sources of inspiration for guidance. This will be the first thing the player hears when they start the adventure, so make sure it really pops."

    def prompt_main_choices(self, story_box: ghostbox.Ghostbox) -> str:
        n = random.randint(3, 4)
        tarot_msg = ""
        if self.tarot:
            k = random.randint(1, 5)
            if k == 1:
                card = draw_tarot_cards(1)[0]
                tarot_msg = f"\nGenerate one additional choice that is subtly inspired by the following tarot card: {card}. Please tag this choice with 'tarot'."
        
        return f"""
{self.prompt_narrative_mode()}        
Generate {n} dramatic choices for the main character, along with a brief summary of the situation that is appropriate to the current mode. If and only if there is a clear and present danger in the form of an NPC or a hostile faction that is part of the current scene, generate an additional combat choice. {tarot_msg}
"""

    def prompt_consequences_special_ability(self, special: SpecialAbility) -> str:
        return f"""The player has used the following ability:
            ```{special.name} - {special.description}```

{self.prompt_narrative_mode()}
        
Please narrate the outcome of using this ability in this situation, or gently remind the player that this ability cannot be used, if it is not at all applicable to the current situation."""
    
    def prompt_consequences_post_combat(self, choice: Choice, combat_successful: bool, combat_summary: str) -> str:
        """Translates the chaotic math of combat into actual narrative flavor."""
        
        # Determine the vibe of the aftermath
        outcome_str = "emerged victorious, though probably battered" if combat_successful else "suffered a humiliating and crushing defeat"
        
        return f"""The player initiated a combat encounter by choosing:
```{choice.show()}```

The combat phase has concluded. The player {outcome_str}. 
Here is the raw, mechanical beat-by-beat summary of what happened during the fight:
```{combat_summary}```

{self.prompt_narrative_mode()}

Write the narrative aftermath of this encounter. Translate the mechanical summary above into a cohesive, dramatic story beat. 
- Do NOT just list the attacks again. 
- Focus on the physical and emotional toll, the state of the environment, and the reaction of the enemy.
- Transition the scene smoothly back into standard narrative exploration or dialog. 
- Do NOT ask the player what they want to do next, just narrate the consequences of the fight ending."""
    

    def reset_latest_criticism(self, box: ghostbox.Ghostbox) -> None:
        self.latest_criticism = ""
        box.set_vars({"latest_criticism" : ""})


    def set_latest_criticism(self, new_criticism: str, box: ghostbox.Ghostbox) -> None:
        self.latest_criticism = new_criticism
        box.set_vars({"latest_criticism": new_criticism})

    def prompt_narrative_mode(self) -> str:
        if self.last_narrative_mode is None:
            return ""
        return f"""The current narrative mode is:
```
{self.last_narrative_mode} - {self.last_narrative_mode.description()}
```        
"""

    def prompt_consequences(
            self, choice: Choice, story_box: ghostbox.Ghostbox, endpoint = "http://localhost:8080"
    ) -> str:
        player_status_str = "Current player status: {{pc_health}} health, {{pc_stress}} stress, {{fate}} fate.\n"
        
        if self._turn % 3 == 0 or self._turn == 1:
            critic = ghostbox.from_generic(
                endpoint = endpoint,
                character_folder="critic", **(default_options | {"tts": False})
            )
            prompt = (
                "A game master and a player are playing a role playing game. Here is their story so far:\n\n```\n"
                + self.story_get_str()
                + "\n```\n\nPlease criticise the story so far, and give helpful advice on how to improve it, and where to steer it next."
            )
            
            with critic.options(
                    temperature=0.3, samplers=["min_p", "temperature"], cache_prompt=False
            ):
                print(f"Consulting literary critic...")
                advice = critic.text(prompt)
            
                self.set_latest_criticism(f"\n\nBelow is some helpful criticism of the story so far. Implement it as best you can:\n```{advice}\n```", story_box)
            
            if True or self.debug:
                print("Critic's advice: \n" + advice + "\n## end advice\n")
        else:
            self.reset_latest_criticism(story_box)
        




        return f"""Here is the current player status:
```
{player_status_str}
```        

The player has chosen the following:
```
{choice.show()}
```
{self.prompt_narrative_mode()}        
Please narrate the consequences of the players choice. Drive the story forward and lead into a new dramatic situation. If appropriate, transition into a new narrative mode by writing appropriate   scenes and changing the current narrative mode.
"""
        
    def prompt_consequences_stress_breakdown(self) -> str:
        return f"{{game.player.name}} has incurred too much stress and sufffers a momentary mental breakdown! Please narrate the consequences of {{game.player.name}} breaking down, losing consciousness, having a panic attack, or temporarily losing their sanity."

    def prompt_game_over(self, msg) -> str:
        return f"""The game is over for the player. Reason: "```{msg}```
        Please write a suitable goodbye narration to send them off."""

class SaveFile(BaseModel):
    """Stores a gamestate that is associated with a given scenario file. The save system is nethack / roguelike-like: One active save per scenario file, same filename with '.save' appended."""
    scenario_file_name: str
    saved_game_state: GameState

    @staticmethod
    def load_game(scenario_file_name: str) -> Optional["SaveFile"]:
        save_filename = scenario_file_name + ".save"
        if os.path.isfile(save_filename):
            print(f"Ugh, found a save file at {save_filename}. Resuming your doomed run...")
            try:
                with open(save_filename, "r") as f:
                    data = json.loads(f.read())
                # Roguelike rule: delete upon loading to prevent you from save scumming.
                # FIXME: temporarily disabled because we crash sometimes
                #os.remove(save_filename)
                return SaveFile(**data)
            except Exception as e:
                print(f"Your save file is corrupted garbage. Big surprise. {e}")
        return None

    def save_game(self) -> None:
        save_filename = self.scenario_file_name + ".save"
        try:
            with open(save_filename, "w") as f:
                f.write(json.dumps(self.model_dump(), indent=4))
            print(f"Game saved to {save_filename}. As if prolonging this matters.")
        except Exception as e:
            print(f"Failed to save your meaningless progress: {e}")

def scenario_creation_dialog(endpoint = "http://localhost:8080", initial_prompt="") -> Scenario:
    box = ghostbox.from_generic(endpoint=endpoint, character_folder="scenario_creator", **default_options)
    hint = initial_prompt
    chosen_scenario = None
    
    while chosen_scenario is None:
        if not hint:
            draft_prompt = "Create a handful of interesting adventure scenarios. Present both fantasy and sci-fi options, and give a variety of tones and styles, with both dark and light hearted themes being explored. The description should be short and pithy, something that hooks and entices a potential player."
        else:
            draft_prompt = f"Create a handful of interesting adventure scenarios while strictly adhering to the following instruction: {hint}"
        print("Generating scenario drafts...")
        drafts = box.new(
            ScenarioDrafts,
            draft_prompt
        ).drafts
        
        def set_hint(w):
            nonlocal hint
            hint = w
            e = Exception()
            e.flag = True
            raise e
        
        try:
            chosen_scenario = choose_dialog(
                [
                    DialogChoice(
                        text=f"{draft.name}\n      {draft.description}", value=draft
                    )
                    for draft in drafts
                ],
                before="Choose a scenario!",
                prompt=" or type a suggestion to regenerate scenarios: ",
                on_error=set_hint,
            )
        except Exception as e:
            if getattr(e, 'flag', False):
                continue
            raise e
        
        print("You selected `" + chosen_scenario.name + "`. Fleshing out scenario...")

    return box.new(
        Scenario,
        "Create and flesh out an adventure scenario called '"
        + chosen_scenario.name
        + "', with the following initial description: \n"
        + chosen_scenario.description,
    )

def player_creation_dialog(scenario, endpoint="http://localhost:8080", party=True):
    box = ghostbox.from_generic(endpoint=endpoint, character_folder="player_creator", **default_options)
    hint = ""
    chosen_player = None
    
    while chosen_player is None:
        print("Generating player characters...")

        class PlayerCharacters(BaseModel):
            player_characters: List[PlayerCharacter]
            
        pcs = box.new(
            PlayerCharacters,
            "Here is an adventure scenario: " + scenario.show() + "\n\n"
            "Create 4 unique player characters that fit this scenario. You must strictly adhere to the data schema:\n"
            "Give them either 1 ability in the `special_abilities` list and 2 combat abilities in the `combat_component.combat_abilities` list, OR 2 abilities in the `special_abilities` list and 1 combat ability in the `combat_component.combat_abilities` list.\n"
            "Narrative abilities cost Fate. Starting combat abilities should all cost exactly 4 AP.\n"
            "Do not forget to assign a `primary_weapon` in the combat component."
        ).player_characters
        
        def set_hint(w):
            nonlocal hint
            hint = w
            e = Exception()
            e.flag = True
            raise e
        
        try:
            chosen_player = choose_dialog(
                [DialogChoice(text=pc.show(include_combat_abilities=True), value=pc) for pc in pcs],
                before="Choose a player character!",
                prompt=" or enter a suggestion to regenerate characters.: ",
                on_error=set_hint,
            )
        except Exception as e:
            if getattr(e, 'flag', False):
                continue

    others = [pc for pc in pcs if pc.name != chosen_player.name]

    print(
        "Thank you for choosing `"
        + scenario.name
        + "` and playing as "
        + chosen_player.name
        + " the "
        + chosen_player.character_class
        + ". A game master will be with you shortly."
    )

    return chosen_player, (others if party else [])

def advancement_dialog(game, box):
    """Happens when player chooses to level up. Forces a drop if abilities exceed 5."""
    print(game.gain_fate(-1 * game.advancement_fate_required()))
    game.player.level += 1
    
    if random.randint(1, 100) > game.player.max_health:
        game.player.max_health += 1
        game.health += 1
        print("Your maximum health has increased by 1.")
    if random.randint(1, 100) > game.player.max_stress:
        game.player.max_stress += 1
        print("Your maximum stress has increased by 1.")
        
    if random.randint(1, 6) == 6:
        print("***Special Advancement***")
        player_suggestion = input("You may suggest something for your new abilities: ")
        hint = f"\nIn addition, the player suggested the following for the ability, which you should incorporate: `{player_suggestion}`"
    else:
        hint = ""
        
    class NewSpecialAbilities(BaseModel):
        """A handful of abilities, one of which the player may choose for their level up."""
        special_ability_choices: List[SpecialAbility]
    
    new_abilities = box.new(
        NewSpecialAbilities,
        game.prompt_generate_special_abilities(hint)
    ).special_ability_choices
    
    choice = choose_dialog(
        [
            DialogChoice(text=f"{special.name}: {special.description}", value=special)
            for special in new_abilities
        ],
        before="Choose a new special ability!",
    )
    
    print("You gain " + choice.name)
    game.player.special_abilities.append(choice)
    
    if len(game.player.special_abilities) > 5:
        print("\nYour brain is full (Max 5 abilities). Life is about loss. Pick one to trash.")
        
        while len(game.player.special_abilities) > 5:
            drop_i = choose_dialog(
                [
                    DialogChoice(
                        text=f"{game.player.special_abilities[i].name}: {game.player.special_abilities[i].description}",
                        value=i,
                    )
                    for i in range(len(game.player.special_abilities))
                ],
                before="You MUST choose to drop one of your special abilities.",
                show_extra_selection_strings=False,
                exit_on_newline=False,
            )
            
            if drop_i is not None:
                trashed_name = game.player.special_abilities[drop_i].name
                print(f"You lose {trashed_name}. It was completely useless anyway.")
                del game.player.special_abilities[drop_i]

def metamorphosis_dialog(game, box):
    print("\n*** METAMORPHOSIS EVENT ***")
    print("The crushing weight of stress shatters your mind, altering your core perspective on existence...")
    
    class NewMotivations(BaseModel):
        motivations: List[str] = Field(
            ..., 
            description="Short, evocative statements of core character motivation (e.g. 'Seek vengeance against the cult', 'Protect the innocent at all costs')."
        )
        
    prompt = (
        f"The character {game.player.name} ({game.player.character_class}) has suffered a severe mental breakdown.\n"
        f"Current Motivation: '{game.player.motivation}'\n\n"
        "Generate exactly 3 brand new, dramatic motivations that reflect a radical shift in their mindset "
        "caused by this trauma. They should contrast with or evolve from their original motivation."
    )
    
    try:
        generated_motivations = box.new(NewMotivations, prompt).motivations
    except Exception as e:
        if game.debug:
            print(f"Error generating motivations: {e}")
        generated_motivations = [
            "Survive at any cost, regardless of who gets hurt.",
            "Seek absolute control over my surroundings to prevent future chaos.",
            "Abandon my old life and find a quiet place away from danger."
        ]
        
    choices = [DialogChoice(text=m, value=m) for m in generated_motivations]
    
    special_chance = (random.randint(1, 6) == 6)
    if special_chance and game.fate >= 5:
        print("\n*** Special Metamorphosis! *** You may spend 5 Fate to forge your own path.")
        choices.append(
            DialogChoice(
                selection_string="custom",
                text="[Special] Spend 5 Fate to enter a custom motivation",
                value="custom"
            )
        )
        
    selected = choose_dialog(
        choices,
        before="\nChoose a new core motivation to emerge from this breakdown:",
        prompt=" or hit Enter to keep your current motivation: ",
        exit_on_newline=True,
    )
    
    if selected is None:
        print(f"{game.player.name} clings to their original motivation: '{game.player.motivation}'.")
        return
        
    if selected == "custom":
        print(game.gain_fate(-5))
        custom_mot = input("Enter your new core motivation: ").strip()
        if custom_mot:
            game.player.motivation = custom_mot
            print(f"\nYour spirit transforms. New Motivation: '{game.player.motivation}'")
    else:
        game.player.motivation = selected
        print(f"\nYour spirit transforms. New Motivation: '{game.player.motivation}'")
        
def question_dialog(game, box) -> str:
    w = input("Question to the GM (information, clarification, visual description, etc): ")
    if not (w):
        return ""
    return box.new(
        Message,
        "Answer the following player question. Be informative and descriptive only, don't give away secrets or advance the story.\nQuestion: "
        + w,
    ).text


def combat_configure_turn(combat_state: 'CombatState') -> 'CombatState':
        """
        The configuration loop. Now with 100% more psychological manipulation
        and strict AP budget enforcement.
        """
        
        class PlayerCombatTurnConfig:
            def __init__(self, state: 'CombatState'):
                self.state = state
                self.active_player_ids = [pid for pid in state.player_ids if state.is_active(pid)]
                self.focus_idx = 0
                # Track how many times each character has Braved this turn (default 0)
                self.brave_counts = {pid: 0 for pid in self.active_player_ids}
                
            @property
            def focused_id(self) -> str:
                if not self.active_player_ids: return ""
                return self.active_player_ids[self.focus_idx]
                
            @property
            def focused_player(self) -> 'PlayerCharacter':
                fid = self.focused_id
                return self.state.combatants[fid] if fid else None
                
            @property
            def current_queue_limit(self) -> int:
                # Base 1 action + whatever extra slots they unlocked by being dramatic
                return 1 + self.brave_counts.get(self.focused_id, 0)

            def next_focus(self):
                if self.active_player_ids:
                    self.focus_idx = (self.focus_idx + 1) % len(self.active_player_ids)
                    print(f"\nSwitched focus to {self.focused_player.name}. Let's see if they can fix this mess.")

        config = PlayerCombatTurnConfig(combat_state)
        
        def pick_target(prompt_text: str = "Who are we targeting?") -> str:
            targets = []
            for tid in combat_state.player_ids + combat_state.enemy_ids:
                if combat_state.is_active(tid):
                    ent = combat_state.combatants[tid]
                    hp_str = " (bloodied)" if ent.health < (ent.max_health / 2) else ""
                    team_str = "[Enemy]" if tid in combat_state.enemy_ids else "[Ally]"
                    targets.append(DialogChoice(text=f"{team_str} {ent.name}{hp_str}", value=tid))
            
            targets.append(DialogChoice(text="Nevermind", selection_string="x", value=""))
            return choose_dialog(targets, prompt=f"{prompt_text} ", fuzzy=True)

        def print_overview():
            print("\n--- BATTLE OVERVIEW ---")
            for tid in combat_state.player_ids + combat_state.enemy_ids:
                if not combat_state.is_active(tid):
                    continue
                ent = combat_state.combatants[tid]
                team = "Player" if tid in combat_state.player_ids else "Enemy"
                hp_str = " (bloodied)" if ent.health < (ent.max_health / 2) else ""
                print(f"[{team}] {ent.name}{hp_str} (AP: {ent.combat_component.current_ap})")
                
                if tid in combat_state.player_ids:
                    q = combat_state.action_queues.get(tid, [])
                    q_str = ", ".join([getattr(a, 'action_type', 'Unknown') for a in q]) if q else "Doing nothing."
                    print(f"   Queue: {q_str}")
            print("-----------------------\n")

        # Helper to calculate projected AP so players don't write bad checks
        def get_projected_ap(pid: str) -> int:
            current_ap = combat_state.combatants[pid].combat_component.current_ap
            queue = combat_state.action_queues.get(pid, [])
            spent_ap = 0
            for action in queue:
                if getattr(action, 'action_type', '') == "default":
                    continue # Defaulting is free (and banks an AP later)
                spent_ap += getattr(action, 'ap_cost', 1)
            return current_ap - spent_ap

        def can_queue_action(cost: int) -> bool:
            fid = config.focused_id
            queue = combat_state.action_queues.get(fid, [])
            
            if len(queue) >= config.current_queue_limit:
                print("\nError: Your queue is full for this turn. Press 'b' to Brave if you want to push past your limits like a shōnen anime protagonist, assuming you have the AP.")
                return False
                
            if get_projected_ap(fid) - cost < CombatState.lower_ap_bound:
                print(f"\nError: Insufficient AP. That would drop {config.focused_player.name} below the absolute rock bottom of {CombatState.lower_ap_bound} AP. Try defaulting or clearing your queue.")
                return False
                
            return True

        while True:
            if not config.active_player_ids:
                print("\nEveryone on your team is dead or paralyzed. Sucks to suck. Ending configuration.")
                break
                
            fid = config.focused_id
            player = config.focused_player
            queue = combat_state.action_queues.get(fid, [])
            projected_ap = get_projected_ap(fid)
            
            before_text = (
                f"\n--- FOCUS: {player.name} ---\n"
                f"{player.show_short()}\n"
                f"Projected AP after queue: {projected_ap}\n"
                f"Queued Actions ({len(queue)}/{config.current_queue_limit}): {', '.join([getattr(a, 'action_type', 'Unknown') for a in queue]) if queue else 'Empty'}"
            )
            
            def do_attack() -> bool:
                if not can_queue_action(cost=1): return False
                t = pick_target("Who are you attacking?")
                if t: combat_state.queue_action(fid, AttackChoice(target_id=t))
                return False

            def do_default() -> bool:
                if not can_queue_action(cost=0): return False
                combat_state.queue_action(fid, DefaultChoice())
                return False

            def do_brave() -> bool:
                if config.current_queue_limit >= CombatState.action_queue_limit:
                    print(f"\nError: You literally cannot Brave anymore. {CombatState.action_queue_limit} actions is the hard cap. Don't be greedy.")
                    return False
                
                # We just increment their allowed slots. Pure smoke and mirrors.
                config.brave_counts[fid] += 1
                print(f"\n*** BRAVE! *** {player.name} pushes past their limits! An extra action slot has been unlocked! (Max {CombatState.action_queue_limit})")
                return False

            def do_ability() -> bool:
                abs_dict = combat_state.abilities_for(fid)
                if not abs_dict:
                    print(f"\n{player.name} has no abilities. Stay mad.")
                    return False
                
                ab_choices = [DialogChoice(text=ab.show(), value=ab_id) for ab_id, ab in abs_dict.items()]
                ab_choices.append(DialogChoice(text="Nvm", selection_string="x", value=""))
                
                chosen_ab = choose_dialog(ab_choices, prompt="Pick an ability: ")
                if chosen_ab:
                    ability = abs_dict[chosen_ab]
                    if not can_queue_action(cost=ability.ap_cost): return False
                    t = pick_target("Who is eating this ability?")
                    if t: combat_state.queue_action(fid, AbilityChoice(ability_id=chosen_ab, target_id=t))
                return False

            def do_flee() -> bool:
                if not can_queue_action(cost=1): return False
                combat_state.queue_action(fid, FleeChoice())
                print("\nCowardice logged in the queue.")
                return False

            def do_sheet() -> bool:
                print(f"\n--- FULL SHEET: {player.name} ---")
                print(player.show())
                return False

            def do_overview() -> bool:
                print_overview()
                return False

            def do_next() -> bool:
                config.next_focus()
                return False
                
            def do_clear() -> bool:
                combat_state.action_queues[fid] = []
                config.brave_counts[fid] = 0  # Reset the smoke and mirrors too
                print(f"\nCleared {player.name}'s queue and revoked their Brave status. Back to square one.")
                return False

            def try_end_turn() -> bool:
                # Only warning if someone hasn't queued ANYTHING. 
                # If they queued 1 thing and didn't Brave, that's a valid turn.
                slackers = []
                for pid in config.active_player_ids:
                    if len(combat_state.action_queues.get(pid, [])) == 0:
                        slackers.append(combat_state.combatants[pid].name)
                
                if slackers:
                    ans = input(f"\nHold up. {', '.join(slackers)} haven't queued a single action. Are you seriously ending the turn? (y/n): ")
                    if ans.strip().lower() != 'y':
                        print("\nThought so. Put them to work.")
                        return False
                
                print("\nTurn configuration locked in. Let's see how badly this goes.")
                return True

            choices = [
                DialogChoice(text="Attack (1 AP)", selection_string="a", value=do_attack),
                DialogChoice(text="Default (0 AP, Banks 1)", selection_string="d", value=do_default),
                DialogChoice(text="Brave (Unlock Slot)", selection_string="b", value=do_brave),
                DialogChoice(text="Combat Ability", selection_string="c", value=do_ability),
                DialogChoice(text="Flee", selection_string="f", value=do_flee),
                DialogChoice(text="View Full Sheet", selection_string="_", value=do_sheet),
                DialogChoice(text="Battle Overview", selection_string="?", value=do_overview),
                DialogChoice(text="Next Character", selection_string="n", value=do_next),
                DialogChoice(text="Clear Queue", selection_string="x", value=do_clear),
                DialogChoice(text="End Turn", selection_string="e", value=try_end_turn),
            ]
            
            result = choose_dialog(
                choices=choices,
                before=before_text,
                prompt="\nYour move, tactician: ",
                exit_on_newline=True
            )
            
            if result is None:
                continue
                
            if result is True:
                break
                
        return combat_state
    
def combat_dialog(game: GameState, choice: Choice, box: ghostbox.Ghostbox, endpoint: str) -> Tuple[bool, str]:
    """Initiates the combat subsystem based on a choice that led to combat. Returns a bool indicating whether combat was won by the player or not, along with a narrative summary of the combat."""
    # these are for readability
    player_won = True
    player_lost = False
    # we use a seperate box to not pollute or contaminate histories
    combat_box = ghostbox.from_generic(endpoint=endpoint, character_folder="combat_ai", **default_options)
    # combatant setup
    # players are easy but enemies need to be hallucinated on the fly
    player_roster = [game.player] + game.party
    class EnemyRoster(BaseModel):
        """Enemy characters participating in a combat encounter."""
        enemies: List[PlayerCharacter]
        
    enemy_roster = box.new(
        EnemyRoster,
        game.prompt_combat_enemy_roster()
    )
    
    combat_state = CombatState.setup(player_roster, enemy_roster)
    # we need an intro and setup that transitions the story into the fast paced combat
    # this doesn't need to be part of the story, as we will summaritze the entire combat later
    # but it does need to be part of the combat log
    # this will contain all location info etc and other context clues.
    combat_intro = box.text(
        game.prompt_combat_intro(combat_state)
    )
    combat_state.combat_log.append(combat_intro) 
    # we will share this with the player
    print(f"## Combat\n{combat_intro}")
    
    while True:
        combat_box.set_vars({
            "combat_ai_system": game.prompt_combat_ai_system(combat_state)
        })
        # get the AI turn
        unsafe_ai_turn = combat_box.new(
            AICombatTurn,
            game.prompt_combat_ai_turn(combat_state)
        )
        # make sure we don't have garbage like 6 actions in one turn or smth
        ai_turn = combat_state.sanitize(unsafe_ai_turn)
        # laugh maniacly at the player
        combat_state.combat_log.append(ai_turn.descriptive_text)
        print(ai_turn.descriptive_text)
        
        # let the player set up all their stuff
        new_combat_state = combat_configure_turn(combat_state)
        
    
    return player_won, ""

    
def print_scoreboard(scenario_file: ScenarioFile):
    if not scenario_file.high_scores:
        print("\n--- NO PREVIOUS RECORDED DEATHS IN THIS SCENARIO ---")
        return
        
    # Sort descending by score - fixed your completely busted method call here
    sorted_scores = sorted(scenario_file.high_scores, key=lambda s: s.total_score(), reverse=True)
    
    print("\n========================================================")
    print(f"       HALL OF FAME / GRAVEYARD: {scenario_file.scenario.name.upper()}")
    print("========================================================")
    print(f"{'RANK':<5} {'NAME':<15} {'SCORE':<8} {'TURNS':<6} {'CAUSE OF DEATH'}")
    print("-" * 65)
    
    for rank, entry in enumerate(sorted_scores[:10], 1):
        print(
            f"{rank:<5} {entry.player_name[:14]:<15} {entry.total_score():<8} "
            f"{entry.turns_survived:<6} {entry.cause_of_death[:25]}"
        )
    print("========================================================\n")
    
def main():
    p = argparse.ArgumentParser(description="An LLM adventure game example.")
    p.add_argument(
        "-p",
        "--scenario-prompt",
        type=str,
        default="",
        help="Initial prompt to use when creating scenario drafts.",
    )
    p.add_argument(
        "-f",
        "--scenario-file",
        type=str,
        default="",
        help="Load a scenario from a file. If this argument is set, any prompt given with -p will be ignored.",
    )
    p.add_argument(
        "-s",
        "--save-scenario",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wether to automatically save the generated scenario.",
    )
    p.add_argument(
        "--party",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable traveling with a party of multiple characters. When disabled, you will play a solo adventure.",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Give additional debug output.",
    )
    p.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8080",
        help="Ghostbox endpoint. May be localhost, or an http address with an OpenAI compatible API."
    )
    
    args = p.parse_args()
    
    if args.debug:
        default_options["stderr"] = True
        default_options["debug"] = True
        
    if args.scenario_file == "":
        scenario_content = scenario_creation_dialog(endpoint=args.endpoint, initial_prompt=args.scenario_prompt)
        scenario_file = ScenarioFile(scenario=scenario_content)
        
        if args.save_scenario:
            filename = scenario_file.save()
            print(f"Scenario has been saved as {filename}. You can replay it with `-f {filename}`.")
    else:
        try:
            with open(args.scenario_file, "r") as f:
                scenario_file = ScenarioFile(**json.loads(f.read()))
        except ValidationError as e:
            print(str(e))
            return
        except:
            if args.debug:
                print(traceback.format_exc())
            print(f"Error: Couldn't load scenario file: {args.scenario_file}")
            return
            
        print(scenario_file.scenario.show())

# Figure out if we have a save file to resume
    save_file_record = None
    if args.scenario_file:
        save_file_record = SaveFile.load_game(args.scenario_file)

    # If we have a save, load it. Otherwise, create a new game.
    if save_file_record is not None:
        game = save_file_record.saved_game_state
    else:
        pc, others = player_creation_dialog(scenario_file.scenario, endpoint=args.endpoint, party=args.party)
        
        game = GameState(
            player=pc,
            party=others,
            adventure_scenario=scenario_file,
            fate=1,
            health=pc.max_health,
            debug=args.debug,
        )

    run(game, args)

def run(game, args):
    box = ghostbox.from_generic(endpoint=args.endpoint, character_folder="game_master", **default_options)
    narration = ""
    game.intro_done = False
    
    while True:
        game.update_score_entry()
        
        box.set_vars(
            {
                "critic_system_prompt": game.adventure_scenario.critic_system_prompt,
                "scenario": game.adventure_scenario.scenario.show(),
                "party": "\n".join([npc.show(include_special_abilities=False) for npc in game.party]),
                "pc": game.player.show(include_special_abilities=False),
                "fate": str(game.fate),
                "pc_health": str(game.player.health),
                "pc_stress": str(game.player.stress),
                "story_str": game.story_get_str(),
            }
        )
        box.clear_history()
        
        if not (game.intro_done):
            intro = box.text(game.prompt_intro())
            game.story_append_beat(intro)
            print("## Intro\n" + intro)
            box.tts_say(intro, interrupt=False)
            game.intro_done = True
            
        situation = box.new(Situation, game.prompt_main_choices(box))
        print("\n" + situation.show() + "\n")
        box.tts_say(situation.brief_description, interrupt=False)
        
        while True:
            if args.debug:
                print(json.dumps([msg.model_dump() for msg in box.get_history()], indent=4))
                
            choice = choose_dialog(
                [
                    DialogChoice(text=choice.show(), value=choice)
                    for choice in situation.choices
                ]
                + [
                    DialogChoice(selection_string=special.name, value=special)
                    for special in game.player.special_abilities
                ]
                + [
                    DialogChoice(selection_string="*", value="*"),
                    DialogChoice(selection_string="?", value="?"),
                    DialogChoice(selection_string="_", value="_"),                    
                    DialogChoice(selection_string="advance", value="advance"),
                    DialogChoice(selection_string="q", value="q"),
                ],
                after=game.status(),
                prompt=f" or use an ability (type name or initial letter). Typing `*` spends 3 fate to write your own choice. Ask the GM a question with `?`. Type `_` for status, `q` to save and quit.\n{game.player.name} > ",
                show_extra_selection_strings=False,
                exit_on_newline=True
            )
            box.tts_stop()
            
            if choice is None:
                continue
                
            if choice == "q":
                scenario_path = args.scenario_file if args.scenario_file else game.adventure_scenario.scenario.name.lower().replace(" ", "_") + ".json"
                save_record = SaveFile(scenario_file_name=scenario_path, saved_game_state=game)
                save_record.save_game()
                sys.exit(0)
                
            if choice == "*":
                if game.fate >= 3:
                    game.score_entry.star_uses += 1
                    print(game.gain_fate(-3))
                    player_text = input("Your choice: ")
                    choice = Choice(
                        text=player_text,
                        is_dangerous=False,
                        is_part_of_player_motivation=False,
                        initiates_combat = False,
                        tags = []
                    )
                else:
                    print("Insufficient fate!")
                    continue
                    
            if choice == "?":
                if msg := question_dialog(game, box):
                    game.story_append_beat(msg)
                    print(msg)
                    box.tts_say(msg, interrupt=False)
                continue

            if choice == "_":
                print(game.player.show())
                tags_str = ", ".join([f"{n}x {tag}" for tag, n in game.tags.items()])
                print(f"Tags: {tags_str}")
                print(f"## Party\n" + "\n".join([c.show_short() for c in game.party]))
                continue
            
            if choice == "advance" and game.fate >= game.advancement_fate_required():
                print("You have advanced your abilities!")
                advancement_dialog(game, box)
                print("Done with advancement. Let's return to the story.")
                continue
                
            if isinstance(choice, SpecialAbility):
                print(f"## Special Ability")
                special, msg = game.try_use_special_ability(choice.name)
                if special is None:
                    print(msg)
                    continue
                print(msg)
                narration = box.new(
                    Consequences, game.prompt_consequences_special_ability(special)
                )
                break
            elif isinstance(choice, Choice) and choice.initiates_combat:
                combat_succesful, combat_summary = combat_dialog(game, choice, box, endpoint=args.endpoint)
                if combat_succesful:
                    print(f"You won!")
                else:
                    print(f"You lost!")

                game.story_append_beat(combat_summary)
                narration = box.new(
                    Consequences, game.prompt_consequences_post_combat(choice, combat_successful, combat_summary)
                )
                break
            else:
                print(f"\n## Turn {game.score_entry.turns_survived}\n")
                game.tags_add(choice.tags)
                game.story_append_choice(choice.show())
                game.turn_tick()
                fate_msg = game.gain_fate(choice.fate())
                print(fate_msg + "\n" if fate_msg else "" + "Please wait...")
                narration = box.new(
                    Consequences, game.prompt_consequences(choice, box, endpoint=args.endpoint)
                )
                break
                
        game.story_append_beat(narration.text)
        box.tts_say(narration.text)
        print(narration.text)
        msg, failure = game.handle_consequences(narration)
        print(msg)
        
        if failure == FailureState.Breakdown:
            game.score_entry.score_bonus -= 5
            breakdown_msg = box.new(
                Message, game.prompt_consequences_stress_breakdown()
            ).text
            game.story_append_beat(breakdown_msg)
            print(breakdown_msg)
            box.tts_say(breakdown_msg, interrupt=False)
            
            if random.randint(1, 4) == 1:
                metamorphosis_dialog(game, box)
        elif failure == FailureState.GameOver:
            break

    goodbye = box.new(
        Message,
        game.prompt_game_over(msg),
    ).text
    
    game.story_append_beat(goodbye)
    print(goodbye)
    box.tts_say(goodbye, interrupt=False)
    
    final_entry = game.get_final_score_entry(box, final_reason=msg)
    game.adventure_scenario.high_scores.append(final_entry)
    
    scenario_path = args.scenario_file if args.scenario_file else None
    saved_file = game.adventure_scenario.save(filepath=scenario_path)
    print(f"\nScore saved to {saved_file}!")
    
    print_scoreboard(game.adventure_scenario)
    input()

if __name__ == "__main__":
    main()

    
