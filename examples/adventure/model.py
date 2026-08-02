from pydantic import BaseModel, ValidationError, Field
from enum import Enum, StrEnum
import sys
from typing import *
from collections import Counter
from datetime import datetime
import  json, argparse, random, os
import traceback

MAX_HP = 20
MAX_STRESS = 20



class CombatAbility(BaseModel):
    name: str = Field(description="A short, evocative name for the special ability.")
    description: str = Field(description="Visual and mechanical description. Does it burn, stun, or just emotionally damage the target?")
    ap_cost: int = Field(ge=1, le=3, description="Cost to use. normal impact abilities have 1 point cost, high impact is 2, 3 is reserved for legendary abilities.")


    def show(self) -> str:
        return f"{self.name} ({self.ap_cost}) - {self.description}"
    
class CombatComponent(BaseModel):
    # Anchor the LLM's vibe right at the top
    combat_style: str = Field(
        description="One sentence summarizing their combat approach (e.g., 'A cowardly opportunist who strikes from the shadows' or 'A relentless brute')."
    )
    primary_weapon: str = Field(
        description = "Weapon used if this character gets into a combat situation."
    )
    # Bounded AP stats so the AI doesn't completely lose its mind
    max_ap: int = Field(default=10, ge=5, le=15, description="Maximum Action Points. Usually 10, up to 15 for bosses.")
    ap_regen: int = Field(default=1, ge=1, le=3, description="AP regained per turn. 1 is standard, 2 is terrifying.")
    current_ap: int = Field(default=3, description="Current Action Points.")
    
    # Only the special, flavor-heavy moves go here
    combat_abilities: List[CombatAbility] = Field(
        default_factory=list, 
        max_length=5, 
        description="Character-specific special moves. DO NOT include basic attacks or defending."
    )




    def show_short(self) -> str:
        """One-liner for the initiative tracker. Minimal brainpower required."""
        return f"AP: {self.current_ap}/{self.max_ap} | Style: {self.combat_style}"

    def show(self) -> str:
        """Full character sheet dump for combat."""
        lines = [
            f"AP: {self.current_ap}/{self.max_ap} (Regen: {self.ap_regen})",
            f"Combat Style: {self.combat_style}",
            f"Primary Weapon: {self.primary_weapon}",
            "Techniques:"
        ]
        
        if not self.combat_abilities:
            lines.append("  - None. Literally useless.")
        else:
            for ability in self.combat_abilities:
                # Assuming you fixed the broken f-string in CombatAbility.show()
                # to return f"{self.name} ({self.ap_cost} AP) - {self.description}"
                lines.append(f"  - {ability.show()}")
                
        return "\n".join(lines)
    
    def gain_ap(self, amount: int) -> str:
        """Modifies AP and returns a string for the terminal UI because we love reading text."""
        if amount == 0:
            return f"AP unchanged. Stagnation is the default state of the universe. ({self.current_ap}/{self.max_ap})"

        old_ap = self.current_ap
        self.current_ap = max(CombatState.lower_ap_bound, min(self.max_ap, self.current_ap + amount))
        actual_change = self.current_ap - old_ap

        if actual_change > 0:
            return f"Regained {actual_change} AP. (Current: {self.current_ap}/{self.max_ap})"
        elif actual_change < 0:
            return f"Burned {abs(actual_change)} AP. (Current: {self.current_ap}/{self.max_ap})"
        else:
            return f"AP is literally capped out or at rock bottom. Nothing matters. (Current: {self.current_ap}/{self.max_ap})"    

    
class SpecialAbility(BaseModel):
    """A special ability that is usable by a player character during play. Its fate cost should reflect its power to influence the story, with higher impact abilities costing more fate. The description should not refer to game mechanics, as it will be interpreted and applied by an LLM."""

    name: str
    description: str
    fate_cost: int = Field(ge=1, le=6)

    def show(self) -> str:
        return f"{self.name} ({self.fate_cost} fate) - {self.description}"
    
class PlayerCharacter(BaseModel):
    name: str
    gender: str
    character_class: str
    description: str
    motivation: str
    special_abilities: List[SpecialAbility]
    max_health: int = Field(ge=1, le=MAX_HP)
    health: int = 1
    max_stress: int = Field(ge=1, le=MAX_STRESS)
    stress: int = 0
    level: int = 1
    combat_component: CombatComponent

    
    def health_mod(self, amount: int) -> None:
        """Modify health while erspecting min and max hp."""
        self.health = min(self.max_health, max(0, self.health + amount))

    def stress_mod(self, amount: int) -> None:
        """Modify stress while respecting min and max stress."""
        self.stress = min(self.max_stress, max(0, self.stress + amount))

    def show_short(self) -> str:
        """One-liner for the party status screen. Keeping it brief because our attention spans are literally fried."""
        base = f"Lv.{self.level} {self.character_class} '{self.name}' | HP: {self.health}/{self.max_health} | Stress: {self.stress}/{self.max_stress}"
        
        if self.combat_component:
            # Slaps the combat component's short string right on the end
            return f"{base} | {self.combat_component.show_short()}"
        return f"{base} | Combat: None (Literally useless)"    
    
    def show(self, indent: str = "", include_special_abilities: bool = True, include_combat_abilities: bool = False) -> str:
        """Dumps the player sheet. Prepare to be disappointed by their stats."""
        # We use a list to collect lines because string concatenation in a loop is literal garbage.
        lines = []
        
        lines.append(f"{self.name} ({self.gender})")
        
        # Handle multi-line descriptions without breaking the indent like a total noob
        desc = self.description.replace("\n", f"\n{indent}")
        lines.append(f"{indent}{desc}")
        
        lines.append(f"{indent}Class: {self.character_class}")
        lines.append(f"{indent}Health: {self.health} / {self.max_health}; tress: {self.stress} / {self.max_stress}")
        lines.append(f"{indent}Motivation: {self.motivation}")
        
        if include_special_abilities:
            lines.append(f"{indent}Special Abilities:")
            if not self.special_abilities:
                lines.append(f"{indent * 2}- None. Because being special is a myth.")
            else:
                for special in self.special_abilities:
                    # Using the recursive show method you explicitly asked for
                    lines.append(f"{indent * 2}- {special.show()}")
                    
        if include_combat_abilities and self.combat_component:
            lines.append(f"{indent}Combat Profile:")
            # Grab the combat dump, then indent every single line of it so it aligns
            combat_dump = self.combat_component.show()
            indented_combat = combat_dump.replace("\n", f"\n{indent * 2}")
            lines.append(f"{indent * 2}{indented_combat}")
            
        return "\n".join(lines) + "\n"

class ScenarioDraft(BaseModel):
    """A draft for an adventure scenario."""
    name: str
    description: str

class ScenarioDrafts(BaseModel):
    drafts: List[ScenarioDraft]

class ImportantCharacter(BaseModel):
    name: str
    description: str

class ImportantPlace(BaseModel):
    name: str
    description: str

class ImportantEvent(BaseModel):
    name: str
    description: str

class ImportantFaction(BaseModel):
    name: str
    description: str

class ScoreEntry(BaseModel):
    player_name: str = Field(default="John Doe")
    character_class: str = Field(default="Tourist")
    cause_of_death: Optional[str] = None
    turns_survived: int = 0
    total_fate_earned: int = 0
    level_ups: int = 0
    score_bonus: int = 0
    star_uses: int = 0
    tarot_chosen: int = 0
    unique_tags_collected: int = 0
    narrative_transitions: int = 0
    win_ending: bool = False
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    def total_score(self) -> int:
        """Calculates total score based on the member fields."""
        total = 0
        
        total += 10 * self.total_fate_earned
        total += 100 * self.level_ups
        total += self.score_bonus
        # random crap
        total += 25 * self.narrative_transitions
        total -= 10 * self.star_uses
        total += 5 * self.tarot_chosen
        total += 3 * self.unique_tags_collected
        
        # we modify score based on floor and ceiling of turns
        # this is to avoid degenerate 3 turn strategies
        if self.turns_survived <= 3:
            total *= 0.2
        elif self.turns_survived <= 5:
            total *= 0.3
        else:
            total += min(self.turns_survived * 5, 250)
        
        # a scenario win catapults you into anothe rtier via x10
        if self.win_ending:
            total *= 10
        
        return int(total)


class Scenario(BaseModel):
    """A fleshed out adventure scenario, with instructions for a game Master, world building notes, and style guidance."""

    # this class is supposed to contain fields that are generated by an LLM.
    name: str
    description: str
    inspired_by: List[str]
    game_master_instructions: str
    style_guide: str
    world_calendar_and_timekeeping: str
    unique_world_feature: str
    typical_sayings_and_idioms: List[str]
    important_places: List[ImportantPlace]
    important_past_world_events: List[ImportantEvent]
    important_factions: List[ImportantFaction]
    important_characters: List[ImportantCharacter]
    
    def show(self):
        w = ""
        for k, v in self.model_dump().items():
            k_str = k.capitalize().replace("_", " ")
            if isinstance(v, list):
                w += "\n# " + k_str + "\n"
                for item in v:
                    if isinstance(item, str):
                        w += " - " + item + "\n"
                    elif isinstance(item, dict) and "name" in item:
                        w += " - " + item["name"] + ": " + item.get("description", "") + "\n"
            else:
                w += "\n# " + k_str + "\n\n" + str(v) + "\n"

        return w

class ScenarioFile(BaseModel):
    """Wrapper class that encapsulates a scenario along with metadata (e.g. high scores)."""
    scenario: Scenario
    critic_system_prompt: str = "You are a literary critic. You analyse stories and narrative works for their quality. You are ruthless in spotting tired tropes, stereotypes, bad writing, tiresome and repetetive narration, unengaging ideas, confusing story structure, aimless drivel, and many other problems in writing.\nYou love stories that are action packed, tight, engaging, and economical. You are accepting of poetic description and purple prose, as long as it serves a purpose and is used sparingly. Although you are familiar with and bored of all literary tropes, you understand their purpose and necessity. In a story, the most important thing for you is that it makes the reader feel something.\nWhen you give advice, you are ruthless and unsparing. You do not waste time with praise. You know that the bitter truth inevitably serves to improve the writer and bring the best out of a story. However, you always give tips on how to improve and where to steer the story  next."
    high_scores: List[ScoreEntry] = []
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Saves the scenario to a file. Returns the filename used."""
        if filepath and os.path.isfile(filepath):
            filename_candidate = filepath
        else:
            filename_candidate = self.scenario.name.lower().replace(" ", "_") + ".json"
            while os.path.isfile(filename_candidate) or os.path.isdir(filename_candidate):
                filename_candidate = (
                    self.scenario.name.lower().replace(" ", "_") + f"_{random.randint(1, 1024)}.json"
                )

        with open(filename_candidate, "w") as f:
            f.write(json.dumps(self.model_dump(), indent=4))
        return filename_candidate


class Choice(BaseModel):
    "A short text describing a player's possible action in a dramatic situation, from their perspective."
    text: str
    is_dangerous: bool
    is_part_of_player_motivation: bool
    initiates_combat: bool
    tags: List[str]  

    def has_tag(self, tag: str) -> bool:
        for w in self.tags:
            if w.lower() == tag.lower():
                return True
        return False
        
    def fate(self) -> int:
        """Returns the amount of fate points this choice is worth."""
        fate = 0
        if self.is_dangerous:
            fate += 1
        if self.is_part_of_player_motivation:
            fate += 1
        if self.has_tag("tarot"):
            fate += 1

        return fate

    def show(self) -> str:
        w = ""
        if self.initiates_combat:
            w += "[Combat] "
        w += self.text
        return w

FailureState = Enum("FailureState", "NoFailure Breakdown GameOver")

class Mode(StrEnum):
    action = "action"
    exploration = "exploration"
    investigation = "investigation"
    montage = "montage"
    dialog = "dialog"
    reflection = "reflection"
    downtime = "downtime"
    
    def description(self) -> str:
        match self:
            case Mode.action:
                return "High-stakes, fast-paced situations requiring immediate physical reaction, evasion, or survival against active threats or environmental hazards."
            case Mode.exploration:
                return "Moving through and observing the environment, establishing geography, atmosphere, and discovering broad points of interest."
            case Mode.investigation:
                return "Focused, detailed examination of a specific area, object, or puzzle to uncover hidden information, clues, or hidden mechanics."
            case Mode.montage:
                return "A time-compressed narrative sequence summarizing travel, routine tasks, or training, rapidly advancing the timeline."
            case Mode.dialog:
                return "Character-driven interaction focused heavily on conversation, negotiation, interrogation, or relationship-building."
            case Mode.reflection:
                return "Introspective moments focusing on the player character's internal thoughts, emotional state, or processing of recent narrative events."
            case Mode.downtime:
                return "A period of rest and preparation in a safe environment, allowing for recovery, inventory management, and planning."
            case _ as unreachable:
                assert_never(unreachable)


                
class Consequences(BaseModel):
    """Narration of the consequences to a choice or ability use."""

    current_narrative_mode: Mode
    text: str
    stress_gained: int
    stress_lost: int
    health_gained: int
    health_lost: int





class Situation(BaseModel):
    brief_description: str
    current_location: str
    characters_present: List[str]
    choices: List[Choice]

    def show(self) -> str:
        return (
            f"Location: {self.current_location}\nPresent: "
            + ", ".join(self.characters_present)
            + "\n\n"
            + self.brief_description
        )

class Message(BaseModel):
    text: str

# Combat subsystem
# --- The Sum Types for the AI (and Player Queue) ---

class AttackChoice(BaseModel):
    action_type: Literal["attack"] = "attack"
    target_id: str = Field(description="ID of the entity to attack (e.g., 'e1', 'p2').")

class DefaultChoice(BaseModel):
    action_type: Literal["default"] = "default"


class AbilityChoice(BaseModel):
    action_type: Literal["ability"] = "ability"
    ability_id: str = Field(description="The exact ID of the ability to use (e.g., 'a1').")
    target_id: str = Field(description="ID of the entity to target (e.g., 'e1', 'p1').")

class FleeChoice(BaseModel):
    action_type: Literal["flee"] = "flee"

# The master union type. The LLM is forced to pick exactly one of these schemas.
AnyCombatChoice = Union[AttackChoice, DefaultChoice, AbilityChoice, FleeChoice]

class AICombatTurn(BaseModel):
    descriptive_text: str = Field(description = "A short, descriptive paragraph that establishes and telegraphs the enemy moves for this turn. Keep it flavorful and vague, include snarky one liners and villainous monologues if appropriate.")
    combat_actions: Dict[str, List[AnyCombatChoice]] = Field(description = "Maps enemy player character IDs to a list of combat moves they want to execute this turn. Max 4 actions per turn.")
    


# --- The Combat State ---


# --- The Atomic Lego Blocks for Resolving Actions ---

class DamageEffect(BaseModel):
    effect_type: Literal["damage"] = "damage"
    target_id: str = Field(description="ID of the poor soul taking damage (e.g., 'e1', 'p2').")
    amount: int = Field(default=1, ge=0, description="Amount of HP to violently remove.")

class HealEffect(BaseModel):
    effect_type: Literal["heal"] = "heal"
    target_id: str = Field(description="ID of the entity getting a temporary reprieve from death.")
    amount: int = Field(default=1, ge=0, description="Amount of HP to restore.")

class StressEffect(BaseModel):
    effect_type: Literal["stress"] = "stress"
    target_id: str = Field(description="ID of the entity having a mental breakdown.")
    amount: int = Field(description="Amount of stress to add (positive number). Can be negative to relieve stress.")

# The master union type for effects.
AnyCombatEffect = Union[DamageEffect, HealEffect, StressEffect]

class CombatResolution(BaseModel):
    """The LLM generates this for EVERY single action popped from the queue."""
    flavor_text: str = Field(
        description="A punchy, dramatic paragraph narrating the action. Make it sound devastating."
    )
    effects: List[AnyCombatEffect] = Field(
        description="The strictly mechanical puzzle pieces to apply to the game state."
    )
    
class CombatState(BaseModel):
    """The miserable sandbox where your characters go to die."""
    
    # The actual entity data, tracked by their temporary combat IDs
    combatants: Dict[str, 'PlayerCharacter'] = Field(default_factory=dict)
    
    # Tracking which ID belongs to which team because iterating a dict is mid
    player_ids: List[str] = Field(default_factory=list)
    enemy_ids: List[str] = Field(default_factory=list)
    
    round_number: int = 1
    
    # We dump all the combat math in here as strings.
    # At the end of the fight, we feed this exact list to the LLM to write the summary.
    combat_log: List[str] = Field(default_factory=list)
    
    # Maps combatant ID to their queued choices for the current round
    action_queues: Dict[str, List[AnyCombatChoice]] = Field(default_factory=dict)

    lower_ap_bound: ClassVar[int] = -3
    upper_ap_bound: ClassVar[int] = 3
    action_queue_limit: ClassVar[int] = 4
    
    @staticmethod
    def setup(player_side: List['PlayerCharacter'], enemy_side: List['PlayerCharacter']) -> 'CombatState':
        """Sets up the combat state and assigns temporary IDs because proper game dev is too hard for us."""
        state = CombatState()
        
        # Populate players
        for i, pc in enumerate(player_side, 1):
            pid = f"p{i}"
            state.combatants[pid] = pc
            state.player_ids.append(pid)
            state.action_queues[pid] = []
            
        # Populate enemies
        for i, npc in enumerate(enemy_side, 1):
            eid = f"e{i}"
            state.combatants[eid] = npc
            state.enemy_ids.append(eid)
            state.action_queues[eid] = []
            
        return state


def maybe_winner(self) -> Optional[Literal["players", "enemies"]]:
        """Checks if we can finally end this pointless digital suffering."""
        players_alive = any(self.combatants[pid].health > 0 for pid in self.player_ids)
        enemies_alive = any(self.combatants[eid].health > 0 for eid in self.enemy_ids)

        # If everyone is dead, enemies win by default because the universe hates you.
        if not players_alive:
            return "enemies"
        if not enemies_alive:
            return "players"
            
        return None

    def queue_action(self, entity_id: str, action: AnyCombatChoice) -> Tuple[bool, str]:
        """Queues an action, assuming you haven't already bungled the queue length."""
        if entity_id not in self.combatants:
            return False, f"Entity {entity_id} doesn't even exist. Massive L."
            
        queue = self.action_queues.get(entity_id, [])
        
        # Hardcapping at 4 because of the Brave system.
        if len(queue) >= 4:
            return False, f"Queue is full. {self.combatants[entity_id].name} cannot act more than 4 times."
            
        self.action_queues[entity_id].append(action)
        return True, f"Successfully queued {action.action_type} for {self.combatants[entity_id].name}."

    def abilities_for(self, entity_id: str) -> Dict[str, 'CombatAbility']:
        """
        Maps 'a1', 'a2' etc. to the actual CombatAbility objects so the LLM 
        doesn't completely hallucinate random moves.
        """
        if entity_id not in self.combatants:
            return {}
            
        entity = self.combatants[entity_id]
        
        # If this entity doesn't have a combat component or it's empty, return zip.
        if not hasattr(entity, 'combat_component') or not entity.combat_component:
            return {}
            
        # Dynamically generate the a1, a2 mapping based on their current loadout
        return {
            f"a{i+1}": ability 
            for i, ability in enumerate(entity.combat_component.combat_abilities)
        }

    def is_active(self, entity_id: str) -> bool:
        """
        Checks if an entity actually exists, has a pulse, and isn't paralyzed by debt.
        Because checking this anywhere else was apparently a crime against architecture.
        """
        if entity_id not in self.combatants:
            return False
            
        entity = self.combatants[entity_id]
        
        if entity.health <= 0:
            return False
        if not getattr(entity, 'combat_component', None):
            return False
            
        # The ultimate vibe check. Are they in the negatives?
        return entity.combat_component.current_ap >= 0    

    def sanitize(self, ai_turn: 'AICombatTurn') -> 'AICombatTurn':
        """Brutally prunes AI hallucinations and mocks them for being overconfident."""
        
        modified = False
        pruned_actions = {}
        
        for eid, actions in ai_turn.combat_actions.items():
            # Using the new state method so the code is *aesthetic*
            if eid not in self.enemy_ids or not self.is_active(eid):
                modified = True
                continue
            
            enemy = self.combatants[eid]
            current_ap = enemy.combat_component.current_ap
            valid_actions = []
            
            for action in actions:
                # Max 4 actions rule. 
                if len(valid_actions) >= 4:
                    modified = True
                    break
                
                # Defaulting is free. 
                cost = 0 if getattr(action, 'action_type', '') == "default" else getattr(action, 'ap_cost', 1)
                
                if current_ap - cost < -3:
                    modified = True
                    break # Stop processing actions, they are broke.
                    
                current_ap -= cost
                valid_actions.append(action)
                
            if valid_actions:
                pruned_actions[eid] = valid_actions
                
        # Overwrite the hallucinated garbage 
        ai_turn.combat_actions = pruned_actions
        
        # If we had to fix their math, drag them.
        if modified:
            snark = " (However, the villains were completely delulu and vastly overestimated their own stamina, stumbling mid-attack and dropping their combos like absolute clowns.)"
            ai_turn.descriptive_text += snark
            
        return ai_turn    
    
    def next_round(self) -> None:
        """Advance the round and increase AP etc. Do housekeeping."""
        self.round_number += 1
        for _, c in self.combatants.items():
            c.gain_ap(c.ap_regen)

    def apply_turn(self, ai_turn: 'AICombatTurn') -> None:
        """
        Takes the freshly sanitized hallucinations of the AI and actually 
        puts them into the queue. Wow. Groundbreaking.
        """
        for eid, actions in ai_turn.combat_actions.items():
            if self.is_active(eid):
                # Overwrite or extend? Let's just overwrite for safety, 
                # assuming the AI plans its whole turn at once.
                self.action_queues[eid] = actions            

    def pop_action(self) -> Optional[Tuple[str, AnyCombatChoice]]:
        """
        Pops the next action from the queues. Defaults get priority because 
        turtling up to delay the inevitable is the only valid response to existence.
        """
        # Pass 1: Look for cowards (Defaults) at the front of ANY queue
        for eid, queue in self.action_queues.items():
            if queue and getattr(queue[0], 'action_type', '') == 'default':
                return eid, queue.pop(0)
                
        # Pass 2: Literally whatever else is left in the order we iterate
        for eid, queue in self.action_queues.items():
            if queue:
                return eid, queue.pop(0)
                
        # The queues are empty. We are free.
        return None                

