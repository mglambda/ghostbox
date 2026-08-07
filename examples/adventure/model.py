# model.py
from pydantic import BaseModel, ValidationError, Field, model_validator
from enum import Enum, StrEnum
import json
import sys
from typing import *
from collections import Counter
import  json, argparse, random, os
import traceback
from utility import shorten_name


T = TypeVar('T', bound='BaseCondition')            
MAX_HP = 20
MAX_STRESS = 20



class CombatAbility(BaseModel):
    name: str = Field(description="A short, evocative name for the special ability.")
    description: str = Field(description="Visual and mechanical description. Does it burn, stun, or just emotionally damage the target?")
    ap_cost: int = Field(ge=1, le=3, description="Cost to use. normal impact abilities have 1 point cost, high impact is 2, 3 is reserved for legendary abilities.")


    def show(self) -> str:
        return f"{self.name} ({self.ap_cost} AP) - {self.description}"
    
class CombatComponent(BaseModel):
    # Anchor the LLM's vibe right at the top
    combat_style: str = Field(
        description="One sentence summarizing their combat approach (e.g., 'A cowardly opportunist who strikes from the shadows' or 'A relentless brute')."
    )
    primary_weapon: str = Field(
        description = "Weapon used if this character gets into a combat situation."
    )
    # Bounded AP stats so the AI doesn't completely lose its mind
    max_ap: int = Field(default=3, ge=3, le=3, description="3 for everyone")
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
            # AP display removed from long display for now - we just don't want it to show up in char creation because it's confusing
            #f"AP: {self.current_ap}/{self.max_ap} (Regen: {self.ap_regen})",
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
    
    def mod_ap(self, amount: int) -> str:
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


from pydantic import BaseModel, Field, model_validator
from typing import Optional


    
class PlayerCharacter(BaseModel):
    name: str
    gender: str
    character_class: str
    description: str
    motivation: str
    special_abilities: List[SpecialAbility]
    max_health: int = Field(ge=1, le=MAX_HP)
    health: int = Field(default = 1, description = "Current health of the character. Should be equal to max_health on generation.")
    max_stress: int = Field(ge=1, le=MAX_STRESS)
    stress: int = 0
    level: int = 1
    combat_component: CombatComponent


    @model_validator(mode='after')
    def sync_health_default(self) -> 'PlayerCharacter':
        """If health wasn't explicitly passed, sync it to max_health."""
        if self.health < self.max_health:
            self.health = self.max_health
        return self

    
    def mod_health(self, amount: int) -> str:
        old_health = self.health
        self.health = max(0, min(self.max_health, self.health + amount))
        actual_change = self.health - old_health
        
        if actual_change == 0:
            return f"{self.name}'s health remains unchanged, much like their tragic existence."
            
        verb = "gains" if actual_change > 0 else "loses"
        return f"{self.name} {verb} {abs(actual_change)} health."

    def mod_stress(self, amount: int) -> str:
        old_stress = self.stress
        self.stress = max(0, min(self.max_stress, self.stress + amount))
        actual_change = self.stress - old_stress
        
        if actual_change == 0:
            return f"{self.name}'s stress is unfazed."
            
        verb = "gains" if actual_change > 0 else "loses"
        return f"{self.name} {verb} {abs(actual_change)} stress."
    

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
    """Character choose to do an attack using their primary weapon."""
    action_type: Literal["attack"] = "attack"
    target_id: str = Field(description="ID of the entity to attack (e.g., 'e1', 'p2').")

    ap_cost: ClassVar[int] = 1
    
class DefaultChoice(BaseModel):
    """Character choose to spend their turn defensively."""
    action_type: Literal["default"] = "default"
    ap_cost: ClassVar[int] = 0

class AbilityChoice(BaseModel):
    """Character choose to use an ability."""
    action_type: Literal["ability"] = "ability"
    ability_id: str = Field(description="The exact ID of the ability to use (e.g., 'a1').")
    target_id: str = Field(description="ID of the entity to target (e.g., 'e1', 'p1').")
    # ap_cost has to be deduced from combat context
    
class FleeChoice(BaseModel):
    """Character flees combat. Usually followed by a FleeEffect."""
    action_type: Literal["flee"] = "flee"
    
    ap_cost: ClassVar[int] = 1
    
# The master union type. The LLM is forced to pick exactly one of these schemas.
AnyCombatChoice = Union[AttackChoice, DefaultChoice, AbilityChoice, FleeChoice]

class AICombatTurn(BaseModel):
    descriptive_text: str = Field(description = "A short, descriptive paragraph that establishes and telegraphs the enemy moves for this turn. Keep it flavorful and vague, include snarky one liners and villainous monologues if appropriate.")
    combat_actions: Dict[str, List[AnyCombatChoice]] = Field(description = "Maps enemy player character IDs to a list of combat moves they want to execute this turn. Max 4 actions per turn.")
    def show_debug(self, combat_state: 'CombatState') -> str:
        """Output that makes it easy to debug weird AI turns, as if debugging matters."""
        lines = ["--- AI TURN DEBUG (Brace for disappointment) ---"]
        
        if not self.combat_actions:
            lines.append("Literally doing nothing. Peak flop era.")
            return "\n".join(lines)
            
        for eid, actions in self.combat_actions.items():
            npc = combat_state.combatants.get(eid)
            # If the ID doesn't exist, we just name and shame the void
            name = npc.name if npc else "Unknown Ghost"
            
            if not actions:
                lines.append(f"[{eid}] {name}: Chose absolute stagnation. (0 actions)")
                continue
                
            # Yanking just the action types so your screen reader doesn't choke on the raw objects
            action_types = [getattr(a, 'action_type', 'unknown_garbage') for a in actions]
            lines.append(f"[{eid}] {name}: {', '.join(action_types)}")
            
        return "\n".join(lines)
    
        
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

class FleeEffect(BaseModel):
    effect_type: Literal["flee"] = "flee"
    target_id: str = Field(description="ID of the entity fleeing combat.")

# The master union type for effects.
AnyCombatEffect = Union[DamageEffect, HealEffect, StressEffect, FleeEffect]

class CombatResolution(BaseModel):
    """The LLM generates this for EVERY single action popped from the queue."""
    flavor_text: str = Field(
        description="A punchy, dramatic paragraph narrating the action. Make it sound devastating."
    )
    effects: List[AnyCombatEffect] = Field(
        description="The strictly mechanical puzzle pieces to apply to the game state."
    )


class CombatEndResult(StrEnum):
    players_win = "player_win"
    enemies_win = "enemies_win"
    players_fled = "players_fled"
    enemies_fled = "enemies_fled"

from pydantic import BaseModel, Field
from typing import Literal, Union


class BaseCondition(BaseModel):
    """The base class for all suffering."""
    # How many rounds this misery lasts before the universe grants sweet release
    duration: int = Field(default=1, ge=0)

class DodgeCondition(BaseCondition):
    """Under this condition, characters have a chance to evade attacks. This conditions is always removed at the end of the round. """
    condition_type: Literal["dodging"] = "dodging"
    dodge_chance: float = 0.5 

class FearCondition(BaseCondition):
    """Under this condition, characters have a chance to spontaneously flee."""
    condition_type: Literal["fear"] = "fear"
    # 50% chance they just pack it up and leave
    flee_chance: float = 0.5

class InvisibleCondition(BaseCondition):
    """Characters with this condition cannot be seen."""
    condition_type: Literal["invisible"] = "invisible"
    # 100% narrative. The LLM gets to hallucinate what this means. God help us.
    

AnyCondition = Annotated[
    Union[
        DodgeCondition, 
        FearCondition, 
        InvisibleCondition, 
    ], 
    Field(discriminator="condition_type")
]

# some condition groupings
# transient conditions fade at the end of the round
transient_conditions: List[Type[AnyCondition]] = [DodgeCondition]

class CombatantStatus(StrEnum):
    """Whether a combat is active, dead, has fled etc."""
    active = "active"
    dead = "dead"
    fled = "fled"


class SceneFeature(BaseModel):
    """An  environmental feature that is part of the current scene, e.g. a car in a city street, a tree in a forest, or an emergency medical kit on a spaceship."""
    name: str
    description: str

class Scene(BaseModel):
    """An environment in which a combat scene takes place."""
    name: str
    short_description: str
    features: Dict[str, SceneFeature] = Field(default_factory = dict, description = "Contains all features that are part of the scene, usually 2 or 3. Keys to the dictionary are the IDs of the scene features, e.g. 'f1', 'f2' and so on.")

    @staticmethod
    def default_scene() -> 'Scene':
        # we don't want to give default values to the fields above because it will influence hallucination
        # so we use this
        return Scene(name="White Room", short_description = "A white-walled, featureless room")
    
class CombatState(BaseModel):
    """The miserable sandbox where your characters go to die."""
    
    # The actual entity data, tracked by their temporary combat IDs
    combatants: Dict[str, PlayerCharacter] = Field(default_factory=dict)
    
    # Tracking which ID belongs to which team because iterating a dict is mid
    player_ids: List[str] = Field(default_factory=list)
    enemy_ids: List[str] = Field(default_factory=list)
    
    round_number: int = 1
    
    # We dump all the combat math in here as strings.
    # At the end of the fight, we feed this exact list to the LLM to write the summary.
    combat_log: List[str] = Field(default_factory=list)
    
    # Maps combatant ID to their queued choices for the current round
    action_queues: Dict[str, List[AnyCombatChoice]] = Field(default_factory=dict)

    # tracks enemies that have flown the scene
    fleeing_combatants: Set[str] = Field(default_factory = set)

    conditions: Dict[str, List[AnyCondition]] = Field(default_factory = dict, description = "Cotnains conditions for entities. Keys are character IDs.")
    
    lower_ap_bound: ClassVar[int] = -3
    upper_ap_bound: ClassVar[int] = 3
    action_queue_limit: ClassVar[int] = 4
    scene: Scene = Field(default_factory= Scene.default_scene)

    @staticmethod
    def setup(player_side: List['PlayerCharacter'], enemy_side: List['PlayerCharacter'], scene: Scene) -> 'CombatState':
        """Sets up the combat state and aggressively scrubs the LLM's hallucinated AP garbage."""
        state = CombatState()
        state.scene = scene
        
        # Populate players
        for i, pc in enumerate(player_side, 1):
            pid = f"p{i}"
            state.combatants[pid] = pc
            state.player_ids.append(pid)
            state.action_queues[pid] = []
            
            # Brutally reset AP to 0 so they stop starting fights with god-tier action economy
            # new: actually we start with 1 otherwise nobody can do anything. AP regen triggers at end of loop. this is adesign choice I swear
            if getattr(pc, 'combat_component', None):
                pc.combat_component.current_ap = 1
                
        # Populate enemies
        for i, npc in enumerate(enemy_side, 1):
            eid = f"e{i}"
            state.combatants[eid] = npc
            state.enemy_ids.append(eid)
            state.action_queues[eid] = []
            
            if getattr(npc, 'combat_component', None):
                npc.combat_component.current_ap = 1

            # FIXME: hotfix because the LLM generates NPCs with 1 HP for some reason
            if npc.health < npc.max_health:
                print(f"debug: fixing {npc.name} health from {npc.health} to max.")
                npc.health = npc.max_health
                
        return state

    
    def get_combatant_status(self, entity_id: str) -> CombatantStatus:
        """Returns the active, dead, or fled status of a combatant."""
        # so if we can't find it it's dead to us
        if (entity := self.combatants.get(entity_id)) is None:
            return CombatantStatus.dead

        if entity.health <= 0:
            return CombatantStatus.dead

        if entity_id in self.fleeing_combatants:
            return CombatantStatus.fled

        # in the future, we can check for more status effects here (like paralysis)

        return CombatantStatus.active

    def conditions_for(self, entity_id: str) -> List['AnyCondition']:
        """
        Dumps every single condition afflicting this entity.
        Mypy understands this perfectly because it's dead simple.
        """
        return self.conditions.get(entity_id, [])

    def specific_conditions_for(self, entity_id: str, condition_type: Type[T]) -> List[T]:
        """
        Fetches ONLY the specific condition you ask for.
        No Optional, no None. You MUST pass a type like DodgeCondition.
        """
        return [
            cond for cond in self.conditions.get(entity_id, []) 
            if isinstance(cond, condition_type)
        ]
    

    def has_condition(self, entity_id: str, condition_type: Type[T]) -> bool:
        return bool(self.specific_conditions_for(entity_id, condition_type))


    def give_condition(self, entity_id: str, condition: AnyCondition) -> None:
        """Bestows a condition upon a character."""
        if entity_id not in self.conditions:
            self.conditions[entity_id] = [condition]
            return
        self.conditions[entity_id].append(condition)

    def remove_condition(self, entity_id: str, condition_type: Type[T]) -> bool:
        """
        Purges every condition of the given type because nuance is dead.
        Returns True if we actually deleted something, False if it was a total waste of compute.
        """
        if entity_id not in self.conditions:
            return False
            
        original_garbage = self.conditions[entity_id]
        
        # Keep only the conditions that DO NOT match the type you're trying to evict.
        surviving_garbage = [
            cond for cond in original_garbage 
            if not isinstance(cond, condition_type)
        ]
        
        # If the length changed, congratulations, you actually removed something.
        if len(original_garbage) != len(surviving_garbage):
            self.conditions[entity_id] = surviving_garbage
            return True
            
        return False
    
    def maybe_winner(self) -> Optional[CombatEndResult]:
        """Checks if we can finally end this pointless digital suffering."""
        
        # You're only active if you have HP AND haven't run away
        players_active = any(self.combatants[pid].health > 0 and pid not in self.fleeing_combatants for pid in self.player_ids)
        enemies_active = any(self.combatants[eid].health > 0 and eid not in self.fleeing_combatants for eid in self.enemy_ids)

        if not players_active:
            # If no players are active, check if it's because they ran away like cowards
            if any(pid in self.fleeing_combatants for pid in self.player_ids):
                return CombatEndResult.players_fled
            return CombatEndResult.enemies_win
            
        if not enemies_active:
            if any(eid in self.fleeing_combatants for eid in self.enemy_ids):
                return CombatEndResult.enemies_fled
            return CombatEndResult.players_win
        
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
This is a check essentially only made at the top of the round.
        """
        if entity_id not in self.combatants:
            return False
            
        entity = self.combatants[entity_id]
        
        if entity.health <= 0:
            return False

        # have they flown?
        if entity_id in self.fleeing_combatants:
            return False
        
        # The ultimate vibe check. Are they in the negatives?
        return entity.combat_component.current_ap > 0

    def has_ap(self, entity_id: str) -> bool:
        """Checks if an entity has enough AP to take an action.
        This is checked e.g. immediately before their action is executed as they may have lost AP between queueing their action and it procuring."""
        if (entity := self.combatants.get(entity_id)) is None:
            return False
        return entity.combat_component.current_ap > CombatState.lower_ap_bound

    
    def is_alive(self, entity_id: str) -> bool:
        return self.get_combatant_status(entity_id) != CombatantStatus.dead

    def has_fled(self, entity_id: str) -> bool:
        """Returns true if an entity is currently fleeing the combat."""
        return self.get_combatant_status(entity_id) == CombatantStatus.fled
    
    def is_targetable(self, target_id: str) -> bool:
        """Returns true if an entity can be targeted with an attack or ability."""
        if (target := self.combatants.get(target_id)) is None:
            return False

        if target.health <= 0:
            return False

        if self.get_combatant_status(target_id) == CombatantStatus.fled:
            return False

        # future: check for invis
        
        return True

    def can_act(self, entity_id: str) -> bool:
        """Determines whether an entity can execute an action in combat. Takes various factors into account (death, fleeing, ap)."""
        return self.has_ap(entity_id) and self.is_alive(entity_id) and not self.has_fled(entity_id)
    
    def sanitize(self, ai_turn: 'AICombatTurn', debug: bool = False) -> 'AICombatTurn':
        """Brutally prunes AI hallucinations and mocks them for being overconfident."""
        modified = False
        pruned_actions = {}
        
        if debug:
            print("\n--- SANITIZE START: Praying the AI didn't completely ruin everything ---")
            
        for eid, actions in ai_turn.combat_actions.items():
            if debug:
                print(f"Checking entity ID: {eid}...")
            if eid not in self.enemy_ids:
                if debug: print(f"  ❌ Entity {eid} isn't even an enemy. AI is hallucinating ghosts. Skipped.")
                modified = True
                continue
                
            if not self.is_active(eid):
                if debug: print(f"  ❌ Entity {eid} is already dead or MIA. AI is trying to weekend-at-bernies them. Skipped.")
                modified = True
                continue
                
            current_ap = self.combatants[eid].combat_component.current_ap
            valid_actions = []
            
            if debug:
                print(f"  Entity {eid} starts with {current_ap} AP. Trying to queue {len(actions)} actions.")
                
            # We literally scan the entire queue for cowardice first.
            coward_action = next((a for a in actions if a.action_type in ("default", "flee")), None)
            
            if coward_action:
                valid_actions = [coward_action]
                if len(actions) > 1:
                    if debug: print(f"      🤡 AI tried to combo with {coward_action.action_type}. Invalidating their entire queue and forcing cowardice. Snipping the rest.")
                    modified = True
                else:
                    if debug: print(f"      ✔️ Clean single {coward_action.action_type}. Acceptable cowardice.")
                
                pruned_actions[eid] = valid_actions
                continue
                
            # If they aren't cowards, we actually do the math. 
            for i, action in enumerate(actions):
                action_type = action.action_type
                if debug:
                    print(f"    Action {i+1}: {action_type}")
                
                if isinstance(action, AbilityChoice):
                    try:
                        cost = self.abilities_for(eid)[action.ability_id].ap_cost
                        if debug: print(f"      ✨ Ability {action.ability_id} found. Cost: {cost} AP.")
                    except KeyError:
                        # AI hallucinated a bad ID
                        cost = 1
                        if debug: print(f"      💀 AI hallucinated ability ID '{getattr(action, 'ability_id', 'UNKNOWN')}'. Charging 1 AP idiot tax.")
                else:
                    cost = getattr(action, 'ap_cost', 1)
                    if debug: print(f"      Basic action cost: {cost} AP.")
                    
                # The AP bank declines their card.
                if current_ap - cost < -3:
                    if debug: print(f"      📉 Bankrupt! {current_ap} AP minus {cost} violates the -3 debt limit. Action denied.")
                    modified = True
                    break
                    
                # Stop the 5+ action spam.
                if len(valid_actions) >= 4:
                    if debug: print(f"      🛑 Action spam detected. Hitting the 4-action cap.")
                    modified = True
                    break
                    
                current_ap -= cost
                valid_actions.append(action)
                if debug: print(f"      ✔️ Action approved. AP drops to {current_ap}.")
                
            if valid_actions:
                pruned_actions[eid] = valid_actions
                
        ai_turn.combat_actions = pruned_actions
        if debug:
            print(f"--- SANITIZE COMPLETE. Modified: {modified}. It is all still meaningless anyway. ---\n")
            
        return ai_turn
    
    def next_round(self) -> None:
        """Advance the round and increase AP etc. Do housekeeping."""
        self.round_number += 1
        for cid, c in self.combatants.items():
            if self.is_alive(cid) and (self.get_combatant_status(cid) != CombatantStatus.fled):
                c.combat_component.mod_ap(c.combat_component.ap_regen)
                # these conditions only last 1 round
                for transient_condition in transient_conditions:
                    self.remove_condition(cid, transient_condition)

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


    def drain_action_queues(self) -> List['AnyCombatEvent']:
        """
        Drains the queues for the anime super combo initiative system.
        Cowards go first, then we randomize character order and dump their entire combo.
        """
        drained_events: List['AnyCombatEvent'] = []
        
        # Pass 1: The Cowards. Rip all 'default' actions out of everyone's queues first.
        for eid in self.player_ids + self.enemy_ids:
            if not self.is_active(eid):
                continue
            
            queue = self.action_queues.get(eid, [])
            
            # Separate the defaults from the actual actions
            defaults = [action for action in queue if getattr(action, 'action_type', '') == 'default']
            non_defaults = [action for action in queue if getattr(action, 'action_type', '') != 'default']
            
            for d in defaults:
                drained_events.append(CombatChoiceEvent(source_id=eid, choice=d))
                
            # Leave only the non-defaults in the queue for the next pass
            self.action_queues[eid] = non_defaults
            
        # Pass 2: The Anime Combos. Get all active IDs, shuffle them for initiative, and drain.
        active_entities = [eid for eid in self.player_ids + self.enemy_ids if self.is_active(eid)]
        random.shuffle(active_entities)
        
        for eid in active_entities:
            queue = self.action_queues.get(eid, [])
            for action in queue:
                drained_events.append(CombatChoiceEvent(source_id=eid, choice=action))
            
            # We drained them, so clear their queue completely
            self.action_queues[eid] = []
            
        return drained_events
    
    def show_status(self, debug: bool = False) -> str:
        """
        Dumps a quick summary of the battlefield so you can watch your impending doom in real-time,
        or literally spits out the entire raw JSON if you want to stare into the matrix.
        """
        if debug:
            return self.model_dump_json(indent=4)
            
        lines = ["--- ENEMIES ---"]
        for eid in self.enemy_ids:
            enemy = self.combatants.get(eid)
            # Skip the dead ones, they don't matter anymore.
            if not enemy or enemy.health <= 0:
                continue
                
            weapon = getattr(enemy.combat_component, 'primary_weapon', 'literal garbage')
            bloodied = " (Bloodied)" if enemy.health < (enemy.max_health / 2) else ""
            lines.append(f"{enemy.name} wielding {weapon}{bloodied}")
            
        lines.append("")
        lines.append("--- PLAYERS ---")
        
        for pid in self.player_ids:
            player = self.combatants.get(pid)
            if not player:
                continue
                
            if player.health <= 0:
                lines.append(f"{player.name} is literally dead. RIP bozo.")
            else:
                lines.append(player.show_short())
                
        return "\n".join(lines)    


    def json_overview(self) -> str:
        """
        Dumps the combat state for the LLM. 
        Zero safety checks. We die like men.
        """
        overview = {}
        
        for eid, combatant in self.combatants.items():
            team = "Player Team" if eid in self.player_ids else "Enemy Team"
            
            # Using your basic little helper method
            status = self.get_combatant_status(eid).value
            
            # Raw-dogging the attributes because you hate safety
            stats: Dict[str, Any] = {
                "name": combatant.name,
                "character_class": combatant.character_class,
                "description": combatant.description,
                "motivation": combatant.motivation,
                "health": f"{combatant.health}/{combatant.max_health}",
                "stress": f"{combatant.stress}/{combatant.max_stress}",
                "status": status
            }
            
            if combatant.combat_component:
                cc = combatant.combat_component
                combat_stats = {
                    "combat_style": cc.combat_style,
                    "primary_weapon": cc.primary_weapon,
                    "ap": f"{cc.current_ap}/{cc.max_ap}",
                    "ap_regen": cc.ap_regen,
                }
                
                # Perfect little ID map for the AI to completely ignore later
                abilities_dict = {}
                for ability_id, ability in self.abilities_for(eid).items():
                    abilities_dict[ability_id] = {
                        "name": ability.name,
                        "description": ability.description,
                        "ap_cost": ability.ap_cost
                    }
                
                combat_stats["abilities"] = abilities_dict
                stats["combat_component"] = combat_stats
                
            overview[eid] = {
                "team": team,
                "stats": stats
            }
            
        # Returning just the combatants dictionary directly since scene_features got nixed.
        return json.dumps({"combatants": overview}, separators=(',', ':'))

# bunch of helper functions

    def can_dodge(self, entity_id: str) -> bool:
        """Returns true if a character has successfully dodged (an attack). This happens when they e.g. default with 50% chance on each attack they receive."""
        # so there is the case of having multiple dodge effects
        # we just sort out the highest one and stick with that
        if (dodge_conditions := self.specific_conditions_for(entity_id, DodgeCondition)) == []:
            return False
        dodge_chance = max([c.dodge_chance for c in dodge_conditions])
        return random.random() < dodge_chance
    
class CombatChoiceEvent(BaseModel):
    """The catalyst. The calm before the LLM hallucination."""
    event_type: Literal["choice"] = "choice"
    source_id: str = Field(description="Who is making the terrible decision.")
    choice: AnyCombatChoice

            
    def procure(self, combat_state: 'CombatState') -> Tuple[str, List['AnyCombatEvent']]:
        """
        Handles any non-LLM choice mechanics. 
        Returns msg, List[triggered_events]
        """
        # mechanical messages go into msg
        msgs: List[str] = []
        # actual effects go in here
        new_events: List[AnyCombatEvent] = []        
        source = combat_state.combatants.get(self.source_id)
        if not source:
            return "The void does nothing.", []
        name = shorten_name(source.name)


        # this  is the place to hook in guaranteed mechanical effects.
        # note that all these choices will be handled by an LLM that hallucinates appropriate effects and flavor                
        match self.choice:
            case DefaultChoice() as default_choice:
                cost = default_choice.ap_cost
                # defaulting gives 50% dodge chance
                combat_state.give_condition(
                    self.source_id,
                    DodgeCondition(dodge_chance=0.5)
                )
                msgs.append(f"🛡 {name} is dodging.")
            case FleeChoice() as flee_choice:
                # LLM doesn't like to generate a flee effect on flee choice, so we do the mechanical thing here
                # fortunately there is no downside to doing this twice
                combat_state.fleeing_combatants.add(self.source_id) 
                cost = flee_choice.ap_cost
                msgs.append(f"🐔 {name} cowers in fear and flees.")
            case AttackChoice() as attack_choice:
                cost = attack_choice.ap_cost
                if (target := combat_state.combatants.get(attack_choice.target_id)) is not None:
                    target_name = f" {shorten_name(target.name)}"
                else:
                    target_name = ""
                    
                if combat_state.can_dodge(attack_choice.target_id):
                                    msgs.append(f"🤷 {name} misses{target_name}.")
                else:
                    msgs.append(f"⚔ {name} attacks{target_name}.")
            case AbilityChoice() as ability_choice:
                ability_id = ability_choice.ability_id
                abilities = combat_state.abilities_for(self.source_id)
                if (ability := abilities.get(ability_id)) is not None:
                    msgs.append(f"✨ Uses their {ability.name} ability.")
                    cost = abilities[ability_id].ap_cost                    
                else:
                    # weird but ok
                    msgs.append(f"✨ Uses an unknown ability.")
                    cost = 1

        # Deduct the AP mechanically. Welcome to capitalism.
        ap_msg = source.combat_component.mod_ap((-1) * cost)
        # we just tack this on at the end to not spam too much
        if msgs:
            msgs[-1] += f" {ap_msg}"
        else:
            msgs.append(ap_msg)

        return "\n".join(msgs), new_events
    
class CombatEffectEvent(BaseModel):
    """The actual math. Ruins someone's day, and maybe their life."""
    event_type: Literal["effect"] = "effect"
    source_id: str = Field(description="Who caused this suffering (or 'The Void').")
    effect: 'AnyCombatEffect' # Forward reference

    def procure(self, combat_state: 'CombatState') -> Tuple[str, List['AnyCombatEvent']]:
        """
        Applies the math. If it kills them, casually spawns a Death Event.
        """
        source = combat_state.combatants.get(self.source_id)
        source_name = source.name if source else "The Void"
        
        target = combat_state.combatants.get(self.effect.target_id)
        if not target:
            return f"👻 {source_name} targets a ghost. Complete flop.", []
            
        was_alive = target.health > 0
        msg = ""
        
        match self.effect:
            case DamageEffect() as combat_damage_effect:
                target.mod_health(-self.effect.amount)
                msg = f"👊 {source_name} deals {self.effect.amount} damage to {target.name}. Now at {target.health}/{target.max_health} HP."
            case HealEffect() as combat_heal_effect:
                target.mod_health(self.effect.amount)
                msg = f"☤ {source_name} heals {target.name} for {self.effect.amount}. Now at {target.health}/{target.max_health} HP."
            case StressEffect() as combat_stress_effect:
                target.mod_stress(self.effect.amount)
                msg = f"⚠ {source_name} inflicts {self.effect.amount} stress on {target.name}. Now at {target.stress}/{target.max_stress} Stress."
            case FleeEffect() as combat_flee_effect:
                combat_state.fleeing_combatants.add(self.effect.target_id)
                if source:
                    msg = f"🪶 {source_name} causes {target.name} to flee."
                else:
                    msg = f"🪶 {target.name} flees the scene."
            case _ as unreachable:
                assert_never(unreachable)
                msg = f"🐦 Unknown effect. The simulation is actively breaking down."
                
        triggered_events: List[AnyCombatEvent] = []
        if was_alive and target.health <= 0:
            triggered_events.append(CombatDeathEvent(target_id=self.effect.target_id))
            
        return msg, triggered_events

class CombatDeathEvent(BaseModel):
    """The inevitable end. Truly a mood."""
    event_type: Literal["death"] = "death"
    target_id: str = Field(description="Who finally gets to log off from existence.")

    def procure(self, combat_state: 'CombatState') -> Tuple[str, List['AnyCombatEvent']]:
        """Announces their failure to the universe."""
        target = combat_state.combatants.get(self.target_id)
        if not target:
            return "", []

        
        return f"💀 {target.name} has expired. Their existence is now completely irrelevant.", []

# The master union type. Must be at the bottom so it can see the classes.
AnyCombatEvent = Union[CombatChoiceEvent, CombatEffectEvent, CombatDeathEvent]
