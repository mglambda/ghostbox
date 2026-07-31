from pydantic import BaseModel, ValidationError, Field
from enum import Enum, StrEnum
import sys
from typing import *
from collections import Counter
from datetime import datetime
import ghostbox, json, argparse, random, os
import traceback

MAX_HP = 40
MAX_STRESS = 20



class CombatAbility(BaseModel):
    name: str = Field(description="A short, evocative name for the special ability.")
    description: str = Field(description="Visual and mechanical description. Does it burn, stun, or just emotionally damage the target?")
    ap_cost: int = Field(ge=4, le=10, description="Cost to use.")


    def show(self) -> str:
        return f"self.name ({self.ap_cost}) - {self.description}"
    
class CombatComponent(BaseModel):
    # Anchor the LLM's vibe right at the top
    combat_style: str = Field(
        description="One sentence summarizing their combat approach (e.g., 'A cowardly opportunist who strikes from the shadows' or 'A relentless brute')."
    )
    
    # Bounded AP stats so the AI doesn't completely lose its mind
    max_ap: int = Field(default=10, ge=5, le=15, description="Maximum Action Points. Usually 10, up to 15 for bosses.")
    ap_regen: int = Field(default=3, ge=1, le=6, description="AP regained per turn. 3 is standard, 6 is terrifying.")
    
    # Only the special, flavor-heavy moves go here
    abilities: List[CombatAbility] = Field(
        default_factory=list, 
        max_length=5, 
        description="Character-specific special moves. DO NOT include basic attacks or defending."
    )

    def gain_ap(self, amount: int) -> str:
    """Modifies AP and returns a string for the terminal UI because we love reading text."""
    if amount == 0:
        return f"AP unchanged. Stagnation is the default state of the universe. ({self.current_ap}/{self.max_ap})"
        
    old_ap = self.current_ap
    # The absolute lowest your AP can go before the game physically stops you.
    # Assuming max 4 actions at 3 AP each, minus your starting 3 AP.
    min_ap = -9 
    
    self.current_ap = max(min_ap, min(self.max_ap, self.current_ap + amount))
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

class PlayerCharacter(BaseModel):
    name: str
    gender: str
    character_class: str
    description: str
    motivation: str
    special_abilities: List[SpecialAbility]
    max_health: int = Field(ge=1, le=MAX_HP)
    max_stress: int = Field(ge=1, le=MAX_STRESS)
    level: int = 1

    def show(pc, indent: str = "", include_special_abilities: bool =True) -> str:
        w = ""
        w += pc.name + "\n"
        w += indent + pc.description.replace("\n", "\n" + indent) + "\n"
        w += indent + "Class: " + pc.character_class + "\n"
        w += indent + f"Max Health: {pc.max_health}; Max Stress: {pc.max_stress}\n"
        w += indent + "Motivation: " + pc.motivation + "\n"
        if include_special_abilities:
            w += indent + "Special Abilities" + "\n"
            for special in pc.special_abilities:
                w += (
                    2 * indent
                    + " - "
                    + special.name
                    + ". "
                    + special.description
                    + f"({special.fate_cost} fate)"
                    + "\n"
                )
        return w + "\n"

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


