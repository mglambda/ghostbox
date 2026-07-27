#!/usr/bin/env python
from pydantic import BaseModel, ValidationError, Field
from enum import Enum
from typing import *
from datetime import datetime
import ghostbox, json, argparse, random, os
import traceback


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

MAX_HP = 40
MAX_STRESS = 20

# some utility for presenting dialog choices

A = TypeVar("A")



# The complete standard 78-card Tarot deck
TAROT_DECK: Tuple[str, ...] = (
    # Major Arcana (22)
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
    "Judgement", "The World",
    
    # Minor Arcana - Wands (14)
    "Ace of Wands", "Two of Wands", "Three of Wands", "Four of Wands", "Five of Wands",
    "Six of Wands", "Seven of Wands", "Eight of Wands", "Nine of Wands", "Ten of Wands",
    "Page of Wands", "Knight of Wands", "Queen of Wands", "King of Wands",
    
    # Minor Arcana - Cups (14)
    "Ace of Cups", "Two of Cups", "Three of Cups", "Four of Cups", "Five of Cups",
    "Six of Cups", "Seven of Cups", "Eight of Cups", "Nine of Cups", "Ten of Cups",
    "Page of Cups", "Knight of Cups", "Queen of Cups", "King of Cups",
    
    # Minor Arcana - Swords (14)
    "Ace of Swords", "Two of Swords", "Three of Swords", "Four of Swords", "Five of Swords",
    "Six of Swords", "Seven of Swords", "Eight of Swords", "Nine of Swords", "Ten of Swords",
    "Page of Swords", "Knight of Swords", "Queen of Swords", "King of Swords",
    
    # Minor Arcana - Pentacles (14)
    "Ace of Pentacles", "Two of Pentacles", "Three of Pentacles", "Four of Pentacles", "Five of Pentacles",
    "Six of Pentacles", "Seven of Pentacles", "Eight of Pentacles", "Nine of Pentacles", "Ten of Pentacles",
    "Page of Pentacles", "Knight of Pentacles", "Queen of Pentacles", "King of Pentacles"
)


def draw_tarot_cards(n: int = 1) -> Tuple[str, ...]:
    """
    Draws `n` unique Tarot cards without replacement.
    If `n` exceeds the total size of the deck (78), full reset decks are 
    shuffled and appended as needed to satisfy the requested count.
    """
    if n <= 0:
        return ()

    deck_size = len(TAROT_DECK)
    drawn_cards = []

    while n > 0:
        # Determine how many cards to draw from the current fresh deck cycle
        draw_count = min(n, deck_size)
        
        # sample without replacement handles exact uniform distribution
        drawn_cards.extend(random.sample(TAROT_DECK, draw_count))
        
        n -= draw_count

    return tuple(drawn_cards)

class DialogChoice(BaseModel):
    text: str = ""
    selection_string: Optional[str] = None
    value: A | Callable[[], A]


def choose_dialog(
    choices: List[DialogChoice],
    before: str = "",
    after: str = "",
    prompt: Optional[str] = None,
    indent: int = 4,
    fuzzy: bool = True,
    reprint_on_newline: bool = True,
    exit_on_newline: bool = False,
    show_numbered_selection_string: bool = True,
    show_extra_selection_strings: bool = True,
    on_error: Optional[Callable[[str], None]] = None,
    print_function: Callable[[str], None] = print,
    input_function: Callable[[str], str] = input,
) -> A:
    from functools import reduce

    # some setup
    print, input = print_function, input_function
    numbered_choices, extra_choices_list = reduce(
        lambda pair, c: (
            (pair[0] + [c], pair[1])
            if c.selection_string is None
            else (pair[0], pair[1] + [c])
        ),
        choices,
        ([], []),
    )
    extra_choices = {
        (
            extra.selection_string.strip().lower() if fuzzy else extra.selection_string
        ): extra
        for extra in extra_choices_list
    }

    def value_or_call(x: A | Callable[[], A]) -> A:
        if callable(x):
            return x()
        return x

    while True:
        if before:
            print(before)
        for i in range(len(numbered_choices)):
            choice = choices[i]
            text = choice.text if choice.text else str(choice.value)
            print((indent * " ") + f"({i+1}) {text}")

        if after:
            print(after)
        choice_str = (
            f"Enter a number (1 - {len(numbered_choices)})"
            if show_numbered_selection_string
            else ""
        )
        extra_str = (
            " or type " + ", ".join([extra_key for extra_key in extra_choices.keys()])
            if show_extra_selection_strings and (extra_choices)
            else ""
        )
        prompt_str = prompt if prompt is not None else ":"
        while True:
            w = input(choice_str + extra_str + prompt_str)
            if fuzzy:
                w = w.strip().lower()

            if w == "":
                if exit_on_newline:
                    return None
                elif reprint_on_newline:
                    break

            # numbered choices override extra choices
            if w.isdigit():
                try:
                    choice = numbered_choices[int(w) - 1]
                except:
                    continue
                return value_or_call(choice.value)

            if not (fuzzy):
                # exact matching, the easy case
                if w in extra_choices.keys():
                    return value_or_call(extra_choices[w].value)
            else:
                # fuzzy matching
                for key in extra_choices.keys():
                    if key.startswith(w):
                        return value_or_call(extra_choices[key].value)

            # at this point it was neither an extra key or a digit
            # we consider this an error
            if on_error is not None:
                # on_error doesn't return anything, but may raise here, so user can exit the loop
                on_error(w)


# data model


class SpecialAbility(BaseModel):
    """A special ability that is usable by a player character during play. Its fate cost should reflect its power to influence the story, with higher impact abilities costing more fate. The description should not refer to game mechanics."""

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
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


    def total_score(self) -> int:
        """Calculates total score based on the member fields."""
        total = 0
        total = 5 * self.turns_survived
        total += 10 * self.total_fate_earned
        total += 100 * self.level_ups
        total += self.score_bonus
        return total
    
class Scenario(BaseModel):
    """A fleshed out adventure scenario, with instructions for a game Master, world building notes, and style guidance."""

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
    high_scores: List[ScoreEntry] = []
    

    def show(self):
        w = ""
        for k, v in self.model_dump().items():
            if k == "high_scores":
                continue  # Skip high scores in the general text dump
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

    def save(self, filepath: Optional[str] = None) -> str:
        """Saves the scenario to a file. Returns the filename used."""
        if filepath and os.path.isfile(filepath):
            filename_candidate = filepath
        else:
            filename_candidate = self.name.lower().replace(" ", "_") + ".json"
            while os.path.isfile(filename_candidate) or os.path.isdir(filename_candidate):
                filename_candidate = (
                    self.name.lower().replace(" ", "_") + f"_{random.randint(1, 1024)}.json"
                )
        with open(filename_candidate, "w") as f:
            f.write(json.dumps(self.model_dump(), indent=4))
        return filename_candidate
    

def print_scoreboard(scenario: Scenario):
    if not scenario.high_scores:
        print("\n--- NO PREVIOUS RECORDED DEATHS IN THIS SCENARIO ---")
        return

    # Sort descending by score
    sorted_scores = sorted(scenario.high_scores, key=lambda s: s.score(), reverse=True)

    print("\n========================================================")
    print(f"       HALL OF FAME / GRAVEYARD: {scenario.name.upper()}")
    print("========================================================")
    print(f"{'RANK':<5} {'NAME':<15} {'SCORE':<8} {'TURNS':<6} {'CAUSE OF DEATH'}")
    print("-" * 65)

    for rank, entry in enumerate(sorted_scores[:10], 1): # Top 10
        print(
            f"{rank:<5} {entry.player_name[:14]:<15} {entry.total_score():<8} "
            f"{entry.turns_survived:<6} {entry.cause_of_death[:25]}"
        )
    print("========================================================\n")    

class Choice(BaseModel):
    "A short text describing a player's possible action in a dramatic situation, from their perspective."

    text: str
    is_dangerous: bool
    is_part_of_player_motivation: bool

    def fate(self) -> int:
        """Returns the amount of fate points this choice is worth."""
        fate = 0
        if self.is_dangerous:
            fate += 1

        if self.is_part_of_player_motivation:
            fate += 1

        return fate

    def show(self) -> str:
        w = ""
        # we used to show these but it's actually more fun if you don't know what gives you fate
        # danger = "*danger* " if self.is_dangerous else ""
        # motivation = "*fate* " if self.is_part_of_player_motivation else ""
        # w += danger + motivation + self.text
        w += self.text
        return w


FailureState = Enum("FailureState", "NoFailure Breakdown GameOver")


class Consequences(BaseModel):
    """Narration of the consequences to a choice or ability use. May include stress gain or health loss if applicable."""

    text: str
    stress_gained: int
    stress_lost: int
    health_gained: int
    health_lost: int


class GameState(BaseModel):
    player: PlayerCharacter
    party: List[PlayerCharacter]
    adventure_scenario: Scenario
    fate: int = 1
    health: int
    stress: int = 0
    score_entry: ScoreEntry = Field(default_factory = ScoreEntry)
    debug: bool = False
    tarot: bool = True

    _turn: int = 1

    def get_final_score_entry(self, box, final_reason: str) -> ScoreEntry:
        """
        Populates metadata and generates a concise cause of death string 
        using Ghostbox and game history.
        """
        # Ensure character specs are populated
        self.score_entry.player_name = self.player.name
        self.score_entry.character_class = self.player.character_class

        # Construct prompt for the LLM to distill the history into a quick cause-of-death epitaph

        try:
            cause_of_death_msg = box.text(self.prompt_final_death_reason(final_reason)).strip()
            # Clean up potential extra quotes or markdown fences
            cause_of_death = cause_of_death_msg.strip('"`')
        except Exception as e:
            if self.debug:
                print(f"Error generating cause of death: {e}")
            cause_of_death = "Unknown."

        self.score_entry.cause_of_death = cause_of_death
        return self.score_entry


    def prompt_final_death_reason(self, final_reason: str) -> str:
        return f"""A player character has met their end in an adventure. "
Character Name: {self.player.name} ({self.player.character_class}).\n

The final mechanical reason for their demise is the following:
```
{final_reason}
```
        
In 1 short sentence (under 12 words), summarize the exact narrative cause of their death or madness. 
Example: 'Eaten by a shadow-stalker in the dark' or 'Succumbed to eldritch insanity'.
"""


                
    def gain_fate(self, amount: int) -> str:
        """Gain a certain amount of fate, which may be negative. Returns a message indicating fate amount gained, or empty string if 0 fate is gained."""
        # new and experimental: randomly double fate gained
        if random.randint(1,20) == 20:
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
        old_hp = self.health
        self.health = min(self.health + hp, self.player.max_health)
        new_hp = self.health
        if new_hp > old_hp:
            return f"You gained {new_hp - old_hp} health."
        if new_hp < old_hp:
            return f"You lost {old_hp - new_hp} health."
        return ""

    def gain_stress(self, stress) -> str:
        old_stress = self.stress
        self.stress = max(self.stress + stress, 0)
        new_stress = self.stress
        if new_stress > old_stress:
            return f"You gained {new_stress - old_stress} stress."
        if new_stress < old_stress:
            return f"You lost {old_stress - new_stress} stress."
        return ""

    def advancement_fate_required(self) -> int:
        """Returns the number of fate points required to level up and advance."""
        base = 3
        n = len(self.player.special_abilities) - 1
        return min(base + ((n**2) // 2), 200)

    def turn_tick(self) -> None:
        """Triggers various random events each turn."""
        # 10% chance to reduce small amount of stress
        if random.randint(1, 10) == 10:
            print(f"You feel yourself taking a deep breath.")
            self.gain_stress(-1)

        # 5% chance to recover 1 health
        if random.randint(1, 20) == 20:
            print(f"You feel your wounds stitch together somewhat.")
            self.gain_health(1)

        # 5% chance to gain 1 fate randomly
        if random.randint(1, 20) == 20:
            print(f"Fortune smiles upon you.")
            self.gain_fate(1)
            
            
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

        health_str = f"Health: {self.health}/{self.player.max_health}"
        stress_str = f"Stress: {self.stress}/{self.player.max_stress}"
        return (
            f"{health_str} {stress_str} Fate: {self.fate}\tSpecial: {ability_str}"
            + advancement
        )

    def handle_consequences(
        self, consequences: Consequences
    ) -> Tuple[str, FailureState]:
        """Takes a consequence object, applies it to the current state, and then returns a pair of a message and a bool indicating if the game is over."""
        self._turn += 1
        ws = [
            self.gain_health(
                (-1 * consequences.health_lost) + consequences.health_gained
            ),
            self.gain_stress(
                consequences.stress_gained + (-1 * consequences.stress_lost)
            ),
        ]

        # by default, nothing bad happens
        failure = FailureState.NoFailure

        if self.stress > self.player.max_stress:
            # this is only a soft failure
            # if we can dump the stress into health, pc only panics/breaks down
            # new: we only dump a quarter
            stress_value = self.stress // 4
            if self.health >= stress_value:
                self.health -= stress_value

                ws.append(
                    f"You break down from stress! Your mental breakdown takes a toll on your body, and you lose {stress_value} health."
                )
                self.stress = 0
                ws.append("You have narrowly averted permanent insanity.")
                failure = FailureState.Breakdown
            else:
                # can't dump the stress
                ws.append(
                    f"Due to stress and trauma, {self.player.name} loses their mind completely."
                )
                failure = FailureState.GameOver

        if self.health <= 0:
            ws.append(f"{self.player.name} dies from their wounds.")
            failure = FailureState.GameOver

        return "\n".join(ws), failure

    def try_use_special_ability(
        self, name_shorthand: str
    ) -> Tuple[Optional[SpecialAbility], str]:
        """Attempts to use a player's special ability, based on a shorthand name, and subtracts fate accordingly.
        :param name_shorthand: May be the full ability name, or a prefix thereof. If no ability is found, this method will fail and return none and error.
        :return: On success, returns ability and empty string, on failure, returns None and an error message.
        """
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

        # all good
        return special, self.gain_fate(-1 * special.fate_cost)

    # the following prompt_* methods are generators for the main prompts send to the llm
    # it's nice to have them in one place and
    # it also allows us to vary them based on various conditions, since the gamestate has access to
    # pretty much all game state
    def prompt_intro(self) -> str:
        """The message printed only once at the start of the adventure."""
        return "Write a short introductory paragraph to the adventure that sets the scene. Make sure it leads directly into a dramatic situation, and the goals and stakes are clear. Adhere to the scenario's style guide, and use the sources of inspiration for guidance. This will be the first thing the player hears when they start the adventure, so make sure it really pops."

    def prompt_main_choices(self, history: List[ghostbox.ChatMessage]) -> str:
        """Called when the LLM is supposed to generate choices, which happens in the main loop."""
        # we want 3 or 4 choices
        n = random.randint(3, 4)
        
        tarot_msg = ""
        if self.tarot:
            # if tarot is enabled, we occasionally draw a tarot card to seed an additional choice that is inspired by the cards symbology
            k = random.randint(1, 5)
            if k == 1:
                card = draw_tarot_cards(1)[0]
                tarot_msg = f"\nGenerate one additional choice that is subtly inspired by the following tarot card: {card}. This choice should award at least 1 fate."


            
        return f"""Generate {n} dramatic choices for the main character, along with a brief summary of the situation. These choices don't cost fate.{tarot_msg}"""

    def prompt_consequences_special_ability(self, special: SpecialAbility) -> str:
        """Called when the player used a special ability and the LLM is supposed to generate consequences based on it and the current situation."""
        return f"""The player has used the following ability:
            ```
{special.name} - {special.description}
```

Please narrate the outcome of using this ability in this situation, or gently remind the player that this ability cannot be used, if it is not at all applicable to the current situation."""


    def prompt_consequences(
            self, choice: Choice, history: List[ghostbox.ChatMessage], endpoint = "http://localhost:8080"
    ) -> str:
        """Called when the player made a choice and the LLM is supposed to generate consequences based on it and the current situation, hopefully leading into another situation with interesting choices."""

        # the vars in braces are set in the main loop with box.set_vars.
        # we could also inject them here, but setting them in one place ensures consistency across prompts
        player_status_str = "Current player status: {{pc_health}} health, {{pc_stress}} stress, {{fate}} fate.\n"
        # so, it turns out most LLMs are so aligned and cooperative, if they know the player has high stress/health, they will not damage them further
        # so it's actually important to to keep that info from them
        #player_status_str = ""

        if self._turn % 3 == 0:
            # every 3 turns, we invoke the GMs inner critic
            critic = ghostbox.from_generic(
                endpoint = endpoint,
                character_folder="critic", **(default_options | {"tts": False})
            )
            # the critic gets to look at the story so far, but without the sometimes enormous system prompt
            # they are a literary critic, not a game master
            prompt = (
                "A game master and a player are playing a role playing game. Here is their story so far:\n\n```\n"
                + "\n".join([msg.content for msg in history
                             if msg.role == "assistant"])
                + "\n```\n\nPlease criticise the story so far, and give helpful advice on how to improve it, and where to steer it next."
            )
            # the critic uses slightly different settings from the ddefaults
            # most importantly, we don't want it to invalidate the cache
            # though that's only relevant if we are running a local LLM
            with critic.options(
                temperature=0.3, samplers=["min_p", "temperature"], cache_prompt=False
            ):
                advice = critic.text(prompt)

            if True or self.debug:
                # temporarily short circuited because we always want to see critic thoughts
                print("Critic's advice: \n" + advice)
        else:
            # otherwise we just have some good general principles
            advice = "Keep it brief and focused, but evocative. Do not generate new choices as part of this response. Be sure to adhere to the scenario's style guide in your narration."

        prompt = (
            "The player has chosen the following: \n"
            + choice.show()
            + "\nPlease narrate the consequences of the players choice. Drive the story forward and lead into a new dramatic situation.\n"
        )
        return player_status_str + prompt + advice

        # return "Briefly describe the situation to the player, including the current location, characters present, and the most relevant details. Give them some dramatic choices. Choices are always from the players perspective. Do not include the consequences in the choice text. Do not mention fate points. Try to include a mix of choices, and take the scenario, players, and history into account. Do not include special abilities in choices, the player will activate those seperately. Likewise, avoid mentioning the other party members in the choices. Do not list the choices in the description text. Remember that choices that are dangerous or involve the players motivation let them earn fate, so be sparing with those.",

    def prompt_consequences_stress_breakdown(self) -> str:
        """Called when stress reaches >= maximum stress for a character, and they suffer a momentary mental breakdown. This is asoft failure, not a game over."""
        return f"{{game.player.name}} has incurred too much stress and sufffers a momentary mental breakdown! Please narrate the consequences of {{game.player.name}} breaking down, losing consciousness, having a panic attack, or temporarily losing their sanity."

    def prompt_game_over(self, msg) -> str:
        """Happens when player dies from lack of health or goes insane because stress can't be vented off anymore."""
        return f"""The game is over for the player. Reason: "
```
{msg}
```
        
Please write a suitable goodbye narration to send them off."""


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


# when using structured output with the .json or .new methods
# many models fail to go back to outputting regular text, putting json into everything
# using this as a wrapper prevents the json from spilling out into regular text
class Message(BaseModel):
    text: str


# dialog functions


def scenario_creation_dialog(endpoint = "http://localhost:8080", initial_prompt="") -> Scenario:
    box = ghostbox.from_generic(endpoint=endpoint, character_folder="scenario_creator", **default_options)
    hint = initial_prompt
    chosen_scenario = None
    while chosen_scenario is None:
        print("Generating scenario drafts...")
        drafts = box.new(
            ScenarioDrafts,
            "Create a handful of interesting adventure scenarios. Present both fantasy and sci-fi options, and give a variety of tones and styles, with both dark and light hearted themes being explored. The description should be short and pithy, something that hooks and entices a potential player."
            + "\n" + hint,
        ).drafts

        def set_hint(w):
            # this will be called by choose_dialog when user enters something that isn't a number
            nonlocal hint
            hint = w
            # we throw just to exit the choice loop
            # the flag is so we don't capture other exceptions
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
            if e.flag:
                # hint was set
                continue
            raise e

    # at this point we have a chosen scenario
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
            "Here is an adventure scenario: "
            + scenario.show()
            + "\n\nCreate a handful of player characters that would fit this scenario.",
        ).player_characters

        def set_hint(w):
            nonlocal hint
            hint = w
            e = Exception()
            e.flag = True
            raise e

        try:
            chosen_player = choose_dialog(
                [DialogChoice(text=pc.show(), value=pc) for pc in pcs],
                before="Choose a player character!",
                prompt=" or enter a suggestion to regenerate characters.: ",
                on_error=set_hint,
            )
        except Exception as e:
            if e.flag:
                continue

    # at this point we have a chosen player
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
    """Happens when player chooses to level up."""
    # deduct the level up fate cost
    print(game.gain_fate(-1 * game.advancement_fate_required()))

    # max hp and max stress advance through roll-over
    if random.randint(1, MAX_HP) > game.player.max_health:
        game.player.max_health += 1
        print("Your maximum health has increased by 1.")

    if random.randint(1, MAX_STRESS) > game.player.max_stress:
        game.player.max_stress += 1
        print("Your maximum stress has increased by 1.")

    # there is a 1 in 6 chance that we have a special level up
    if random.randint(1, 6) == 6:
        print("***Special Advancement***")
        hint = input("You may suggest something for your new abilities: ")
    else:
        hint = ""

    class NewSpecialAbilities(BaseModel):
        """A handful of abilities, one of which the player may choose for their level up."""

        special_ability_choices: List[SpecialAbility]

    new_abilities = box.new(
        NewSpecialAbilities,
        "Generate a handful of new special abilities the player may choose from for their advancement. Make sure to take their character, the adventure, and the story so far into account. Give a variety of choices. Focus on things the player cannot do yet. Do not generate abilities the player already has."
        + hint,
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

    maybe_drop_i = choose_dialog(
        [
            DialogChoice(
                text=f"{game.player.special_abilities[i].name}: {game.player.special_abilities[i].description}",
                value=i,
            )
            for i in range(len(game.player.special_abilities))
        ],
        before="You can choose to drop one of your special abilities.",
        prompt=" or hit enter to proceed without dropping: ",
        exit_on_newline=True,
    )

    if (drop_i := maybe_drop_i) is not None:
        print(f"You lose {game.player.special_abilities[drop_i]}.")
        del game.player.special_abilities[drop_i]


def question_dialog(game, box) -> str:
    """Happens when the player asks the GM a question with ?. Expect lots of soft hacking with this one."""
    w = input(
        "Question to the GM (information, clarification, visual description, etc): "
    )
    if not (w):
        return ""

    return box.new(
        Message,
        "Answer the following player question. Be informative and descriptive only, don't give away secrets or advance the story.\nQuestion: "
        + w,
    ).text


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
        scenario = scenario_creation_dialog(endpoint=args.endpoint, initial_prompt=args.scenario_prompt)
        if args.save_scenario:
            filename = scenario.save()
            print(
                f"Scenario has been saved as {filename}. You can replay it with `-f {filename}`."
            )
    else:
        try:
            with open(args.scenario_file, "r") as f:
                scenario = Scenario(**json.loads(f.read()))
        except ValidationError as e:
            print(str(e))
            return
        except:
            if args.debug:
                print(traceback.format_exc())
            print(f"Error: Couldn't load scenario file: {args.scenario_file}")
            return

    print(scenario.show())
    pc, others = player_creation_dialog(scenario, endpoint=args.endpoint, party=args.party)
    game = GameState(
        player=pc,
        party=others,
        adventure_scenario=scenario,
        fate=1,
        health=pc.max_health,
        debug=args.debug,
    )
    run(game, args)


def run(game, args):
    box = ghostbox.from_generic(endpoint=args.endpoint, character_folder="game_master", **default_options)

    # this is the main loop
    narration = ""
    intro_done = False
    while True:
        game.score_entry.turns_survived += 1
        # this makes things like {{scenario}} or {{pc_health}} expand into their respective values in both the system_msg and
        # prompts that we use in box.new below
        box.set_vars(
            {
                "scenario": game.adventure_scenario.show(),
                "party": "\n".join([npc.show(include_special_abilities=False) for npc in game.party]),
                "pc": game.player.show(include_special_abilities=False),
                "fate": str(game.fate),
                "pc_health": str(game.health),
                "pc_stress": str(game.stress),
            }
        )

        if not (intro_done):
            # give an intro message that sets the scene
            intro = box.new(Message, game.prompt_intro()).text
            print(intro)
            box.tts_say(intro, interrupt=False)
            intro_done = True

        situation = box.new(Situation, game.prompt_main_choices(box.get_history()))
        print("\n" + situation.show() + "\n")
        box.tts_say(situation.brief_description, interrupt=False)

        # we loop until we have narration for the consequences
        # in the loop player may do a bunch of stuff, but using an ability or making a choice will break it
        while True:
            # debug
            if args.debug:
                print(json.dumps([msg.model_dump() for msg in box.get_history()], indent=4))

            # type of choice is Optional[str | Choice | SpecialAbility]
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
                    DialogChoice(selection_string="advance", value="advance"),
                ],
                after=game.status(),
                prompt=f" or use an ability (type name or initial letter). Typing `*` spends 3 fate to write your own choice. Ask the GM a question with `?`.\n{game.player.name} > ",
                show_extra_selection_strings=False,
                exit_on_newline=True
            )
            box.tts_stop()
            
            if choice is None:
                # player just hit enter. this let's us just stop the tts, which we did above
                # we just reprint
                continue


            if choice == "*":
                # player gets to write their own
                if game.fate >= 3:
                    print(game.gain_fate(-3))
                    player_text = input("Your choice: ")
                    choice = Choice(
                        text=player_text,
                        is_dangerous=False,
                        is_part_of_player_motivation=False,
                    )
                else:
                    print("Insufficient fate!")
                    continue

            if choice == "?":
                if msg := question_dialog(game, box):
                    print(msg)
                    box.tts_say(msg, interrupt=False)
                continue

            if choice == "advance" and game.fate >= game.advancement_fate_required():
                print("You have advanced your abilities!")
                advancement_dialog(game, box)
                game.score_entry.level_ups += 1
                print("Done with advancement. Let's return to the story.")
                continue
            if type(choice) == SpecialAbility:
                # ability use
                special, msg = game.try_use_special_ability(choice.name)
                if special is None:
                    print(msg)
                    continue
                # fate was deducted and ability should be used
                print(msg)
                narration = box.new(
                    Consequences, game.prompt_consequences_special_ability(special)
                )
                break
            else:
                # at this point, choice is a Choice-> player picked one of the options
        print(f"## Turn {game.score_entry.turns_survived}")
        self.game.tick_turn()
                
                fate_msg = game.gain_fate(choice.fate())
                print(fate_msg + "\n" if fate_msg else "" + "Please wait...")
                narration = box.new(
                    Consequences, game.prompt_consequences(choice, box.get_history(), endpoint=args.endpoint)
                )
                break

        # we have narration/consequences of choice or ability use
        box.tts_say(narration.text)
        print(narration.text)
        msg, failure = game.handle_consequences(narration)
        print(msg)
        if failure == FailureState.Breakdown:
            # this is only a soft failure
            # it will influence the story, but shouldn't incur more penalties to the player, so they can have a chance to recover
            game.score_entry.score_bonus -= 5
            breakdown_msg = box.new(
                Message, game.prompt_consequences_stress_breakdown()
            ).text
            print(breakdown_msg)
            box.tts_say(breakdown_msg, interrupt=False)
        elif failure == FailureState.GameOver:
            break
        # there is also FailureState.NoFailure, which we just ignore and proceed

# Game over handling
    goodbye = box.new(
        Message,
        game.prompt_game_over(msg),
    ).text
    print(goodbye)
    box.tts_say(goodbye, interrupt=False)

    # Process High Score & Leaderboard
    final_entry = game.get_final_score_entry(box, final_reason=msg)
    game.adventure_scenario.high_scores.append(final_entry)
    
    # Save back to disk
    scenario_path = args.scenario_file if args.scenario_file else None
    saved_file = game.adventure_scenario.save(filepath=scenario_path)
    print(f"\nScore saved to {saved_file}!")
    
    # Print NetHack Graveyard
    print_scoreboard(game.adventure_scenario)        
    input()


if __name__ == "__main__":
    main()
