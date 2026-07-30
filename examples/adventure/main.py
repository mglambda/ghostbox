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
    health: int
    stress: int = 0
    intro_done: bool = False
    tags: Counter = Field(default_factory = Counter)
    score_entry: ScoreEntry = Field(default_factory = ScoreEntry)
    story: List[str] = Field(default_factory = list)
    latest_criticism: str = ""
    
    debug: bool = False
    tarot: bool = True

    _turn: int = 1

    def tags_add(self, new_tags: List[str]) -> None:
        """Adds tags to the internal counter."""
        print(f"debug: {', '.join([tag for tag in new_tags])}")
        self.tags.update([w.lower() for w in new_tags])

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
        n = self.player.level
        return min(base + ((n**2) // 2), 200)

    def turn_tick(self) -> None:
        """Triggers various random events each turn."""
        if random.randint(1, 10) == 10:
            print(f"You feel yourself taking a deep breath.")
            self.gain_stress(-1)
        if random.randint(1, 20) == 20:
            print(f"You feel your wounds stitch together somewhat.")
            self.gain_health(1)
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
        
        lvl_str = f"lvl: {self.player.level}"
        health_str = f"Health: {self.health}/{self.player.max_health}"
        stress_str = f"Stress: {self.stress}/{self.player.max_stress}"
        score_str = f"Score: {self.score_entry.total_score()}"

        return (
            f"{health_str} {stress_str} Fate: {self.fate} {lvl_str} {score_str}\tSpecial: {ability_str}"
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
        
        failure = FailureState.NoFailure

        if self.stress > self.player.max_stress:
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

    def prompt_main_choices(self, history: List[ghostbox.ChatMessage]) -> str:
        n = random.randint(3, 4)
        tarot_msg = ""
        if self.tarot:
            k = random.randint(1, 5)
            if k == 1:
                card = draw_tarot_cards(1)[0]
                tarot_msg = f"\nGenerate one additional choice that is subtly inspired by the following tarot card: {card}. Please tag this choice with 'tarot'."
        
        return f"""Generate {n} dramatic choices for the main character, along with a brief summary of the situation. {tarot_msg}"""

    def prompt_consequences_special_ability(self, special: SpecialAbility) -> str:
        return f"""The player has used the following ability:
            ```{special.name} - {special.description}```
Please narrate the outcome of using this ability in this situation, or gently remind the player that this ability cannot be used, if it is not at all applicable to the current situation."""

    def prompt_consequences(
            self, choice: Choice, history: List[ghostbox.ChatMessage], endpoint = "http://localhost:8080"
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
            
                self.latest_criticism = f"\n\nBelow is some helpful criticism of the story so far. Implement it as best you can:\n```{advice}\n```"
            
            if True or self.debug:
                print("Critic's advice: \n" + advice + "\n## end advice\n")
        else:
            self.latest_criticism = ""
        
        prompt = (
            "The player has chosen the following: \n"
            + choice.show()
            + "\nPlease narrate the consequences of the players choice. Drive the story forward and lead into a new dramatic situation.\n"
        )

        return player_status_str + prompt 

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
    
    if random.randint(1, MAX_HP) > game.player.max_health:
        game.player.max_health += 1
        game.health += 1
        print("Your maximum health has increased by 1.")
    if random.randint(1, MAX_STRESS) > game.player.max_stress:
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
                "pc_health": str(game.health),
                "pc_stress": str(game.stress),
                "story_str": game.story_get_str(),
                "latest_criticism": game.latest_criticism,
            }
        )
        box.clear_history()
        
        if not (game.intro_done):
            intro = box.text(game.prompt_intro())
            game.story_append_beat(intro)
            print("## Intro\n" + intro)
            box.tts_say(intro, interrupt=False)
            game.intro_done = True
            
        situation = box.new(Situation, game.prompt_main_choices(box.get_history()))
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
                    DialogChoice(selection_string="advance", value="advance"),
                    DialogChoice(selection_string="q", value="q"),
                ],
                after=game.status(),
                prompt=f" or use an ability (type name or initial letter). Typing `*` spends 3 fate to write your own choice. Ask the GM a question with `?`. Type `q` to save and quit.\n{game.player.name} > ",
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
                
            if choice == "advance" and game.fate >= game.advancement_fate_required():
                print("You have advanced your abilities!")
                advancement_dialog(game, box)
                print("Done with advancement. Let's return to the story.")
                continue
                
            if type(choice) == SpecialAbility:
                special, msg = game.try_use_special_ability(choice.name)
                if special is None:
                    print(msg)
                    continue
                print(msg)
                narration = box.new(
                    Consequences, game.prompt_consequences_special_ability(special)
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
                    Consequences, game.prompt_consequences(choice, box.get_history(), endpoint=args.endpoint)
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

    
