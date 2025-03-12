#!/usr/bin/env python
from pydantic import BaseModel, ValidationError, Field
from typing import *
import ghostbox, json, argparse, random, os

default_options = {
    "quiet": True,
    "stderr": False,
    "max_context_length": 32000,
    "max_length": -1,
    "tts": True,
    "tts_model": "kokoro",
    "tts_voice": "af_sky",
    "temperature": 0.6,
    "samplers": ["min_p", "dry", "xtc", "temperature"],
}

MAX_HP = 40
MAX_STRESS = 20

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

    def show(pc, indent: str = "") -> str:
        w = ""
        w += pc.name + "\n"
        w += indent + pc.description.replace("\n", "\n" + indent) + "\n"
        w += indent + "Class: " + pc.character_class + "\n"
        w += indent + f"Max Health: {pc.max_health}; Max Stress: {pc.max_stress}\n"
        w += indent + "Motivation: " + pc.motivation + "\n"
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

    def show(self):
        w = ""
        for k, v in self.model_dump().items():
            k_str = k.capitalize().replace("_", " ")
            if type(v) == type([]):
                w += "\n# " + k_str + "\n"
                for item in v:
                    if type(item) == str:
                        w += " - " + item + "\n"
                    else:
                        # by convention, this is a dict with name and description properties
                        w += " - " + item["name"] + ": " + item["description"] + "\n"

            else:
                w += "\n# " + k_str + "\n\n" + v + "\n"
        return w

    def save(self) -> str:
        """Saves the scenario to a file. Returns the filename."""

        filename_candidate = self.name.lower().replace(" ", "_") + ".json"
        while os.path.isfile(filename_candidate) or os.path.isdir(filename_candidate):
            # we get json twice but i don't care
            filename_candidate = (
                filename_candidate + str(random.randint(1, 1024)) + ".json"
            )

        # we just crash if this doesn't work
        with open(filename_candidate, "w") as f:
            f.write(json.dumps(self.model_dump(), indent=4))

        return filename_candidate


class GameState(BaseModel):
    player: PlayerCharacter
    party: List[PlayerCharacter]
    adventure_scenario: Scenario
    fate: int = 1
    health: int
    stress: int = 0

    def gain_fate(self, amount: int) -> str:
        """Gain a certain amount of fate, which may be negative. Returns a message indicating fate amount gained, or empty string if 0 fate is gained."""
        self.fate += amount
        if amount > 0:
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
        self.stress = max(min(self.stress + stress, self.player.max_stress), 0)
        new_stress = self.stress
        if new_stress > old_stress:
            return f"You gained {new_stress - old_stress} stress."
        if new_stress < old_stress:
            return f"You lost {old_stress - new_stress} stress."
        return ""

    def advancement_fate_required(self) -> int:
        """Returns the number of fate points required to level up and advance."""
        base = 10
        n = len(self.player.special_abilities)
        return min(base + ((n**2) // 2), 200)

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

    def handle_consequences(self, consequences) -> Tuple[str, bool]:
        """Takes a consequence object, applies it to the current state, and then returns a pair of a message and a bool indicating if the game is over."""
        ws = [
            self.gain_health(
                1 + (-1 * consequences.health_lost) + consequences.health_gained
            ),
            self.gain_stress(
                consequences.stress_gained + (-1 * consequences.stress_lost)
            ),
        ]
        game_over = False

        if self.stress > self.player.max_health:
            ws.append(
                f"Due to stress and trauma, {self.player.name} loses their mind completely."
            )
            game_over = True

        if self.health <= 0:
            ws.append(f"{self.player.name} dies from their wounds.")
            game_over = True

        return "\n".join(ws), game_over

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
        danger = "*danger* " if self.is_dangerous else ""
        motivation = "*fate* " if self.is_part_of_player_motivation else ""
        w += danger + motivation + self.text
        return w


class Situation(BaseModel):
    description: str
    choices: List[Choice]

    def show_choices(self) -> str:
        w = ""
        for i in range(len(self.choices)):
            choice = self.choices[i]
            w += "\n(" + str(i + 1) + ") " + choice.show() + "\n"
        return w


class Consequences(BaseModel):
    """Narration of the consequences to a choice or ability use. May include stress gain or health loss if applicable.
    Health is only lost due to severe physical trauma or other bodily harm to the main character.
    Health is gained when the main character receives healing, rest, or medical care.
    Stress is gained only due to traumatic situations, complete loss of control, or supernatural fear.
    Stress is lost whenever the main character succeeds on a task, a difficult situation resolves, or the players get a moment of rest and respite.
    """

    text: str
    stress_gained: int
    stress_lost: int
    health_gained: int
    health_lost: int


# when using structured output with the .json or .new methods
# many models fail to go back to outputting regular text, putting json into everything
# using this as a wrapper prevents the json from spilling out into regular text
class Message(BaseModel):
    text: str


def scenario_creation_dialog(initial_prompt="") -> Scenario:
    box = ghostbox.from_generic(character_folder="scenario_creator", **default_options)
    hint = initial_prompt
    chosen_scenario = None
    while chosen_scenario is None:
        print("Generating scenario drafts...")
        drafts = box.new(
            ScenarioDrafts,
            "Create a handful of interesting adventure scenarios. Present both fantasy and sci-fi options, and give a variety of tones and styles, with both dark and light hearted themes being explored. The description should be short and pithy, something that hooks and entices a potential player."
            + hint,
        ).drafts
        print("Choose a scenario!\n")
        indent = "    "
        for i in range(len(drafts)):
            draft = drafts[i]
            print("(" + str(i + 1) + ") " + draft.name)
            print(indent + draft.description.replace("\n", "\n" + indent))
        while True:
            w = input(
                "\nChoose a scenario (1 - "
                + str(len(drafts))
                + "), or enter a message for the AI to regenerate scenarios: "
            )
            if w == "":
                continue
            if w.isdigit():
                try:
                    n = int(w)
                    chosen_scenario = drafts[n - 1]
                    break
                except:
                    continue
            else:
                # it was a string input
                hint = w
                break

    # at this point we have a chosen scenario
    print("You selected `" + chosen_scenario.name + "`. Fleshing out scenario...")
    return box.new(
        Scenario,
        "Create and flesh out an adventure scenario called '"
        + chosen_scenario.name
        + "', with the following initial description: \n"
        + chosen_scenario.description,
    )


def player_creation_dialog(scenario, party=True):
    box = ghostbox.from_generic(character_folder="player_creator", **default_options)
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
        print("Choose a character!\n")
        indent = "    "
        for i in range(len(pcs)):
            pc = pcs[i]
            print("(" + str(i + 1) + ") " + pc.show(indent=indent))

        while True:
            w = input(
                "\nChoose a player character (1 - "
                + str(len(pcs))
                + "), or enter a message for the AI to regenerate characters: "
            )
            if w == "":
                continue
            if w.isdigit():
                try:
                    n = int(w)
                    chosen_player = pcs[n - 1]
                    others = pcs[0 : (n - 1)] + pcs[(n - 1) :]
                    break
                except:
                    continue
            else:
                # it was a string input
                hint = w
                break

    # at this point we have a chosen player
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

    print("Choose a new special ability!")
    for i in range(len(new_abilities)):
        special = new_abilities[i]
        print(
            f"({i+1}) {special.name}. {special.description} ({special.fate_cost} fate)"
        )

    while True:
        try:
            choice = new_abilities[
                int(input(f"Choose (1 - {len(new_abilities)+1}): ")) - 1
            ]
            break
        except:
            continue

    print("You gain " + choice.name)
    game.player.special_abilities.append(choice)
    print("You can choose to drop one of your abilities.")
    for i in range(len(game.player.special_abilities)):
        special = game.player.special_abilities[i]
        print(
            f"({i+1}) {special.name}. {special.description} ({special.fate_cost} fate)"
        )

    while True:
        w = input(
            f"Choose (1 - {len(game.player.special_abilities)+1}), or hit enter to keep all and proceed: "
        )
        if w.strip() == "":
            return
        try:
            drop_i = int(w)
        except:
            continue

    print(f"You lose {game.player.special_abilities[drop_i]}.")
    del game.player.special_abilities[drop_i]


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
    args = p.parse_args()

    if args.scenario_file == "":
        scenario = scenario_creation_dialog(initial_prompt=args.scenario_prompt)
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
            print(f"Error: Couldn't load scenario file: {args.scenario_file}")
            return

    print(scenario.show())
    pc, others = player_creation_dialog(scenario, party=args.party)
    game = GameState(
        player=pc,
        party=others,
        adventure_scenario=scenario,
        fate=1,
        health=pc.max_health,
    )
    run(game)


def run(game):
    box = ghostbox.from_generic(character_folder="game_master", **default_options)
    narration = ""
    # this is the main loop
    while True:
        box.set_vars(
            {
                "scenario": game.adventure_scenario.show(),
                "party": "\n".join([npc.show() for npc in game.party]),
                "pc": game.player.show(),
                "fate": str(game.fate),
            }
        )
        situation = box.new(
            Situation,
            "Describe the current situation to the player, and give them some dramatic choices. Choices are always from the players perspective. Do not include the consequences in the choice text. Do not mention fate points. Try to include a mix of choices, and take the scenario, players, and history into account. Do not include special abilities in choices, the player will activate those seperately. Likewise, avoid mentioning the other party members in the choices. Remember that choices that are dangerous or involve the players motivation let them earn fate, so be sparing with those.",
        )
        print("\n" + situation.description + "\n")
        box.tts_say(situation.description, interrupt=False)

        print(situation.show_choices())
        n = len(situation.choices)
        while True:
            w = input(
                game.status()
                + "\nChoose (1 - "
                + str(n)
                + ") or use an ability (type name or initial letter). Typing `*` spends 3 fate to write your own choice.\n"
                + game.player.name
                + " > "
            )
            box.tts_stop()
            if w.strip() == "":
                continue

            if w.strip() == "advance" and game.fate >= game.advancement_fate_required():
                print("You have advanced your abilities!")
                advancement_dialog(game, box)
                print("Done with advancement. Let's return to the story.")
                continue

            if not (w.isdigit()) and not (w.strip() == "*"):
                # ability use
                special, msg = game.try_use_special_ability(w.strip())
                if special is None:
                    print(msg)
                    continue
                # fate was deducted and ability should be used
                print(msg)
                narration = box.new(
                    Consequences,
                    f"The player has used the following ability: {special.name}.\nPlease narrate the outcome of using this ability in this situation, or gently remind the player that this ability cannot be used, if it is not at all applicable to the current situation.",
                )
                break

            if w.strip() == "*":
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
            else:
                # ok, input was number, pick a dialog choice
                try:
                    choice = situation.choices[int(w) - 1]
                except:
                    continue

            fate_msg = game.gain_fate(choice.fate())
            print(fate_msg + "\n" if fate_msg else "" + "Please wait...")
            narration = box.new(
                Consequences,
                "The player has chosen the following: \n"
                + choice.show()
                + "\nPlease narrate the consequences of the players choice, which should drive the story forward and lead into a new dramatic situation. Keep it brief, focused, but evocative. Do not generate new choices as part of this response. Be sure to adhere to the scenario's style guide in your narration.",
            )
            break

        # we have narration/consequences of choice or ability use
        box.tts_say(narration.text)
        print(narration.text)
        msg, game_over = game.handle_consequences(narration)
        print(msg)
        if game_over:
            break

    # game over man
    goodbye = box.new(
        Message,
        "The game is over for the player. Reason: "
        + msg
        + "\nPlease write a suitable goodbye narration to send them off.",
    ).text
    print(goodbye)
    box.tts_say(goodbye, interrupt=False)
    input()


if __name__ == "__main__":
    main()
