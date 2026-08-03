from typing import  *
from pydantic import BaseModel, Field
import random

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


A = TypeVar("A")
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

def shorten_name(name: str) -> str:
    """Shortens a name in a sensible manner. Removes nicknames and lastnames."""
    ws = name.split(" ")
    return ws[0]
