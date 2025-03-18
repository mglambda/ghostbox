from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import *
from functools import reduce
from enum import StrEnum

UID = NewType("UID", int)


class GameResult(ABC, BaseModel, arbitrary_types_allowed=True):
    """Result type for the GameState delta function, representing what happens after one step in the game's logic.
    A GameResult has a handle method, which expects some kind of controller object.
    GameResults may carry arbitrary data, and their handle function may cause side effects, like playing sounds, or altering the graphical interface.
    """

    @abstractmethod
    def handle(self, ctl: 'Controller') -> None:
        pass
    
class NothingHappened(GameResult):
    """Game Result representing a noop."""

    def handle(self, ctl: 'Controller') -> None:
        return
    

R = TypeVar("R", bound=GameResult)
I = TypeVar("I", bound='GameInstruction')
DeltaResultType = Tuple[R, List["I"]]


class GameInstruction(ABC, BaseModel, arbitrary_types_allowed=True):
    """Interface for GameInstruction types. You can consider these entries in the delta function fo the game model's state machine."""

    _types: ClassVar[Dict[str, type]] = {}

    # used to register automatically all the submodels in `_types`.
    def __init_subclass__(cls, type: Optional[str] = None):
        if type is not None:
            cls._types[type] = cls
        else:
            cls._types[cls.__name__] = cls
        # cls._types[type or cls.__name__] = cls

    @abstractmethod
    def delta(self, game: 'GameState') -> DeltaResultType:
        """Execute one step in the game's logic, returning a new GameInstructions along witha GameResult object.
        This function will alter the GameState as a side effect, breaking with a pure finite state automaton formalism for convenience.
        """
        pass

class DoNothing(GameInstruction):
    """Waits for one game tick."""

    def delta(self, game: 'GameState') -> DeltaResultType:
        return NothingHappened(), []
    
class ComponentStore[A](BaseModel):
    name: str
    data: Dict[UID, A] = {}


class GameState(BaseModel):
    next_entity_id: UID = UID(0)
    stores: Dict[type, ComponentStore] = {}

    def enable(self, component_name: type, entity: UID, component: Any) -> UID:
        if component_name not in self.stores:
            self.stores[component_name] = ComponentStore(name=component_name.__name__)

            # so what we want to express here (at compile time) is

        self.stores[component_name].data[entity] = component
        return entity

    def disable(self, component_name: type, entity: UID) -> UID:
        if (store := self.stores.get(component_name, None)) is None:
            return entity

        if entity in store.data:
            del store.data[entity]
        return entity

    def get(self, component_name: type, entity: UID) -> Optional[Any]:
        if (store := self.stores.get(component_name, None)) is None:
            return None

        return store.data.get(entity, None)

    def new(self) -> UID:
        id = int(self.next_entity_id)
        self.next_entity_id = UID(id+1)
        return UID(id)

    def entities(self) -> Set[UID]:
        return set(
            reduce(
                lambda a, b: a + b, #type: ignore
                [store.data.keys() for store in self.stores.values()],
                [],
            )
        )


class Controller(BaseModel):
    """The controller has the overview over all game resources, including the model  (GameState) and the view (i.e. some graphical interface. It also handles user input."""

    game: GameState
    # view is not defined yet
    view: Any = None

    # the entity that the player controls
    # the controller needs to know this, because obviously this is the entity we will display information about
    player: UID

    def run_terminal(self, instructions: List[GameInstruction]) -> None:
        """Runs the game in text only mode.
        We basically eshew a dedicated view class here, simply using input/print.
        Mostly used for debugging.
        This function will use some initial instructions to generate GameResults with the internal GameState, potentially changing it and generating more instructions. This process will potentially never halt.
        :param instructions: A list of instructions to start the game with, or continue a paused one.
        :return: Nothing. The function runs until it is interrupted or until all instructions have been processed and no new ones are generated. Implicitly, the GameState is returned as a property of the controller.
        """
        self.print(f"Starting game with player uid {self.player}")
        while True:
            try:
                while instruction := instructions.pop(0):
                    # execute the instruction with the current gamestate
                    # the GameState will be changed, and we potentially get new instructions
                    result, new_instructions = instruction.delta(self.game)
                    # the result's side effects procure, with access to the controller
                    # sound, graphics, in our case printing
                    result.handle(self)

                    # add the new instructions to the beginning
                    # (appending them would completely change the game)
                    instructions = new_instructions + instructions
            except IndexError as e:
                # the instruction queue is empty
                w = input("Next move: ")
                if w == "quit":
                    break
                else:
                    # FIXME: need to implement more instructions
                    instructions.append(DoNothing())
    def print(self, text: str) -> None:
        """Simple wrapper for print, for the terminal only version of controller."""
        print(text)
        #

class Script(BaseModel):
    """Helper object to group sets of GameInstructions together and document them."""
    name: str
    documentation: Optional[str] = None
    instructions: List[GameInstruction]
    
# Components follow


class Name(BaseModel):
    """Component for entities that can be named."""

    name: str
    description: Optional[str] = None

Material = StrEnum("Material", "Stone Earth Flesh Glass Iron Steel Bronze Obsidian Diamond Marble Wood Cloth Leather Hair Gemstone Paper Copper Porcelain Ice Plutonium")
Group = StrEnum(
    "Group",
    "Human Dwarf Elf Halfling Orc Gnome Plant Beast Giant  Lycanthrope Undead Demon Devil Abomination Angel Outsider Construct Fey Dragon Kobold Gnoll Monstrosity Goblin",
)


class GroupMember(BaseModel):
    """Component for entities that can be grouped, like 'Furniture', or 'Liches'."""

    group_name: Group
    # higher is better
    rank: int = 0


class Move(BaseModel):
    """Component for entities that can be moved in the game world.
    The necessary condition for an entity to be moved is having an x and y coordinate, though more coordinates may be present.
    """

    x: int
    y: int
    dungeon_level: int


class Damage(BaseModel):
    """Component for entities that can be damaged and destroyed or killed."""

    health: int
    leaves_corpse: bool = False

    # uid1 = subject so whoever is dieing
    on_death: Optional[Script] = None


class MeleeWeapon(BaseModel):
    """Component for entities that can be used as melee weapons."""

    damage_min: int
    damage_max: int
    critical_chance: float
    critical_multiplier: float = 2.0

    on_hit: Optional[Script] = None
    on_miss: Optional[Script] = None


class RangedWeapon(BaseModel):
    """Component for entities that can beused as ranged weapons, either thrown or shot."""

    damage_min: int
    damage_max: int
    critical_chance: float
    critical_multiplier: float = 2.0
    verb: Literal["throw", "shoot"]

    on_hit: Optional[Script] = None
    on_miss: Optional[Script] = None


class Inventory(BaseModel):
    """Component for entities that have an inventory."""

    items: List[UID]
    capacity: int

    on_drop_anything: Optional[Script] = None
    on_pickup_anything: Optional[Script] = None


class Consumable(BaseModel):
    """Component for entities that can be eaten or drunk."""

    nutrition: int

    on_eat: Optional[Script] = None
    on_drink: Optional[Script] = None


class Biological(BaseModel):
    """Component for biological entities."""

    satiety: Annotated[int, Field(ge=0, le=1000)] = 800

    # happens at satiety < 200
    on_starve: Optional[Script] = None


class Use(BaseModel):
    """Component for entities that can be used or applied."""

    uses_left: Optional[int] = None

    # an entity can be used without a target
    on_use: Optional[Script] = None

    # apply is transitive, and requires another entity to be applied to
    on_apply: Optional[Script] = None

    # gets called when uses_left hits 0
    on_depleted: Optional[Script] = None


class Carry(BaseModel):
    """Component for entities that can be carried in an inventory."""

    # weight will be compared to Inventory.capacity
    weight: int

    on_drop: Optional[Script] = None
    on_pickup: Optional[Script] = None


class Attributes(BaseModel):
    """Component for entities that have attributes."""

    # 10 is average for humans. 18 is hero. 20+ is usually reserved for monsters or magical beings.
    strength: int = Field(ge=1, le=30)
    dexterity: int = Field(ge=1, le=30)
    constitution: int = Field(ge=1, le=30)
    intelligence: int = Field(ge=1, le=30)
    wisdom: int = Field(ge=1, le=30)
    charisma: int = Field(ge=1, le=30)

    on_str_save: Optional[Script] = None
    on_dex_save: Optional[Script] = None
    on_con_save: Optional[Script] = None
    on_int_save: Optional[Script] = None
    on_wis_save: Optional[Script] = None
    on_cha_save: Optional[Script] = None

    @staticmethod
    def _modifier(n: int) -> int:
        return (n - 10) // 2

    def str(self) -> int:
        return self._modifier(self.strength)

    def dex(self) -> int:
        return self._modifier(self.dexterity)

    def con(self) -> int:
        return self._modifier(self.constitution)

    def wis(self) -> int:
        return self._modifier(self.wisdom)

    def cha(self) -> int:
        return self._modifier(self.charisma)


EquipmentSlot = StrEnum(
    "EquipmentSlot", "helmet torso gauntlets legs boots amulet left_ring right_ring"
)
WieldSlot = StrEnum("WieldSlot", "right_hand left_hand two_handed")


class Wear(BaseModel):
    """Component for entities that can be worn."""

    # armor class. The higher the better.
    ac: int
    slot: EquipmentSlot


    on_wear: Optional[Script] = None
    on_remove: Optional[Script] = None


class Wield(BaseModel):
    """Component for entities that can be wielded."""

    slot: WieldSlot
    
    on_equip: Optional[Script] = None
    on_unwield: Optional[Script] = None


class Visible(BaseModel):
    """Components for entities that can be seen visually by other entities, i.e. that reflect light.
    Almost all entities should have this component enabled by default.
    Disabling this component for an entity turns it invisible."""

    on_spotted: Optional[Script] = None


class Burn(BaseModel):
    """Component for entities that are currently burning."""

    # e.g. for limited fuel situations
    turns_left_burning: Optional[int] = None

    on_extinguish: Optional[Script] = None


class Wet(BaseModel):
    """Component for entities that are wet."""

    # most things dry after a while
    turns_left_wet: int = 3

    on_dry: Optional[Script] = None


class Solid(BaseModel):
    """Component for entities that block movement of other entities.
    Entities that lack this component can be moved across, like items on the ground etc.
    Entities with a solid component can not have a move compeonent with the same coordinates as another solid entity.
    Items that can be piled onto a square, for example, lack a solid component.
    """
    
    on_collide: Optional[Script] = None


class Liquid(BaseModel):
    """Component for entities that are liquid.
    Liquid components can not have a move component with the same coordinates as another liquid entity.
    """

    makes_wet: bool

    on_enter: Optional[Script] = None


class Levitate(BaseModel):
    """Components for entities that hover off the ground."""

    height: int


class SingleTargetEffect(BaseModel):
    """Component for entities that affect a single other entity with something."""

    target: UID
    turns_left: Optional[int] = None

    on_gain_effect: Optional[Script] = None
    on_each_turn: Optional[Script] = None
    on_lose_effect: Optional[Script] = None


class AreaEffect(BaseModel):
    """Component for an entity that affects an area, like a gas cloud or a magic aura.
    This component needs to be combined with the move component to have an origin x,y coordinate and dungeon level.
    """

    radius: int
    turns_left: Optional[int] = None

    on_entity_enter_area: Optional[Script] = None
    on_entity_leave_area: Optional[Script] = None
    on_gain_effect: Optional[Script] = None
    on_each_turn: Optional[Script] = None
    on_lose_effect: Optional[Script] = None

class Matter(BaseModel):
    """Component for entities that are made of something.
    This is e.g. 'Stone' for floors, 'Flesh' for biologicals, 'Steel' for swords etc.
    Being made of matter is still different from being solid. See the Solid component for more.
    Losing this component does not imply death. It is how ghosts are made."""

    material: Material

    # this is a special event for when entities get disintegrated but may remain as disembodied spirits
    on_disintegration: Optional[Script] = None
    
    
class MapTile(BaseModel):
    """Component for entities that are map tiles.
    This is usually combined with the Move component, since map tiles need an x,y and dungeon_level coordinate to make sense.
    This component is somewhat the opposite of the Solid component, as map tiles can share coordinates with other entities."""

    diggable: bool = True

    # when an entity steps onto the tile
    on_enter: Optional[Script] = None
    # when an entity leaves the tile
    on_leave: Optional[Script] = None

    
class Display(BaseModel):
    """Components for entities that can be shown by the graphics engine.
    This is technically not part of the game model, we just keep it here for convenience.
    Since this is an ascii (or rather, unicode) roguelike, all display information is text based.
    """

    unicode_character: str
    color: str
