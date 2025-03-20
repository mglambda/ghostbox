import random
from typing import *
from pydantic import BaseModel, Field
from roguelike_types import *
from roguelike_instructions import *
from roguelike_results import *
from abc import ABC, abstractmethod

# some simple helpers


def make_wall(
    game: GameState,
    x: int,
    y: int,
    dungeon_level: int,
    name: str = "Wall",
    display: str = "#",
    color: str = "grey",
) -> UID:
    wall_uid = game.new()
    game.enable(Name, wall_uid, Name(name=name))
    game.enable(Move, wall_uid, Move(x=x, y=y, dungeon_level=dungeon_level))
    game.enable(Display, wall_uid, Display(unicode_character=display, color=color))
    game.enable(Matter, wall_uid, Matter(material=Material.Stone))
    game.enable(Solid, wall_uid, Solid())
    return wall_uid


def make_floor(
    game: GameState,
    x: int,
    y: int,
    dungeon_level: int,
    name: str = "Floor",
    display: str = ".",
    color: str = "grey",
) -> UID:
    floor_uid = game.new()
    game.enable(MapTile, floor_uid, MapTile())
    game.enable(Name, floor_uid, Name(name=name))
    game.enable(Move, floor_uid, Move(x=x, y=y, dungeon_level=dungeon_level))
    game.enable(Display, floor_uid, Display(unicode_character=display, color=color))
    game.enable(Matter, floor_uid, Matter(material=Material.Stone))
    return floor_uid


def make_room(
    game: GameState,
    x_left: int,
    y_top: int,
    dungeon_level: int,
    width: int,
    height: int,
    floor_display: str = ".",
    wall_display: str = "#",
    floor_color: str = "grey",
    wall_color: str = "grey",
    floor_callback: Optional[Callable[[GameState, UID], None]] = None,
    wall_callback: Optional[Callable[[GameState, UID], None]] = None,
) -> None:
    for x in range(x_left, x_left + width):
        for y in range(y_top, y_top + height):
            if (
                x == x_left
                or x == x_left + width - 1
                or y == y_top
                or y == y_top + height - 1
            ):
                wall_uid = make_wall(
                    game, x, y, dungeon_level, display=wall_display, color=wall_color
                )
                if wall_callback:
                    wall_callback(game, wall_uid)

            floor_uid = make_floor(
                game, x, y, dungeon_level, display=floor_display, color=floor_color
            )
            if floor_callback:
                floor_callback(game, floor_uid)


def make_upstairs(
    game: GameState,
    x: int,
    y: int,
    dungeon_level: int,
    name: str = "Upstairs",
    display: str = "<",
    color: str = "white",
) -> UID:
    upstairs_uid = game.new()
    game.enable(Name, upstairs_uid, Name(name=name))
    game.enable(Move, upstairs_uid, Move(x=x, y=y, dungeon_level=dungeon_level))
    game.enable(Display, upstairs_uid, Display(unicode_character=display, color=color))
    game.enable(Matter, upstairs_uid, Matter(material=Material.Stone))
    game.enable(Upstairs, upstairs_uid, Upstairs())
    return upstairs_uid


def make_downstairs(
    game: GameState,
    x: int,
    y: int,
    dungeon_level: int,
    name: str = "Downstairs",
    display: str = ">",
    color: str = "white",
) -> UID:
    downstairs_uid = game.new()
    game.enable(Name, downstairs_uid, Name(name=name))
    game.enable(Move, downstairs_uid, Move(x=x, y=y, dungeon_level=dungeon_level))
    game.enable(
        Display, downstairs_uid, Display(unicode_character=display, color=color)
    )
    game.enable(Matter, downstairs_uid, Matter(material=Material.Stone))
    game.enable(Downstairs, downstairs_uid, Downstairs())
    return downstairs_uid


def make_door(
    game: GameState,
    x: int,
    y: int,
    dungeon_level: int,
    material: Material = Material.Wood,
    name: str = "Door",
    display: str = "+",
    closed: bool = True,
    locked: bool = False,
    **kwargs,
) -> UID:
    door_uid = game.new()
    game.enable(Name, door_uid, Name(name=name))
    game.enable(Move, door_uid, Move(x=x, y=y, dungeon_level=dungeon_level))
    game.enable(Display, door_uid, Display(unicode_character=display, color="brown"))
    game.enable(Matter, door_uid, Matter(material=material))
    if closed:
        game.enable(Solid, door_uid, Solid())

    # Define scripts for opening and closing the door
    open_script = Script(
        name="Open Door",
        instructions=[
            DisableSolid(entity=door_uid),
            EnableDoor(
                entity=door_uid,
                closed=False,
                locked=locked,
                on_open=None,
                on_close=None,
                on_break=None,
            ),
        ],
    )
    close_script = Script(
        name="Close Door",
        instructions=[
            EnableSolid(entity=door_uid),
            EnableDoor(
                entity=door_uid,
                closed=True,
                locked=locked,
                on_open=None,
                on_close=None,
                on_break=None,
            ),
        ],
    )

    game.enable(
        Interact,
        door_uid,
        Interact(
            options=[
                InteractOption(name="Open", script=open_script),
                InteractOption(name="Close", script=close_script),
            ]
        ),
    )

    game.enable(
        Door,
        door_uid,
        Door(
            closed=closed,
            locked=locked,
            on_open=open_script,
            on_close=close_script,
            on_break=None,
        ),
    )
    return door_uid


class MapTilePrefab(BaseModel, arbitrary_types_allowed=True):
    """Preliminary representation of a dungeon map tile, used for map generation. This can later be turned into an actual entity in a GameState."""

    # corresponds to the move component
    x: int
    y: int
    dungeon_level: int

    # these correspond to various components
    name: str
    description: Optional[str] = None
    material: Material
    display: str
    color: str
    solid: bool

    # this will be called on the created entity in fill(), allowing the enabling of additional components.
    entity_callback: Optional[Callable[[GameState, UID], None]] = None

    # these don't have any corresponding components
    # they are only used as helpers during map generation
    is_exit: bool = False
    is_in_room: bool = False
    is_in_corridor: bool = False
    
    
    def fill(self, game: GameState) -> None:
        """Inserts this tile into a given game state by creating appropriate entity components."""
        tile_uid = game.new()
        game.enable(Name, tile_uid, Name(name=self.name, description=self.description))
        game.enable(
            Move, tile_uid, Move(x=self.x, y=self.y, dungeon_level=self.dungeon_level)
        )
        game.enable(
            Display, tile_uid, Display(unicode_character=self.display, color=self.color)
        )
        game.enable(Matter, tile_uid, Matter(material=self.material))
        game.enable(MapTile, tile_uid, MapTile())
        if self.solid:
            game.enable(Solid, tile_uid, Solid())
        if self.entity_callback:
            self.entity_callback(game, tile_uid)


class MapPrefab(BaseModel):
    """Preliminary representation of a dungeon map."""

    # carries x, y, and z (dungeon_level) coordinates as keys
    data: Dict[Tuple[int, int, int], MapTilePrefab] = {}

    def preview(self) -> str:
        """Gives a string representation of the map as an ASCII image."""
        if not self.data:
            return ""

        # Determine the bounds of the map
        min_x = min(x for (x, y, z) in self.data.keys())
        max_x = max(x for (x, y, z) in self.data.keys())
        min_y = min(y for (x, y, z) in self.data.keys())
        max_y = max(y for (x, y, z) in self.data.keys())

        # Create a grid of characters
        grid = [
            [" " for _ in range(max_x - min_x + 1)] for _ in range(max_y - min_y + 1)
        ]

        # Fill the grid with the display characters from the map tiles
        for (x, y, z), tile in self.data.items():
            grid[y - min_y][x - min_x] = tile.display

        # Convert the grid to a string
        return "\n".join("".join(row) for row in grid)

    def fill_all_at(self, game: GameState, x: int, y: int, dungeon_level: int) -> None:
        """Fills the given GameState with the internal map of prefab tiles. This will change the given game state, essentially rendering tiles onto the map as entities at the given  coordinates.
        :param game: The GameState object to modify.
        :param x: The x-coordinate of the top left corner of the to be rendered map or feature.
        :param y: The y-coordinate of the top left corner of the to be rendered map or feature.
        :param dungeon_level: The z-coordinate or dungeon level of the to be rendered map or feature.
        """
        for (x1, y1, dungeon_level1), tile_prefab in self.data.items():
            # adjust tile offset
            offset_tile = MapTilePrefab(
                **(
                    tile_prefab.model_dump()
                    | {
                        "x": x + x1,
                        "y": y + y1,
                        "dungeon_level": dungeon_level + dungeon_level1,
                    }
                )
            )
            offset_tile.fill(game)


class MapGenerator(Protocol):
    """Generates areas or volumes of map tiles and dungeon features in a preliminary representation.
    Map generators form a monoid under composition (+), with the neutral element being an instance of the MapGeneratorIdentity. Composition isn't guaranteed to commute.
    """

    def generate(self) -> MapPrefab:
        """Generates the actual map or dungeon feature."""
        pass

    def compose(self, other: "MapGenerator") -> "MapGenerator":
        """Compose one map generator with another , yielding a third that will generate a map that is a combination of the maps that would be generated by the composing generators.
        Composition does not in general commute."""
        pass

    # so we can write gen_a + gen_c
    def __add__(self, other: "MapGenerator") -> "MapGenerator":
        return self.compose(other)


class MapGeneratorIdentity(MapGenerator):
    def generate(self) -> MapPrefab:
        return MapPrefab(data={})

    def compose(self, other: "MapGenerator") -> "MapGenerator":
        return other


class FunctionGenerator(MapGenerator):
    def __init__(
        self,
        generate: Callable[[], MapPrefab],
        compose: Callable[[MapGenerator, MapGenerator], MapGenerator],
    ):
        self._generate = generate
        self._compose = compose

    def generate(self) -> MapPrefab:
        return self.generate()

    def compose(self, other: MapGenerator) -> MapGenerator:
        return self._compose(self, other)


class HorizontalComposeGenerator(MapGenerator):

    def __init__(self, _generate: Optional[Callable[[], MapPrefab]] = None, **kwargs):
        self._kwargs = kwargs
        self._generate = _generate

    def generate(self) -> MapPrefab:
        if self._generate is not None:
            return self._generate()
        cls = type(self)
        return cls.generate(self)

    def compose(self, other: "MapGenerator") -> "MapGenerator":
        def combined_generate():
            map1 = self.generate()
            map2 = other.generate()
            max_x1 = max(x for (x, y, z) in map1.data.keys())
            for (x, y, z), tile in map2.data.items():
                map1.data[(x + max_x1 + 1, y, z)] = tile
            return map1

        cls = type(self)
        return cls(_generate=combined_generate, **self._kwargs)


class VerticalComposeGenerator(MapGenerator, ABC):
    def __init__(self, _generate: Optional[Callable[[], MapPrefab]] = None, **kwargs):
        self._kwargs = kwargs
        self._generate = _generate

    @abstractmethod
    def generate(self) -> MapPrefab:
        pass

    def compose(self, other: "MapGenerator") -> "MapGenerator":
        def combined_generate():
            map1 = self.generate()
            map2 = other.generate()
            max_y1 = max(y for (x, y, z) in map1.data.keys())
            for (x, y, z), tile in map2.data.items():
                map1.data[(x, y + max_y1 + 1, z)] = tile
            return map1

        cls = type(self)
        return cls(_generate=combined_generate, **self._kwargs)


class RowGenerator(VerticalComposeGenerator):
    def __init__(self, generators: List[MapGenerator]):
        self.generators = generators

    def generate(self) -> MapPrefab:
        return reduce(lambda a, b: a + b, self.generators).generate()


class FloorGenerator(HorizontalComposeGenerator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self) -> MapPrefab:
        return MapPrefab(
            data={
                (0, 0, 0): MapTilePrefab(
                    x=0,
                    y=0,
                    dungeon_level=0,
                    name="Floor",
                    material=Material.Stone,
                    display=".",
                    color="grey",
                    solid=False,
                    **self.kwargs,
                )
            }
        )


class WallGenerator(HorizontalComposeGenerator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self) -> MapPrefab:
        return MapPrefab(
            data={
                (0, 0, 0): MapTilePrefab(
                    x=0,
                    y=0,
                    dungeon_level=0,
                    name="Wall",
                    material=Material.Stone,
                    display="#",
                    color="grey",
                    solid=True,
                    **self.kwargs,
                )
            }
        )


class DoorGenerator(HorizontalComposeGenerator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self) -> MapPrefab:
        return MapPrefab(
            data={
                (0, 0, 0): MapTilePrefab(
                    x=0,
                    y=0,
                    dungeon_level=0,
                    name="Door",
                    material=Material.Wood,
                    display="+",
                    color="brown",
                    solid=True,
                    **self.kwargs,
                )
            }
        )


class UpstairsGenerator(HorizontalComposeGenerator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self) -> MapPrefab:
        return MapPrefab(
            data={
                (0, 0, 0): MapTilePrefab(
                    x=0,
                    y=0,
                    dungeon_level=0,
                    name="Upstairs",
                    material=Material.Stone,
                    display="<",
                    color="white",
                    solid=False,
                    **self.kwargs,
                )
            }
        )


class DownstairsGenerator(HorizontalComposeGenerator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self) -> MapPrefab:
        return MapPrefab(
            data={
                (0, 0, 0): MapTilePrefab(
                    x=0,
                    y=0,
                    dungeon_level=0,
                    name="Downstairs",
                    material=Material.Stone,
                    display=">",
                    color="white",
                    solid=False,
                    **self.kwargs,
                )
            }
        )


class SimpleRoomGenerator(HorizontalComposeGenerator):
    def __init__(
            self,
            width: int,
            height: int,
            exit_chance: float = 0.25,
            exit_is_door_chance: float = 0.33,
            ensure_exit: bool = True,
            wall_kwargs: Optional[Dict[str, Any]] = None,
            floor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.width = width
        self.height = height
        self.exit_chance = exit_chance
        self.exit_is_door_chance = exit_is_door_chance
        self.ensure_exit = ensure_exit
        self._has_exit = False        
        self.wall_kwargs = wall_kwargs or {}
        self.ensure_exit = ensure_exit
        self.floor_kwargs = floor_kwargs or {}
        self.default_floor_kwargs = {
            k: v
            for k, v in FloorGenerator().generate().data[(0, 0, 0)].model_dump().items()
            if k not in ["x", "y", "dungeon_level"]
        }
        self.default_wall_kwargs = {
            k: v
            for k, v in WallGenerator().generate().data[(0, 0, 0)].model_dump().items()
            if k not in ["x", "y", "dungeon_level"]
        }

    def generate(self) -> MapPrefab:
        # Create the room with walls and floors
        room_map = MapPrefab(data={})

        # Create the floor tiles
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                room_map.data[(x, y, 0)] = MapTilePrefab(
                    x=x,
                    y=y,
                    dungeon_level=0,
                    **(self.default_floor_kwargs | self.floor_kwargs),
                )
                room_map.data[(x, y, 0)].is_in_room = True

        # Create the walls
        for x in range(self.width):
            room_map.data[(x, 0, 0)] = MapTilePrefab(
                x=x,
                y=0,
                dungeon_level=0,
                **(self.default_wall_kwargs | self.wall_kwargs),
            )
            room_map.data[(x, self.height - 1, 0)] = MapTilePrefab(
                x=x,
                y=self.height - 1,
                dungeon_level=0,
                **(self.default_wall_kwargs | self.wall_kwargs),
            )
        for y in range(1, self.height - 1):
            room_map.data[(0, y, 0)] = MapTilePrefab(
                x=0,
                y=y,
                dungeon_level=0,
                **(self.default_wall_kwargs | self.wall_kwargs),
            )
            room_map.data[(self.width - 1, y, 0)] = MapTilePrefab(
                x=self.width - 1,
                y=y,
                dungeon_level=0,
                **(self.default_wall_kwargs | self.wall_kwargs),
            )

        # Randomly place exits in the walls

        placers = [
        # top
            lambda: self._place_exits(room_map, [(x, 0) for x in range(1, self.width-1)]),
        # bottom
            lambda: self._place_exits(room_map, [(x, self.height-1) for x in range(1, self.width-1)]),
        # left
            lambda: self._place_exits(room_map, [(0, y) for y in range(1, self.height-1)]),
        # right
        lambda: self._place_exits(room_map, [(self.width-1, y) for y in range(1, self.height-1)])
            ]

        # we will try to generate multiple times
        # but i don't want to deal with insane users that make exit_chance = 0.0
        # so we have a failsafe
        max_trips = 5
        trip = 0
        while not(self._has_exit):
            trip += 1
            random.shuffle(placers)
            for i in range(len(placers)):
                placers[i]()

            if not(self.ensure_exit):
                break

            if trip >= max_trips:
                break
            
        return room_map

    def _place_exits(
        self, room_map: MapPrefab, positions: List[Tuple[int, int]]
    ) -> None:
        while positions:
            if random.random() > self.exit_chance:
                # no exit
                break

            # ok we add an exit
            random.shuffle(positions)
            x, y = positions.pop()
            print(f"{x},{y}")
            if random.random() < self.exit_is_door_chance:
                # Place a door
                room_map.data[(x, y, 0)] = MapTilePrefab(
                    x=x,
                    y=y,
                    dungeon_level=0,
                    **(
                        self.default_wall_kwargs
                        | self.wall_kwargs
                        | {
                            "display": "+",
                            "name": "Door",
                            "material": Material.Wood,
                            "solid": True,
                        }
                    ),
                )
            else:
                # Place a floor tile
                room_map.data[(x, y, 0)] = MapTilePrefab(
                    x=x,
                    y=y,
                    dungeon_level=0,
                    **(self.default_floor_kwargs | self.floor_kwargs),
                )
            room_map.data[(x, y, 0)].is_exit = True 
                
            # ok we added an exit
            self._has_exit
            # we go back through while loop, but with some positions removed
            # because we don't want openings right next to each other
            forbidden = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            positions = [p for p in positions if p not in forbidden]
