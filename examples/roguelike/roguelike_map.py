import random, re, os
from typing import *
from pydantic import BaseModel, Field
from roguelike_types import *
from roguelike_instructions import *
from roguelike_results import *
from abc import ABC, abstractmethod
from enum import StrEnum

def find_matching_files(directory, regex_pattern):
    """
    Find all files in the given directory and its subdirectories that match the regex pattern.

    :param directory: Path to the directory to search in.
    :param regex_pattern: Regular expression pattern to match filenames.
    :return: List of filenames that match the pattern.
    """
    # Compile the regex pattern for better performance
    pattern = re.compile(regex_pattern)
    
    # List to hold matching filenames
    matching_files = []
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the filename matches the regex pattern
            if pattern.match(filename):
                # Append the full path of the file to the list
                matching_files.append(os.path.join(root, filename))
    
    return matching_files

def find_dungeon_images(pattern, subdir="") -> List[str]:
    dir = "img/dngn/"
    return [w.replace("img/","") for w in find_matching_files(dir + subdir, pattern)]

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
    images: List[str] = []
    color: str
    solid: bool

    # this will be called on the created entity in fill(), allowing the enabling of additional components.
    entity_callback: Optional[Callable[[GameState, UID], None]] = None

    # these don't have any corresponding components
    # they are only used as helpers during map generation
    is_entry: bool = False
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
            Display, tile_uid, Display(unicode_character=self.display, color=self.color, image= random.choice(self.images) if self.images else None)
        )
        game.enable(Matter, tile_uid, Matter(material=self.material))
        game.enable(MapTile, tile_uid, MapTile())
        if self.solid:
            game.enable(Solid, tile_uid, Solid())
        if self.entity_callback:
            self.entity_callback(game, tile_uid)

    def apply_some_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Applies a dictionary of keyword arguments to update some aspects of the tile.
        In particular, this will flat out ignore all x, y, and dungeon level parameters."""
        forbidden = "x y dungeon_level".split(" ")
        for k, v in kwargs.items():
            if k in forbidden:
                continue
            self.__dict__[k] = v

# so if a room has a door on the right wall at position x,y, it would have a connector tile at position x+1,y with type right to left, since you come from the right and go left to enter the room
ConnectorType = StrEnum("ConnectorType", "RightToLeft LeftToRight TopToBottom BottomToTop")

def connectors_match(a: ConnectorType, b: ConnectorType) -> bool:
    """Returns true if connectors a and b match.
    Matching connectors are e.g. LeftToRight and RightToLeft."""
    matches = [(ConnectorType.RightToLeft, ConnectorType.LeftToRight), (ConnectorType.TopToBottom, ConnectorType.BottomToTop)]
    more_matches = matches + [(b, a) for a,b in matches]
    return (a,b) in more_matches

class MapPrefab(BaseModel):
    """Preliminary representation of a dungeon map."""

    # carries x, y, and z (dungeon_level) coordinates as keys
    data: Dict[Tuple[int, int, int], MapTilePrefab] = {}

    # carries information about tiles that aren't technically part of the map, but are locations where other maps may connect
    connectors: Dict[Tuple[int, int, int], ConnectorType] = {}    

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

    def max_bounds(self) -> Tuple[int,int,int]:
        return self._bounds(True)

    def min_bounds(self) -> Tuple[int,int,int]:
        return self._bounds(False)
    

    def _bounds(self, reverse: bool):
        keys = self.data.keys()
        x_bound = sorted([x for x,y,z in keys], reverse=reverse)[0]
        y_bound = sorted([y for x,y,z in keys], reverse=reverse)[0]
        z_bound = sorted([z for x,y,z in keys], reverse=reverse)[0]
        return x_bound, y_bound, z_bound

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

class ConnectorComposeGenerator(MapGenerator, ABC):
    """Composes with other maps by attaching to their connectors according to the type of its own and the other maps connector."""
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
            failure = True
            for coords1, c1 in map1.connectors.items():
                for coords2, c2 in map2.connectors.items():
                    if connectors_match(c1, c2):
                        # Calculate the offset needed to connect map2 to map1
                        if c1 == ConnectorType.RightToLeft:
                            # map2 shifts to right
                            offset_x = coords1[0] - coords2[0] - 1
                            offset_y = coords1[1] - coords2[1]
                        elif c1 == ConnectorType.LeftToRight:
                            offset_x = coords1[0] - coords2[0] + 1
                            offset_y = coords1[1] - coords2[1]
                        elif c1 == ConnectorType.TopToBottom:
                            offset_x = coords1[0] - coords2[0]
                            offset_y = coords1[1] - coords2[1] + 1
                        elif c1 == ConnectorType.BottomToTop:
                            offset_x = coords1[0] - coords2[0]
                            offset_y = coords1[1] - coords2[1] - 1

                        # Apply the offset to all tiles in map2
                        for (x, y, z), tile in map2.data.items():
                            map1.data[(x + offset_x, y + offset_y, z)] = tile

                        # Remove the used connectors
                        del map1.connectors[coords1]
                        del map2.connectors[coords2]

                        # offset the other connectors and add them to us
                        for (x, y, z), con in map2.connectors.items():
                            map1.connectors[(x+offset_x,y+offset_y,z)] = con
                        

                        failure = False
                        break
                if not failure:
                    break

            if failure:
                raise ValueError("No matching connectors found between the two maps.")

            return map1

        #cls = type(self)
        #return cls(**({k:v for k,v in self.__dict__.items() if not(k.startswith("__"))} | self._kwargs | {"_generate":combined_generate}))
        class AnonymousGenerator(ConnectorComposeGenerator):
            def __init__(self, **kwargs):
                self._kwargs = kwargs

            def generate(self):
                return combined_generate()
            
        return AnonymousGenerator()
    

class RowGenerator(VerticalComposeGenerator):
    def __init__(self, generators: List[MapGenerator]):
        self.generators = generators

    def generate(self) -> MapPrefab:
        return reduce(lambda a, b: a + b, self.generators).generate()


class FloorGenerator(HorizontalComposeGenerator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self) -> MapPrefab:
        map = MapPrefab(
            data={
                (0, 0, 0): MapTilePrefab(
                    x=0,
                    y=0,
                    dungeon_level=0,
                    name="Floor",
                    material=Material.Stone,
                    display=".",
                    color="grey",
                    images=[f"dngn/floor/limestone{n}.png" for n in range(10)],
                    solid=False,
                )
            }
        )
        map.data[(0,0,0)].apply_some_kwargs(self.kwargs)
        return map
    


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
                    images=find_dungeon_images(r"brick.*gray"),
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
                    images=find_dungeon_images(r"closed_door"),
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
                    images=find_dungeon_images(r".*return_hell.*"),
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
                    images=find_dungeon_images(r".*enter_hell.*"),
                    solid=False,
                    **self.kwargs,
                )
            }
        )


class SimpleRoomGenerator(ConnectorComposeGenerator):
    def __init__(
            self,
            width: int,
            height: int,
            exit_chance: float = 0.15,
            exit_is_door_chance: float = 0.33,
            ensure_exit: bool = True,
            wall_kwargs: Optional[Dict[str, Any]] = None,
            floor_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
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
        forbidden = []
        decay = 0.00
        while positions:
            if random.random() > self.exit_chance + decay:
                # no exit
                break

            # ok we add an exit
            random.shuffle(positions)
            x, y = positions.pop()
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
                
            room_map.data[(x, y, 0)].is_entry = True
            x_con, y_con, con_type = self._determine_connector(room_map, x, y, 0)
            if con_type and x_con and y_con:
                room_map.connectors[(x_con, y_con, 0)] = con_type
            
                
            # ok we added an exit
            self._has_exit
            # we go back through while loop, but with some positions removed
            # because we don't want openings right next to each other
            forbidden.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
            positions = [p for p in positions if p not in forbidden]
            # exits get less likely the more we place
            decay += 0.05
            

    def _determine_connector(self, room_map: MapPrefab, x, y, dungeon_level) -> Tuple[Optional[int], Optional[int], Optional[ConnectorType]]:
        # FIXME: I realize this method could have been written more simply (with width, height)
        # but this is a candidate to be generalized and put into MapPrefab later
        tile = room_map.data[(x,y,dungeon_level)]
        failure = (None, None, None)
        if not(tile.is_entry):
            return failure

        # find the bounds
        xs = sorted([x for x,y,z in room_map.data.keys()])
        ys = sorted([y for x, y, z in room_map.data.keys()])
        if xs == [] or ys == []:
            return failure
        
        x_max, x_min = xs[-1], xs[0]
        y_max, y_min = ys[-1], ys[0]

        # this should work since the rooms are guaranteed to be rectangular
        if x == x_max:
            # it's on the right wall
            # no need to consider y since corners are excluded anyway
            return x+1, y, ConnectorType.RightToLeft
        elif x == x_min:
            # left
            return x-1, y, ConnectorType.LeftToRight
        elif y == y_max:
            # bottom
            return x, y+1, ConnectorType.BottomToTop
        elif y == y_min:
            return x, y-1, ConnectorType.TopToBottom

        # not sure what happened here
        return failure



        
            
            
class CorridorGenerator(ConnectorComposeGenerator):
    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        dungeon_level: int = 0,
        has_walls: bool = False,
        diagonal_moves: bool = True,
        wall_kwargs: Dict[str, Any] = {},
        floor_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.dungeon_level = dungeon_level
        self.has_walls = has_walls
        self.diagonal_moves = diagonal_moves
        self.wall_kwargs = wall_kwargs
        self.floor_kwargs = floor_kwargs

        # we can already tell the connectors
        #let's consider the far end of the corridor (x2,y2)
        if x1 <= x2:
            if y1 <= y2:
                # far end is bottom right
                self.connectors = [(x1-1,y1,ConnectorType.LeftToRight),(x1,y1-1,ConnectorType.TopToBottom),(x2+1,y2,ConnectorType.RightToLeft),(x2,y2+1,ConnectorType.BottomToTop)]                
            else:
                # far end is top right
                self.connectors = [(x1-1,y1, ConnectorType.LeftToRight),(x1,y1+1, ConnectorType.BottomToTop),(x2+1,y2, ConnectorType.RightToLeft),(x2,y2-1, ConnectorType.TopToBottom)]                

        else:
            # x1 > x2
            if y1 < y2:
                # far corner is top left
                self.connectors = [(x1+1,y1,ConnectorType.RightToLeft),(x1,y1+1,ConnectorType.BottomToTop),(x2-1,y2,ConnectorType.LeftToRight),(x2,y2-1,ConnectorType.TopToBottom)]
            else:
                # far corner is bottom left
                self.connectors = [(x1+1,y1,ConnectorType.RightToLeft),(x1,y1-1,ConnectorType.TopToBottom),(x2-1,y2,ConnectorType.LeftToRight),(x2,y2+1,ConnectorType.BottomToTop)]

    def generate(self) -> MapPrefab:
        map_prefab = MapPrefab(data={})

        # Generate the floor tiles
        if self.diagonal_moves:
            # Use Bresenham's line algorithm for diagonal corridors
            for x, y in self.bresenham_line(self.x1, self.y1, self.x2, self.y2):
                map_prefab.data[(x, y, self.dungeon_level)] = MapTilePrefab(
                    x=x,
                    y=y,
                    dungeon_level=self.dungeon_level,
                    name="Corridor",
                    material=Material.Stone,
                    display=".",
                    color="grey",
                    images=find_dungeon_images(".*grey_dirt.*", subdir="floor/"),
                    solid=False,
                    is_entry=True if (x == self.x1 and y == self.y1) or (x == self.x2 and y == self.y2) else False,
                )
                map_prefab.data[(x, y, self.dungeon_level)].apply_some_kwargs(self.floor_kwargs)
        else:
            # Generate a corridor with only horizontal and vertical segments
            # First, move horizontally from (x1, y1) to (x2, y1)
            for x in range(min(self.x1, self.x2), max(self.x1, self.x2) + 1):
                map_prefab.data[(x, self.y1, self.dungeon_level)] = MapTilePrefab(
                    x=x,
                    y=self.y1,
                    dungeon_level=self.dungeon_level,
                    name="Floor",
                    material=Material.Stone,
                    display=".",
                    color="grey",
                    images=find_dungeon_images(".*grey_dirt.*", subdir="floor/"),                    
                    solid=False,
                    is_entry=True if (x == self.x1 and self.y1 == self.y1) or (x == self.x2 and self.y1 == self.y1) else False,
                )
                map_prefab.data[(x, self.y1, self.dungeon_level)].apply_some_kwargs(self.floor_kwargs)
            # Then, move vertically from (x2, y1) to (x2, y2)
            for y in range(min(self.y1, self.y2), max(self.y1, self.y2) + 1):
                map_prefab.data[(self.x2, y, self.dungeon_level)] = MapTilePrefab(
                    x=self.x2,
                    y=y,
                    dungeon_level=self.dungeon_level,
                    name="Floor",
                    material=Material.Stone,
                    display=".",
                    color="grey",
                    images=find_dungeon_images(".*grey_dirt.*", subdir="floor/"),                    
                    solid=False,
                    is_entry=True if (self.x2 == self.x2 and y == self.y1) or (self.x2 == self.x2 and y == self.y2) else False,
                )
                map_prefab.data[(self.x2, y, self.dungeon_level)].apply_some_kwargs(self.floor_kwargs)                

        # Generate the walls if needed
        if self.has_walls:
            # Determine the bounding box for the walls
            wall_x1 = min(self.x1, self.x2) - 1
            wall_x2 = max(self.x1, self.x2) + 1
            wall_y1 = min(self.y1, self.y2) - 1
            wall_y2 = max(self.y1, self.y2) + 1

            for x in range(wall_x1, wall_x2 + 1):
                for y in range(wall_y1, wall_y2 + 1):
                    if (x, y, self.dungeon_level) not in map_prefab.data:
                        map_prefab.data[(x, y, self.dungeon_level)] = MapTilePrefab(
                            x=x,
                            y=y,
                            dungeon_level=self.dungeon_level,
                            name="Wall",
                            material=Material.Stone,
                            display="#",
                            color="grey",
                            images=find_dungeon_images(".*cobalt_rock.*", subdir="wall/"),
                            solid=True,
                        )
                        map_prefab.data[(x, y, self.dungeon_level)].apply_some_kwargs(self.wall_kwargs)

        # connectors were calculated in __init__
        for (x, y, con) in self.connectors:
            map_prefab.connectors[(x, y, self.dungeon_level)] = con
        return map_prefab

    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Generate a list of (x, y) tuples for a line from (x0, y0) to (x1, y1) using Bresenham's line algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

class TestGenerator(ConnectorComposeGenerator):
    def __init__(self, con: ConnectorType, **kwargs):
        self.con = con

    def generate(self):
        f = FloorGenerator(display=self.con.name[0])
        x = f.generate()
        if self.con == ConnectorType.LeftToRight:
            x.connectors[(-1,0, 0)] = self.con
        elif self.con == ConnectorType.RightToLeft:
            x.connectors[(1,0, 0)] = self.con
        elif self.con == ConnectorType.TopToBottom:
            x.connectors[(0,-1, 0)] = self.con
        elif self.con == ConnectorType.BottomToTop:
            x.connectors[(0,1, 0)] = self.con            
        return x
    
def test():
    r1 = SimpleRoomGenerator(5, 10)
    r2 = SimpleRoomGenerator(12, 4)
    r = r1 + r2
    c = CorridorGenerator(0,0,5,6, diagonal_moves=False, has_walls=False)
    right = TestGenerator(ConnectorType.RightToLeft)
    left = TestGenerator(ConnectorType.LeftToRight)
    bottom = TestGenerator(ConnectorType.BottomToTop)
    top = TestGenerator(ConnectorType.TopToBottom)
    x = c + left + right + top + bottom

def room_range(min_v, max_v):
    return SimpleRoomGenerator(width=random.randint(max(2,min_v), max_v),
                               height=random.randint(max(2,min_v),max_v))


def small_room():
    return room_range(2, 5)


def medium_room():
    return room_range(5, 12)

def big_room():
    return room_range(10, 15)

def corridor_range(min_v, max_v, diagonal_moves=False):
    sign_x = random.choice([1,-1])
    sign_y = random.choice([1,-1])
    x = random.randint(max(3,min_v), max_v)
    y = random.randint(max(3, min_v), max_v)
    
    return CorridorGenerator(0,0, sign_x * x, sign_y * y, diagonal_moves=diagonal_moves)

def small_corridor():
    return corridor_range(5, 10)

def long_corridor():
    return corridor_range(10, 20)

def mapgen_small(game: GameState, dungeon_level=0) -> None:
    """Fills a given dungeon level with a small map, in a given game state."""
    while True:
        try:
            # this can fail due to connector mismatch
            # we just retry in that case
            a = small_room() + small_corridor() + small_room()
            b = medium_room() + small_corridor() + small_room()
            c = big_room()
            l1, l2 = long_corridor(), long_corridor()
            map = ((c + l1 + a) + l2 + b).generate()
            map.fill_all_at(game, 0, 0, dungeon_level)
        except ValueError as e:
            continue
        return



