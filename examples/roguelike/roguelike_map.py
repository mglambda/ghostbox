import random
from typing import Callable, Optional
from roguelike_types import *


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


def mapgen_generic(game: GameState, dungeon_level: int) -> GameState:
    # Constants for room generation
    MIN_ROOM_WIDTH = 3
    MAX_ROOM_WIDTH = 12
    MIN_ROOM_HEIGHT = 3
    MAX_ROOM_HEIGHT = 12
    MAX_ROOMS = 20
    GRID_WIDTH = 70
    GRID_HEIGHT = 40

    rooms = []
    upstairs_placed = False

    # Step 1: Check for a downstairs on the level above
    if dungeon_level > 0:
        for x in range(1, GRID_WIDTH - 1):
            for y in range(1, GRID_HEIGHT - 1):
                for entity in game.at(x, y, dungeon_level - 1):
                    if game.get(Downstairs, entity):  # Check for Downstairs component
                        # Place a room with an upstairs at the same coordinates
                        width = random.randint(MIN_ROOM_WIDTH, MAX_ROOM_WIDTH)
                        height = random.randint(MIN_ROOM_HEIGHT, MAX_ROOM_HEIGHT)
                        x = max(1, x - width // 2)
                        y = max(1, y - height // 2)
                        x = min(GRID_WIDTH - width - 1, x)
                        y = min(GRID_HEIGHT - height - 1, y)
                        new_room = (x, y, width, height)
                        make_room(game, x, y, dungeon_level, width, height)
                        make_upstairs(
                            game, x + width // 2, y + height // 2, dungeon_level
                        )
                        rooms.append(new_room)
                        upstairs_placed = True
                        break
                if upstairs_placed:
                    break
            if upstairs_placed:
                break

    # Step 2: Generate additional rooms regardless of whether an upstairs was placed
    for _ in range(MAX_ROOMS):
        width = random.randint(MIN_ROOM_WIDTH, MAX_ROOM_WIDTH)
        height = random.randint(MIN_ROOM_HEIGHT, MAX_ROOM_HEIGHT)
        x = random.randint(1, GRID_WIDTH - width - 1)
        y = random.randint(1, GRID_HEIGHT - height - 1)

        new_room = (x, y, width, height)
        # Check for intersections with existing rooms
        intersects = False
        for other_room in rooms:
            if (
                x < other_room[0] + other_room[2]
                and x + width > other_room[0]
                and y < other_room[1] + other_room[3]
                and y + height > other_room[1]
            ):
                intersects = True
                break
        if not intersects:
            make_room(game, x, y, dungeon_level, width, height)
            rooms.append(new_room)

    # Step 3: Connect rooms with corridors
    for i in range(1, len(rooms)):
        (x1, y1, w1, h1) = rooms[i - 1]
        (x2, y2, w2, h2) = rooms[i]
        # Randomly choose to start horizontal or vertical
        if random.choice([True, False]):
            # Horizontal then vertical
            for x in range(x1 + w1 // 2, x2 + w2 // 2):
                make_floor(
                    game, x, y1 + h1 // 2, dungeon_level, name="Corridor", display=","
                )
            for y in range(y1 + h1 // 2, y2 + h2 // 2):
                make_floor(
                    game, x2 + w2 // 2, y, dungeon_level, name="Corridor", display=","
                )
        else:
            # Vertical then horizontal
            for y in range(y1 + h1 // 2, y2 + h2 // 2):
                make_floor(
                    game, x1 + w1 // 2, y, dungeon_level, name="Corridor", display=","
                )
            for x in range(x1 + w1 // 2, x2 + w2 // 2):
                make_floor(
                    game, x, y2 + h2 // 2, dungeon_level, name="Corridor", display=","
                )

    # Step 4: Place a downstairs in a random room, but not the one with the upstairs
    if rooms:
        # Exclude the room with the upstairs
        available_rooms = [
            room
            for room in rooms
            if not (
                room[0] <= x <= room[0] + room[2] and room[1] <= y <= room[1] + room[3]
            )
        ]
        if available_rooms:
            downstairs_room = random.choice(available_rooms)
            (x, y, w, h) = downstairs_room
            make_downstairs(game, x + w // 2, y + h // 2, dungeon_level)

    return game
