from roguelike_types import *

def find_floor_tile(game: GameState, x: int, y: int, dungeon_level: int) -> Optional[Tuple[UID, MapTile]]:
    if (entities := game.at(x, y, dungeon_level)) == set():
        return None

    
    for e in entities:
        if (tile_component := game.get(MapTile, e)) is not None:
            return e, tile_component
    return None    

def find_all_sorted(game: GameState, x: int, y: int, dungeon_level: int) -> List[UID]:
    """Returns all entities found at specific coordinates as a list. If any solid entities are present, they are listed first. If any are map tiles, they come last."""
    return sorted(list(game.at(x, y, dungeon_level)), key=lambda e: (game.get(Solid, e), (game.get(MapTile, e) is None)))
    
def check_collision(game: GameState, x: int, y: int, dungeon_level: int, entity: UID) -> bool:
    """
    Checks if there is a collision with any solid entities at the given coordinates.
    
    :param game: The current game state.
    :param x: The x-coordinate to check.
    :param y: The y-coordinate to check.
    :param dungeon_level: The dungeon level to check.
    :param entity: The UID of the entity attempting to move.
    :return: True if there is a collision, False otherwise.
    """
    for other_entity in game.entities():
        other_move = game.get(Move, other_entity)
        other_solid = game.get(Solid, other_entity)
        if other_move and other_solid and other_entity != entity:
            if other_move.x == x and other_move.y == y and other_move.dungeon_level == dungeon_level:
                return True
    return False
