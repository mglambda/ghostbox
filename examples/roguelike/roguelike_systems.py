from roguelike_types import *

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
