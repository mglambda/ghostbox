from roguelike_types import *
from roguelike_results import *
from roguelike_instructions import *

def make_minimal_gamestate(player_name: str) -> GameState:
    game = GameState()
    
    # Create player entity
    player_uid = game.new()
    game.enable(Move, player_uid, Move(x=5, y=5, dungeon_level=0))
    game.enable(Display, player_uid, Display(unicode_character='@', color='white'))
    game.enable(Name, player_uid, Name(name=player_name))
    game.enable(Inventory, player_uid, Inventory(items=[], capacity=10))
    game.enable(Solid, player_uid, Solid())
    game.enable(Attributes, player_uid, Attributes(strength=10, dexterity=10, constitution=10, intelligence=10, wisdom=10, charisma=10))
    game.enable(GroupMember, player_uid, GroupMember(group_name=Group.Human, rank=0))
    game.enable(Damage, player_uid, Damage(health=100, leaves_corpse=True))
    
    # Create a small room of floor tiles (5x5)
    for x in range(5):
        for y in range(5):
            if (x == 0 or x == 4 or y == 0 or y == 4):
                # Create walls
                wall_uid = game.new()
                game.enable(Move, wall_uid, Move(x=x, y=y, dungeon_level=0))
                game.enable(Display, wall_uid, Display(unicode_character='#', color='grey'))
                game.enable(Matter, wall_uid, Matter(material=Material.Stone))
                game.enable(Solid, wall_uid, Solid())
            else:
                # Create floor tiles
                tile_uid = game.new()
                game.enable(Move, tile_uid, Move(x=x, y=y, dungeon_level=0))
                game.enable(Display, tile_uid, Display(unicode_character='.', color='grey'))
                game.enable(Matter, tile_uid, Matter(material=Material.Stone))
    
    return game

def main():
    # make some defaults for testing
    game = make_minimal_gamestate("Tav")

    # initialize the controller
    # by convention, the player is UID 0
    ctl = Controller(game=game,
                     view=None,
                     player=0)

    # run it in terminal mode
    ctl.run_terminal([DoNothing()])
    

if __name__ == "__main__":
    main()
