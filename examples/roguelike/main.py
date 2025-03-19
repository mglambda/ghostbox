import pygame
from roguelike_types import *
from roguelike_results import *
from roguelike_instructions import *
import roguelike_controller as control
from roguelike_view import *
from roguelike_map import *


def make_minimal_gamestate(player_name: str) -> GameState:
    game = GameState()

    # Create player entity
    player_uid = game.new()
    game.enable(Move, player_uid, Move(x=2, y=2, dungeon_level=0))
    game.enable(Display, player_uid, Display(unicode_character="@", color="white"))
    game.enable(Name, player_uid, Name(name=player_name))
    game.enable(Inventory, player_uid, Inventory(items=[], capacity=10))
    game.enable(Solid, player_uid, Solid())
    game.enable(
        Attributes,
        player_uid,
        Attributes(
            strength=10,
            dexterity=10,
            constitution=10,
            intelligence=10,
            wisdom=10,
            charisma=10,
        ),
    )
    game.enable(GroupMember, player_uid, GroupMember(group_name=Group.Human, rank=0))
    game.enable(Damage, player_uid, Damage(health=100, leaves_corpse=True))
    return game

def make_minimal_room(game: GameState) -> GameState:
    # Create a small room of floor tiles (5x5)
    for x in range(5):
        for y in range(5):
            if x == 0 or x == 4 or y == 0 or y == 4:
                # Create walls
                wall_uid = game.new()
                game.enable(Name, wall_uid, Name(name="Wall"))
                game.enable(Move, wall_uid, Move(x=x, y=y, dungeon_level=0))
                game.enable(
                    Display, wall_uid, Display(unicode_character="#", color="grey")
                )
                game.enable(Matter, wall_uid, Matter(material=Material.Stone))
                game.enable(Solid, wall_uid, Solid())

            # Create floor tiles
            tile_uid = game.new()
            game.enable(MapTile, tile_uid, MapTile())
            game.enable(Name, tile_uid, Name(name="Floor"))
            game.enable(Move, tile_uid, Move(x=x, y=y, dungeon_level=0))
            game.enable(
                Display, tile_uid, Display(unicode_character=".", color="grey")
            )
            game.enable(Matter, tile_uid, Matter(material=Material.Stone))

    return game

def ensure_player_on_floor(game: GameState, player: UID) -> GameState:
    for x in range(1, 200):
        for y in range(1, 200):
            if is_walkable(game, x, y, 0):
                game.enable(Move, player, Move(x=x, y=y, dungeon_level=0))
                return game
            
    raise RuntimeError("No place to put player!")
                    
                            
def main():
    # Initialize Pygame
    pygame.init()
    FPS = 60
    
    # Get the native screen resolution
    screen_info = pygame.display.Info()
    SCREEN_WIDTH = screen_info.current_w
    SCREEN_HEIGHT = screen_info.current_h

    # Set up the display with the native screen resolution
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Roguelike Game")

    # Set up the clock for controlling the frame rate
    clock = pygame.time.Clock()

    # make some defaults for testing
    game = make_minimal_gamestate("Tav")

    # Generate dungeon levels
    for level in range(3):  # Generate 3 levels for testing
        mapgen_generic(game, dungeon_level=level)

    ensure_player_on_floor(game, 0)
    # initialize the controller
    # by convention, the player is UID 0
    ctl = Controller(
        game=game,
        keybindings=control.default_keybindings,
        view=PyGameView(screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT),
        player=0,
    )

    # start the controller's game event loop (different from pygame events)
    control.run(ctl, [DoNothing()])
    
    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                ctl._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                    ctl._running = False
                else:
                    ctl.handle_key_event(event.key)

        # Update the game state
        screen.fill((0, 0, 0))  # Fill the screen with black
        control.draw(ctl, screen)
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
