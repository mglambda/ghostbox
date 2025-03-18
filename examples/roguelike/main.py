import pygame
from roguelike_types import *
from roguelike_results import *
from roguelike_instructions import *
import roguelike_controller as control
from roguelike_view import *


def make_minimal_gamestate(player_name: str) -> GameState:
    game = GameState()

    # Create player entity
    player_uid = game.new()
    game.enable(Move, player_uid, Move(x=5, y=5, dungeon_level=0))
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

    # Create a small room of floor tiles (5x5)
    for x in range(5):
        for y in range(5):
            if x == 0 or x == 4 or y == 0 or y == 4:
                # Create walls
                wall_uid = game.new()
                game.enable(Move, wall_uid, Move(x=x, y=y, dungeon_level=0))
                game.enable(
                    Display, wall_uid, Display(unicode_character="#", color="grey")
                )
                game.enable(Matter, wall_uid, Matter(material=Material.Stone))
                game.enable(Solid, wall_uid, Solid())
            else:
                # Create floor tiles
                tile_uid = game.new()
                game.enable(Move, tile_uid, Move(x=x, y=y, dungeon_level=0))
                game.enable(
                    Display, tile_uid, Display(unicode_character=".", color="grey")
                )
                game.enable(Matter, tile_uid, Matter(material=Material.Stone))

    return game


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

    # initialize the controller
    # by convention, the player is UID 0
    ctl = Controller(
        game=game,
        view=PyGameView(screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT),
        player=0,
    )

    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

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
