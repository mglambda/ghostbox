import pygame
from roguelike_types import *
from queue import Empty

# this file contains various controller functions
# they are not methods because we want to keep the controller object as minimal as possible
# but they all have essentially private membership in the controller class, and the self argument is used as a controller object throughout


def run(self: Controller, instructions: List[GameInstruction]) -> None:
    """Runs the game on a seperate thread, based on initial instructions.
    This function will use some initial instructions to generate GameResults with the internal GameState, potentially changing it and generating more instructions. It continuously pops instructions of an internal input instruction queue.
    :param instructions: A list of instructions to start the game with, or continue a paused one.
    :return: Nothing. The function runs until it is interrupted or until all instructions have been processed and no new ones are generated. Implicitly, the GameState is returned as a property of the controller.
    """
    self.print(f"Starting game with player uid {self.player}")
    def loop():
        while True:
            try:
                instruction = self.input_instruction_queue.get(timeout=1)
            except Empty:
                # don't remove this or the thread won't handle signals
                continue

            # the program might want us to wait, for e.g. user confirmation
            self.continue_execution.wait()
            
            # execute the instruction with the current gamestate
            # the GameState will be changed, and we potentially get new instructions
            result, new_instructions = instruction.delta(self.game)
            # the result's side effects procure, with access to the controller
            # sound, graphics, in our case printing
            result.handle(self)

            # add the new instructions to the beginning
            # (appending them would completely change the game)
            with self.input_instruction_queue.mutex:
                self.input_instruction_queue.queue.extendleft(new_instructions)


def draw(self: Controller, screen: pygame.Surface) -> None:
    """Draws the entire game.
    This completely assumes a PyGameView, or a dreived class."""

    # find player coordinates
    if (player_move := self.game.get(Move, self.player)) is not None:
        # if we don't have a player, we simply don't draw the map
        self.view.draw_map(
            self.game, player_move.x, player_move.y, player_move.dungeon_level, screen
        )
    self.view.draw_status(self.game, self.player, screen)
    self.view.draw_messages(self.messages, screen)
