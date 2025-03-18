import pygame
from roguelike_types import *
from roguelike_instructions import *
from queue import Empty

# this file contains various controller functions
# they are not methods because we want to keep the controller object as minimal as possible
# but they all have essentially private membership in the controller class, and the self argument is used as a controller object throughout

def run(self: Controller, initial_instructions: List[GameInstruction]) -> None:
    """Runs the game on a seperate thread, based on initial instructions.
    This function will use some initial instructions to generate GameResults with the internal GameState, potentially changing it and generating more instructions. It continuously pops instructions of an internal input instruction queue.
    :param initial_instructions: A list of instructions to start the game with, or continue a paused one.
    :return: Nothing. The function runs until it is interrupted or until all instructions have been processed and no new ones are generated. Implicitly, the GameState is returned as a property of the controller.
    """
    self.print(f"Starting game with player uid {self.player}")
    def loop(instructions: List[GameInstruction]):
        self.input_instruction_queue.queue.extendleft(instructions)
        while self._running:
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

    # start the run loop
    t = threading.Thread(target=loop, args=(initial_instructions,), daemon=True)
    self._running = True
    t.start()
    

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


def move_player_left(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=-1, dy=0)])


def move_player_right(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=1, dy=0)])


def move_player_up(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=0, dy=-1)])


def move_player_down(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=0, dy=1)])


def move_player_up_right(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=1, dy=-1)])


def move_player_up_left(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=-1, dy=-1)])


def move_player_down_left(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=-1, dy=1)])


def move_player_down_right(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=1, dy=1)])


def move_player_up_level(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=0, dy=0, dungeon_level_delta=-1)])


def move_player_down_level(self: Controller) -> None:
    self.push_input_instructions([MoveEntity(entity=self.player, dx=0, dy=0, dungeon_level_delta=1)])


def player_wait(self: Controller) -> None:
    self.push_input_instructions([DoNothing()])

def player_confirm(self: Controller) -> None:
    self.confirm()
    
default_keybindings = {
    pygame.K_LEFT: move_player_left,
    pygame.K_RIGHT: move_player_right,
    pygame.K_UP: move_player_up,
    pygame.K_DOWN: move_player_down,
    pygame.K_KP4: move_player_left,
    pygame.K_KP6: move_player_right,
    pygame.K_KP8: move_player_up,
    pygame.K_KP2: move_player_down,
    pygame.K_KP7: move_player_up_left,
    pygame.K_KP9: move_player_up_right,
    pygame.K_KP1: move_player_down_left,
    pygame.K_KP3: move_player_down_right,
    pygame.K_PERIOD: player_wait,
    pygame.K_LESS: move_player_up_level,
    pygame.K_GREATER: move_player_down_level,
    pygame.K_SPACE: player_confirm,   
}


