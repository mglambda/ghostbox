import pygame
from roguelike_types import *
from roguelike_instructions import *
from roguelike_speak import *
from queue import Empty

# this file contains various controller functions
# they are not methods because we want to keep the controller object as minimal as possible
# but they all have essentially private membership in the controller class, and the self argument is used as a controller object throughout


def run(ctl: Controller, initial_instructions: List[GameInstruction]) -> None:
    """Runs the game on a seperate thread, based on initial instructions.
    This function will use some initial instructions to generate GameResults with the internal GameState, potentially changing it and generating more instructions. It continuously pops instructions of an internal input instruction queue.
    :param initial_instructions: A list of instructions to start the game with, or continue a paused one.
    :return: Nothing. The function runs until it is interrupted or until all instructions have been processed and no new ones are generated. Implicitly, the GameState is returned as a property of the controller.
    """
    ctl.print(f"Starting game with player uid {ctl.player}")

    def loop(instructions: List[GameInstruction]):
        reset_focus(ctl)
        ctl.input_instruction_queue.queue.extendleft(instructions)
        while ctl._running:
            try:
                instruction = ctl.input_instruction_queue.get(timeout=1)
            except Empty:
                # don't remove this or the thread won't handle signals
                continue

            # the program might want us to wait, for e.g. user confirmation
            ctl.continue_execution.wait()

            # execute the instruction with the current gamestate
            # the GameState will be changed, and we potentially get new instructions
            result, new_instructions = instruction.delta(ctl.game)
            # the result's side effects procure, with access to the controller
            # sound, graphics, in our case printing
            result.handle(ctl)

            # add the new instructions to the beginning
            # (appending them would completely change the game)
            with ctl.input_instruction_queue.mutex:
                ctl.input_instruction_queue.queue.extendleft(new_instructions)

    # start the run loop
    t = threading.Thread(target=loop, args=(initial_instructions,), daemon=True)
    ctl._running = True
    t.start()


def draw(ctl: Controller, screen: pygame.Surface) -> None:
    """Draws the entire game.
    This completely assumes a PyGameView, or a dreived class."""

    # find player coordinates
    if (player_move := ctl.game.get(Move, ctl.player)) is not None:
        # if we don't have a player, we simply don't draw the map
        ctl.view.draw_map(
            ctl.game,
            player_move.x,
            player_move.y,
            player_move.dungeon_level,
            screen,
            focus=ctl.focus,
        )

        # what status shows depends on focus
        match ctl.focus:
            case FocusEntity(which_entity=which_entity):
                ctl.view.draw_entity_status(ctl.game, which_entity, screen)
            case _:
                ctl.view.draw_player_status(ctl.game, ctl.player, screen)
                
    ctl.view.draw_messages(ctl.messages, screen)


def handle_accessibility_focus(
    ctl: Controller, old_focus: FocusObject, new_focus: FocusObject
) -> None:
    """Speaks whatever is appropriate to a certain focus change context."""
    match new_focus:
        case FocusTile(which_tile_x=x, which_tile_y=y, which_tile_dungeon_level=lvl):
            # we don't need to say the dlvl every time
            position = f"at {x}, {y}"
            if (tile_pair := find_floor_tile(ctl.game, x, y, lvl)) is None:
                ctl.speak("Nothing" + position)
                return
            tile_id, tile_component = tile_pair
            if (tile_name_component := ctl.game.get(Name, tile_id)) is None:
                ctl.speak("Unknown tile" + position)
                return

            # ok it's a real tile, now it gets a little more involved
            tile_name = tile_name_component.name

            # what else is here?
            entities = find_all_sorted(ctl.game, x, y, lvl)
            # we know entities has at least 1 member (the tile)
            # if there are more things, we will speak only one
            # and of those things, we will speak whatever is solid (like a monster)
            if (first := entities[0]) == tile_id:
                things_msg = ""
            else:
                if (name_component := ctl.game.get(Name, first)) is not None:
                    things_msg = f" with {name_component.name} "

            # ok now we got everything
            ctl.speak(tile_name + things_msg + position)
            return
        case FocusEntity(which_entity=which_entity):
            speak_entity_status(ctl, which_entity)
            return
        case FocusMessages(which_msg=which_msg):
            # for now we just speak the latest message
            if ctl.messages == []:
                ctl.speak("No messages yet.")
                return
            msg = ctl.messages[-1]
            ctl.speak(msg)
            return
        case FocusStatus():
            # not implemented yet
            pass


def reset_focus(ctl: Controller) -> None:
    ctl.focus = None
    if (move_component := ctl.game.get(Move, ctl.player)) is None:
        # very weird, which is why we get to do weird things
        ctl.last_tile_focused = FocusTile()
        return
    ctl.last_tile_focused = FocusTile(
        which_tile_x=move_component.x,
        which_tile_y=move_component.y,
        which_tile_dungeon_level=move_component.dungeon_level,
    )


def change_focus(ctl: Controller, new_focus: FocusObject) -> None:
    """Changes the focus selection in the itnerface and procs appropriate side effects."""
    if type(ctl.focus) == FocusTile:
        ctl.last_tile_focused = ctl.focus

    handle_accessibility_focus(ctl, ctl.focus, new_focus)
    ctl.focus = new_focus


def move_player_left(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([MoveEntity(entity=ctl.player, dx=-1, dy=0)])


def move_player_right(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([MoveEntity(entity=ctl.player, dx=1, dy=0)])


def move_player_up(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([MoveEntity(entity=ctl.player, dx=0, dy=-1)])


def move_player_down(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([MoveEntity(entity=ctl.player, dx=0, dy=1)])


def move_player_up_right(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([MoveEntity(entity=ctl.player, dx=1, dy=-1)])


def move_player_up_left(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([MoveEntity(entity=ctl.player, dx=-1, dy=-1)])


def move_player_down_left(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([MoveEntity(entity=ctl.player, dx=-1, dy=1)])


def move_player_down_right(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([MoveEntity(entity=ctl.player, dx=1, dy=1)])


def move_player_up_level(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions(
        [MoveEntity(entity=ctl.player, dx=0, dy=0, dungeon_level_delta=-1)]
    )


def move_player_down_level(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions(
        [MoveEntity(entity=ctl.player, dx=0, dy=0, dungeon_level_delta=1)]
    )


def player_wait(ctl: Controller) -> None:
    reset_focus(ctl)
    ctl.push_input_instructions([DoNothing()])


def player_confirm(ctl: Controller) -> None:
    ctl.confirm()


def move_focus_left(ctl: Controller) -> None:
    if isinstance(ctl.focus, FocusTile):
        focus_origin = ctl.focus
    else:
        focus_origin = ctl.last_tile_focused
    new_focus = FocusTile(
        which_tile_x=focus_origin.which_tile_x - 1,
        which_tile_y=focus_origin.which_tile_y,
        which_tile_dungeon_level=focus_origin.which_tile_dungeon_level,
    )
    change_focus(ctl, new_focus)


def move_focus_right(ctl: Controller) -> None:
    if isinstance(ctl.focus, FocusTile):
        focus_origin = ctl.focus
    else:
        focus_origin = ctl.last_tile_focused
    new_focus = FocusTile(
        which_tile_x=focus_origin.which_tile_x + 1,
        which_tile_y=focus_origin.which_tile_y,
        which_tile_dungeon_level=focus_origin.which_tile_dungeon_level,
    )
    change_focus(ctl, new_focus)


def move_focus_up(ctl: Controller) -> None:
    if isinstance(ctl.focus, FocusTile):
        focus_origin = ctl.focus
    else:
        focus_origin = ctl.last_tile_focused
    new_focus = FocusTile(
        which_tile_x=focus_origin.which_tile_x,
        which_tile_y=focus_origin.which_tile_y - 1,
        which_tile_dungeon_level=focus_origin.which_tile_dungeon_level,
    )
    change_focus(ctl, new_focus)


def move_focus_down(ctl: Controller) -> None:
    if isinstance(ctl.focus, FocusTile):
        focus_origin = ctl.focus
    else:
        focus_origin = ctl.last_tile_focused

    new_focus = FocusTile(
        which_tile_x=focus_origin.which_tile_x,
        which_tile_y=focus_origin.which_tile_y + 1,
        which_tile_dungeon_level=focus_origin.which_tile_dungeon_level,
    )
    change_focus(ctl, new_focus)


def select_entity_at_focus(ctl: Controller) -> None:
    if isinstance(ctl.focus, FocusTile):
        x, y, lvl = (
            ctl.focus.which_tile_x,
            ctl.focus.which_tile_y,
            ctl.focus.which_tile_dungeon_level,
        )
        entities = find_all_sorted(ctl.game, x, y, lvl)
        if entities:
            change_focus(ctl, FocusEntity(which_entity=entities[0]))


def help_dump_model(ctl: Controller) -> None:
    import json

    print(json.dumps(ctl.game.model_dump(), indent=4))


def interact_with_entity(ctl: Controller, option_index: int) -> None:
    if isinstance(ctl.focus, FocusEntity):
        entity = ctl.focus.which_entity
        if (interact_comp := ctl.game.get(Interact, entity)) is not None:
            if 0 <= option_index < len(interact_comp.options):
                option = interact_comp.options[option_index]
                ctl.messages.append(f"Used {option.name} on {name_for(ctl.game, entity)}")
                ctl.push_input_instructions(option.script.instructions)
            else:
                ctl.print("Invalid interaction option.")
        else:
            ctl.print("Entity has no interaction options.")
    else:
        ctl.print("No entity in focus.")


# Generate keybindings for numbers 1-9
#for i in range(1, 10):
#    default_keybindings[getattr(pygame, f"K_{i}")] = lambda ctl, i=i: interact_with_entity(ctl, i - 1)

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
    pygame.K_a: move_focus_left,
    pygame.K_d: move_focus_right,
    pygame.K_w: move_focus_up,
    pygame.K_s: move_focus_down,
    pygame.K_e: select_entity_at_focus,
    pygame.K_h: help_dump_model,
    **{getattr(pygame, f"K_{i}"): lambda ctl, i=i: interact_with_entity(ctl, i - 1) for i in range(1, 10)}
}

