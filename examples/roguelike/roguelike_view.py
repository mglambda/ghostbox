import pygame
from dataclasses import dataclass
from roguelike_types import *
from typing import List, Tuple, Optional
import os

@dataclass
class PyGameView(ViewInterface):
    screen_width: int
    screen_height: int
    _last_center_x: int = 0
    _last_center_y: int = 0
    _grid_size: int = 0
    _map_width: int = 0
    _map_height: int = 0
    _status_width: int = 0
    _messages_height: int = 0
    _image_cache: dict = field(default_factory=dict)

    def __post_init__(self):
        # Calculate grid size based on screen dimensions
        self._grid_size = min(
            self.screen_width // 40, self.screen_height // 20
        )  # Assuming 40x20 grid for map
        self._map_width = int(self.screen_width * 0.8)
        self._map_height = int(self.screen_height * 0.8)
        self._status_width = int(self.screen_width * 0.2)
        self._messages_height = int(self.screen_height * 0.2)

    def _load_image(self, path: str) -> pygame.Surface:
        if path not in self._image_cache:
            full_path = os.path.join("img", path)
            self._image_cache[path] = pygame.image.load(full_path).convert_alpha()
        return self._image_cache[path]

    def draw_map(
        self,
        game: GameState,
        center_x: int,
        center_y: int,
        dungeon_level: int,
        screen: pygame.Surface,
        focus: Optional[FocusObject] = None
    ):
        # Calculate the top-left corner of the map to draw
        map_start_x = max(0, center_x - self._map_width // (2 * self._grid_size))
        map_start_y = max(0, center_y - self._map_height // (2 * self._grid_size))
        map_end_x = min(
            game.next_entity_id, map_start_x + self._map_width // self._grid_size
        )
        map_end_y = min(
            game.next_entity_id, map_start_y + self._map_height // self._grid_size
        )

        # Draw the map
        for x in range(map_start_x, map_end_x):
            for y in range(map_start_y, map_end_y):
                entities = list(game.at(x, y, dungeon_level))
                entities = sorted(
                    entities,
                    key=lambda e: (
                        game.get(Solid, e) is not None,
                        game.get(MapTile, e) is not None,
                    ),
                )
                for entity in entities:
                    display = game.get(Display, entity)
                    if display:
                        if display.image:
                            image = self._load_image(display.image)
                            screen.blit(
                                pygame.transform.scale(image, (self._grid_size, self._grid_size)),
                                (
                                    (x - map_start_x) * self._grid_size,
                                    (y - map_start_y) * self._grid_size,
                                ),
                            )
                        else:
                            text_surface = pygame.font.SysFont(
                                "monospace", self._grid_size
                            ).render(
                                display.unicode_character, True, pygame.Color(display.color)
                            )
                            screen.blit(
                                text_surface,
                                (
                                    (x - map_start_x) * self._grid_size,
                                    (y - map_start_y) * self._grid_size,
                                ),
                            )

        if focus is not None:
            # Draw a white border around the focused tile
            if isinstance(focus, FocusTile):
                fx, fy, _ = (
                    focus.which_tile_x,
                    focus.which_tile_y,
                    focus.which_tile_dungeon_level,
                )
                if map_start_x <= fx < map_end_x and map_start_y <= fy < map_end_y:
                    pygame.draw.rect(
                        screen,
                        pygame.Color("white"),
                        pygame.Rect(
                            (fx - map_start_x) * self._grid_size,
                            (fy - map_start_y) * self._grid_size,
                            self._grid_size,
                            self._grid_size,
                        ),
                        2,  # Border width
                    )

    def draw_player_status(self, game: GameState, player_uid: UID, screen: pygame.Surface):
        # Draw player status on the right side of the screen
        if (name_comp := game.get(Name, player_uid)) is None:
            player_name = "Unknown"
        else:
            player_name = name_comp.name

        if (damage_comp := game.get(Damage, player_uid)) is None:
            player_health = 0
        else:
            player_health = damage_comp.health

        if (player_attributes := game.get(Attributes, player_uid)) is None:
            player_attributes = default_attributes

        status_text = [
            f"Name: {player_name}",
            f"Health: {player_health}",
            f"Strength: {player_attributes.strength}",
            f"Dexterity: {player_attributes.dexterity}",
            f"Constitution: {player_attributes.constitution}",
            f"Intelligence: {player_attributes.intelligence}",
            f"Wisdom: {player_attributes.wisdom}",
            f"Charisma: {player_attributes.charisma}",
        ]
        for i, line in enumerate(status_text):
            text_surface = pygame.font.SysFont(
                "monospace", self._grid_size // 2
            ).render(line, True, pygame.Color("white"))
            screen.blit(
                text_surface,
                (
                    self.screen_width - self._status_width + 10,
                    i * (self._grid_size // 2) + 10,
                ),
            )

    def draw_entity_status(self, game: GameState, entity_uid: UID, screen: pygame.Surface):
        # Draw entity status on the right side of the screen
        status_text = []

        # Entity name
        if (name_comp := game.get(Name, entity_uid)) is not None:
            status_text.append(f"Name: {name_comp.name}")

        # Material
        if (matter_comp := game.get(Matter, entity_uid)) is not None:
            status_text.append(f"Material: {matter_comp.material}")

        # Group
        if (group_comp := game.get(GroupMember, entity_uid)) is not None:
            status_text.append(f"Group: {group_comp.group_name}")

        # Damage status
        if (damage_comp := game.get(Damage, entity_uid)) is not None:
            status_text.append(f"Status: {damage_comp.short_description()}")

        # Description
        if name_comp and name_comp.description:
            status_text.append(f"Description: {name_comp.description}")

        # Interact options
        if (interact_comp := game.get(Interact, entity_uid)) is not None:
            status_text.append("Interact Options:")
            for i, option in enumerate(interact_comp.options, start=1):
                option_description = f"{i}. {option.name}"
                if option.description:
                    option_description += f" - {option.description}"
                status_text.append(option_description)

        # Render the status text
        for i, line in enumerate(status_text):
            text_surface = pygame.font.SysFont(
                "monospace", self._grid_size // 2
            ).render(line, True, pygame.Color("white"))
            screen.blit(
                text_surface,
                (
                    self.screen_width - self._status_width + 10,
                    i * (self._grid_size // 2) + 10,
                ),
            )

    def draw_messages(self, messages: List[str], screen: pygame.Surface):
        # Draw messages at the bottom of the screen
        for i, message in enumerate(messages):
            text_surface = pygame.font.SysFont(
                "monospace", self._grid_size // 2
            ).render(message, True, pygame.Color("white"))
            screen.blit(
                text_surface,
                (
                    10,
                    self.screen_height
                    - self._messages_height
                    + i * (self._grid_size // 2)
                    + 10,
                ),
            )


