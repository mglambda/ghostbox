import pygame
from dataclasses import dataclass
from roguelike_types import *
from typing import List, Tuple


@dataclass
class PyGameView:
    screen_width: int
    screen_height: int
    _last_center_x: int = 0
    _last_center_y: int = 0
    _grid_size: int = 0
    _map_width: int = 0
    _map_height: int = 0
    _status_width: int = 0
    _messages_height: int = 0

    def __post_init__(self):
        # Calculate grid size based on screen dimensions
        self._grid_size = min(
            self.screen_width // 40, self.screen_height // 20
        )  # Assuming 40x20 grid for map
        self._map_width = int(self.screen_width * 0.8)
        self._map_height = int(self.screen_height * 0.8)
        self._status_width = int(self.screen_width * 0.2)
        self._messages_height = int(self.screen_height * 0.2)

    def draw_map(
        self,
        game: GameState,
        center_x: int,
        center_y: int,
        dungeon_level: int,
        screen: pygame.Surface,
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

    def draw_status(
        self, game: GameState, player_uid: UID, screen: pygame.Surface
    ):
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
