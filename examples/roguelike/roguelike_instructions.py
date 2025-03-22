from roguelike_types import *
from roguelike_results import *
from roguelike_systems import *

class MoveEntity(GameInstruction):
    """Instruction to move an entity to a new position."""
    entity: UID
    dx: int
    dy: int
    dungeon_level_delta: int = 0

    def delta(self, game: GameState) -> DeltaResultType:
        move_component = game.get(Move, self.entity)
        if move_component is None:
            return NothingHappened(), []

        new_x = move_component.x + self.dx
        new_y = move_component.y + self.dy
        new_dungeon_level = move_component.dungeon_level + self.dungeon_level_delta
        # Check for collisions with solid entities
        if check_collision(game, new_x, new_y, move_component.dungeon_level, self.entity):
            try:
                obstacle_name = name_for(game, find_all_sorted(game, new_x, new_y, new_dungeon_level)[0])
            except:
                obstacle_name = "Unkown object"
            return Invalid(invalid_verb="move there", reason=f"{obstacle_name} is in the way.", entity=self.entity), []

        # check for nothingness
        if not(game.is_at(MapTile, new_x, new_y, new_dungeon_level)):
            return Invalid(invalid_verb="move there", reason="That's not part of the map.", entity=self.entity), []
        
        # Update the entity's position
        game.enable(Move, self.entity, Move(x=new_x, y=new_y, dungeon_level=new_dungeon_level))
        return EntityMoved(entity=self.entity, new_x=new_x, new_y=new_y), []

class CreateEntity(GameInstruction):
    """Instruction to create a new entity with given components."""
    components: List[Tuple[type, Any]]

    def delta(self, game: GameState) -> DeltaResultType:
        new_entity = game.new()
        for component_type, component in self.components:
            game.enable(component_type, new_entity, component)
        return EntityCreated(entity=new_entity), []

class DestroyEntity(GameInstruction):
    """Instruction to destroy an entity."""
    entity: UID

    def delta(self, game: GameState) -> DeltaResultType:
        for component_type in game.components():
            game.disable(component_type, self.entity)
        return EntityDestroyed(entity=self.entity), []



class EnableSolid(GameInstruction):
    """Instruction to enable the Solid component for an entity."""
    entity: UID

    def delta(self, game: GameState) -> DeltaResultType:
        game.enable(Solid, self.entity, Solid())
        return EnabledSolid(entity=self.entity), []

class DisableSolid(GameInstruction):
    """Instruction to disable the Solid component for an entity."""
    entity: UID

    def delta(self, game: GameState) -> DeltaResultType:
        game.disable(Solid, self.entity)
        return DisabledSolid(entity=self.entity), []

class EnableDoor(GameInstruction):
    """Instruction to enable the Door component for an entity."""
    entity: UID
    closed: bool
    locked: bool
    on_open: Optional[Script]
    on_close: Optional[Script]
    on_break: Optional[Script]

    def delta(self, game: GameState) -> DeltaResultType:
        game.enable(Door, self.entity, Door(closed=self.closed, locked=self.locked, on_open=self.on_open, on_close=self.on_close, on_break=self.on_break))
        return EnabledDoor(entity=self.entity), []

class DisableDoor(GameInstruction):
    """Instruction to disable the Door component for an entity."""
    entity: UID

    def delta(self, game: GameState) -> DeltaResultType:
        game.disable(Door, self.entity)
        return DisabledDoor(entity=self.entity), []


    
