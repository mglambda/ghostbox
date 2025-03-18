from roguelike_types import *


class EntityMoved(GameResult):
    """Game Result representing an entity moving."""
    entity: UID
    new_x: int
    new_y: int
    
    def handle(self, ctl: Controller) -> None:
        ctl.print(f"Entity {self.entity} moved to ({self.new_x}, {self.new_y})")
        if self.entity == ctl.player:
            from roguelike_controller import reset_focus
            reset_focus(ctl)

class EntityCreated(GameResult):
    """Game Result representing an entity being created."""
    entity: UID

    def handle(self, ctl: Controller) -> None:
        ctl.print(f"Entity {self.entity} created")

class EntityDestroyed(GameResult):
    """Game Result representing an entity being destroyed."""
    entity: UID

    def handle(self, ctl: Controller) -> None:
        ctl.print(f"Entity {self.entity} destroyed")
