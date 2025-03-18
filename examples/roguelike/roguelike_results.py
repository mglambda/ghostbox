from roguelike_types import *


class EntityMoved(GameResult):
    """Game Result representing an entity moving."""
    entity: UID
    new_x: int
    new_y: int

    def handle(self, ctl: Controller) -> None:
        ctl.print(f"Entity {self.entity} moved to ({self.new_x}, {self.new_y})")

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
