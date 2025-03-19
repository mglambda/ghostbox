from roguelike_types import *
from roguelike_systems import *


class EntityMoved(GameResult):
    """Game Result representing an entity moving."""
    entity: UID
    new_x: int
    new_y: int
    
    def handle(self, ctl: Controller) -> None:
        if self.entity == ctl.player:
            name, verb = "You", "move"
        else:
            name, verb = name_for(ctl.game, self.entity), "moves"
            if name == "":
                name = f"Entity {self.entity}"
                
        ctl.speak(f"{name} {verb} to {self.new_x}, {self.new_y}")

        # we do the focus seperately, this is a bit of a hack
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


class EnabledSolid(GameResult):
    """Game Result representing an entity having its Solid component enabled."""
    entity: UID

    def handle(self, ctl: Controller) -> None:
        ctl.log(f"Solid component enabled for entity {self.entity}")

class DisabledSolid(GameResult):
    """Game Result representing an entity having its Solid component disabled."""
    entity: UID

    def handle(self, ctl: Controller) -> None:
        ctl.log(f"Solid component disabled for entity {self.entity}")

class EnabledDoor(GameResult):
    """Game Result representing an entity having its Door component enabled."""
    entity: UID

    def handle(self, ctl: Controller) -> None:
        ctl.log(f"Door component enabled for entity {self.entity}")

class DisabledDoor(GameResult):
    """Game Result representing an entity having its Door component disabled."""
    entity: UID

    def handle(self, ctl: Controller) -> None:
        ctl.log(f"Door component disabled for entity {self.entity}")

        
