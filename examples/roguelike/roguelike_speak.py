from roguelike_types import *
from roguelike_systems import *


def speak_entity_status(ctl: Controller, which_entity: UID) -> None:
    # for now we just speak some basic info about entity
    if (name_component := ctl.game.get(Name, which_entity)) is None:
        ctl.speak("A strange, bewildering thing.")
        return
    description = (
        "" if name_component.description is None else ": " + name_component.description
    )

    material_w, group_w, interaction_w = "", "", ""
    if (material_comp := ctl.game.get(Material, which_entity)) is not None:
        material_w = material_comp.material

    if (group_comp := ctl.game.get(Group, which_entity)) is not None:
        group_w = group_comp.group

    if ctl.game.has(Interact, which_entity):
        interaction_w = "*interactions*"
        
    ctl.speak(f"{name_component.name} {material_w} {group_w} {interaction_w};" + description)
    return
