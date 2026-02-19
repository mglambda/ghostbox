import argparse, os
from .util import *
from . import backends
from .definitions import Config, ArgumentTag, ArgumentType, ArgumentGroup, LLMBackend, TTSModel, ZonosTTSModel, TTSOutputMethod, get_ghostbox_data
from typing import Dict, Any, List, Optional, Union, get_origin, get_args, Type
from pydantic_core import PydanticUndefined
from enum import Enum



class TaggedArgumentParser:
    """Creates an argument parser along with a set of tags for each argument.
    Arguments to the constructor are passed on to argparse.ArgumentParser.__init__ .
    You can then use add_arguments just like with argparse, except that there is an additional keyword argument 'tags', which is a dictionary that will be associated with that command line argument.
    """

    def __init__(self, **kwargs: Any):
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(**kwargs)
        self.tags: Dict[str, ArgumentTag] = {}

    def add_argument(self, *args: Any, **kwargs: Any) -> None:
        if "tag" in kwargs:
            arg: str = (
                sorted(args, key=lambda w: len(w), reverse=True)[0]
                .strip("-")
                .replace("-", "_")
            )
            self.tags[arg] = kwargs["tag"]
            del kwargs["tag"]

        self.parser.add_argument(*args, **kwargs)

    def get_parser(self) -> argparse.ArgumentParser:
        return self.parser

    def get_tags(self) -> Dict[str, ArgumentTag]:
        return self.tags


def makeTaggedParser() -> TaggedArgumentParser:
    parser: TaggedArgumentParser = TaggedArgumentParser(
        description="LLM Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    for field_name, field_info in Config.model_fields.items():

        # Skip internal/hidden fields
        if field_name.startswith("__"): 
            continue

        extracted_arg_tag: Optional[ArgumentTag] = None
        
        # Get all arguments from the Annotated type, including the actual type and metadata
        annotated_args = get_args(field_info.annotation)

        # the ARgumenTag will be stored in the matadata
        for annotation_arg in field_info.metadata:
            if isinstance(annotation_arg, ArgumentTag):
                extracted_arg_tag = annotation_arg.model_copy(deep=True) # type: ignore[call-arg]
                break # Found the tag, no need to check other metadata args for this purpose
        
        if extracted_arg_tag is None:

            continue

        # Now, use field_info directly for argparse metadata and defaults
        # field_info.json_schema_extra is the correct place for it.
        if not (
            isinstance(field_info.json_schema_extra, dict)
            and "argparse" in field_info.json_schema_extra
        ):
            continue
        
        argparse_meta = field_info.json_schema_extra["argparse"]


        # Populate name, help, and default_value in the ArgumentTag instance
        extracted_arg_tag.name = field_name
        extracted_arg_tag.help = field_info.description if field_info.description else ""

        # Determine default value for the ArgumentTag
        if field_info.default_factory:
            extracted_arg_tag.default_value = field_info.default_factory()
        elif field_info.default is not PydanticUndefined:
            extracted_arg_tag.default_value = field_info.default
        elif get_origin(field_info.annotation) is Union and type(None) in get_args(field_info.annotation):
            extracted_arg_tag.default_value = None
        else:
            printerr(f"warning: Config option {field_name} has no legal default value.")
            extracted_arg_tag.default_value = None

        # Prepare kwargs for parser.add_argument
        kwargs: Dict[str, Any] = {}

        # Short and long arguments
        short_arg = argparse_meta.get("short")
        long_arg = argparse_meta.get("long", f"--{field_name}")
        
        args_list = []
        if short_arg:
            args_list.append(short_arg)
        args_list.append(long_arg)

        # Help message
        kwargs["help"] = field_info.description

        # Default value for argparse.add_argument
        if not argparse_meta.get("boolean_optional_action"):
            if field_info.default_factory:
                kwargs["default"] = field_info.default_factory()
            elif field_info.default is not PydanticUndefined:
                kwargs["default"] = field_info.default
            elif get_origin(field_info.annotation) is Union and type(None) in get_args(field_info.annotation):
                kwargs["default"] = None
            elif argparse_meta.get("action") == "append":
                kwargs["default"] = []
        else:
            # booleanoptionalaction
            kwargs["default"] = field_info.default
            
        # Type
        cli_type = argparse_meta.get("type")
        if cli_type:
            kwargs["type"] = cli_type
        else:
            raw_type = get_args(field_info.annotation)

            origin = get_origin(raw_type)
            type_args = get_args(raw_type)

            if origin is Union:
                non_none_args = [arg for arg in type_args if arg is not type(None)]
                if len(non_none_args) == 1:
                    raw_type = non_none_args[0]
                    origin = get_origin(raw_type)
                    type_args = get_args(raw_type)
                else:
                    kwargs["type"] = str

            if isinstance(raw_type, type) and issubclass(raw_type, Enum):
                kwargs["type"] = str
                kwargs["choices"] = [e.value for e in raw_type]
            elif raw_type is bool:
                pass
            elif raw_type in (int, float, str):
                kwargs["type"] = raw_type
            elif origin is list:
                if type_args:
                    list_element_type = type_args[0]
                    if isinstance(list_element_type, type) and issubclass(list_element_type, Enum):
                        kwargs["type"] = str
                        kwargs["choices"] = [e.value for e in list_element_type]
                    elif list_element_type in (int, float, str):
                        kwargs["type"] = list_element_type
                    else:
                        kwargs["type"] = str
                else:
                    kwargs["type"] = str
            elif origin is dict or raw_type is Any:
                kwargs["type"] = str
            else:
                kwargs["type"] = str

        # Action
        action = argparse_meta.get("action")
        if action:
            kwargs["action"] = action
        elif argparse_meta.get("boolean_optional_action"):
            kwargs["action"] = argparse.BooleanOptionalAction
            kwargs.pop("type", None)

        # Nargs
        nargs = argparse_meta.get("nargs")
        if nargs:
            kwargs["nargs"] = nargs
            if "default" in kwargs and kwargs["default"] is None and nargs in ("*", "+") and kwargs.get("action") != "append":
                kwargs["default"] = []
            elif "default" not in kwargs and nargs in ("*", "+") and kwargs.get("action") != "append":
                kwargs["default"] = []


        parser.add_argument(*args_list, tag=extracted_arg_tag, **kwargs)


    return parser


def makeDefaultOptions() -> Tuple[argparse.Namespace, Dict[str, ArgumentTag]]:
    """Returns a pair of default options and tags."""
    tp: TaggedArgumentParser = makeTaggedParser({}) 
    parser: argparse.ArgumentParser = tp.get_parser()
    return parser.parse_args(args=""), tp.get_tags()
