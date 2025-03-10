# allows for use of tools with tools.py in char directory.

import os, importlib, inspect, docstring_parser, json, re, traceback
from pydantic import BaseModel
from ghostbox.util import *
from typing import *
from ghostbox.definitions import *

# # example of tool dict
# "tools": [
#     {
#     "type":"function",
#     "function":{
#         "name":"python",
#         "description":"Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.",
#         "parameters":{
#         "type":"object",
#         "properties":{
#             "code":{
#             "type":"string",
#             "description":"The code to run in the ipython interpreter."
#             }
#         },
#         "required":["code"]
#         }
#     }
#     }
# ],

def type_to_json_schema(type_):
    type_map = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        list: {"type": "array", "items": {}},  # You might need to specify the item type
        dict: {"type": "object"},
        type(None): {"type": "null"}
    }
    if type_ in type_map:
        return type_map[type_]
    elif hasattr(type_, "__origin__"):
        if type_.__origin__ is list:
            return {"type": "array", "items": type_to_json_schema(type_.__args__[0])}
        elif type_.__origin__ is dict:
            return {"type": "object", "additionalProperties": type_to_json_schema(type_.__args__[1])}
        elif type_.__origin__ is Optional:
            return {"oneOf": [type_to_json_schema(type_.__args__[0]), {"type": "null"}]}
    return {}


def makeTools(filepath, display_name="tmp_python_module", tools_forbidden=[]) -> Tuple[List[Tool], Any]:
    """Reads a python file and returns a pair with all the top level functions parsed as tools, and the corresponding module for the file."""
    if not(os.path.isfile(filepath)):
        printerr("warning: Failed to generate tool dictionary for '" + filepath + "' file not found.")
        return ({}, None)


    tools = []
    spec = importlib.util.spec_from_file_location(display_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if name.startswith("_") or not callable(value) or name in tools_forbidden:
            continue
        doc = inspect.getdoc(value)
        if doc is None:
            printerr("error: Missing docstring in function '" + name + "' in file '" + filepath + "'. Aborting tool generation.")
            return ({}, None)
        fulldoc = docstring_parser.parse(doc)
        if fulldoc.description is None:
            printerr("warning: Missing description in function '" + name + "' in file '" + filepath + "'. Please make sure you adhere to standard python documentation syntax.")
            description = doc
        else:
            description = fulldoc.description


        sig = inspect.signature(value)
        paramdocs = {p.arg_name : {"type" : p.type_name, "description" : p.description, "optional" : p.is_optional} for p in fulldoc.params}
        properties = {}
        required_params = []
        for (param_name, param) in sig.parameters.items():
            if param.annotation == inspect._empty:
                printerr("warning: Missing type annotations for function '" + name + "' and parameter '" + param_name + "' in '" + filepath + "'. This will significantly degrade AI tool use performance.")
                # default to str
                param_type = "str"
            else:
                param_type = param.annotation.__name__

            # defaults
            param_description = ""
            param_required = True
            if param_name not in paramdocs:
                printerr("warning: Missing documentation for parameter '" + param_name + "' in function '" + name + "' in '" + filepath + "'. This will significantly degrade AI tool use performance.")
            else:
                p = paramdocs[param_name]
                if p["description"] is None:
                    printerr("warning: Missing description for parameter '" + param_name + "' in function '" + name + "' in '" + filepath + "'. This will significantly degrade AI tool use performance.")
                else:
                    param_description = p["description"]

                #if p["type"] != param_type:
                    #printerr("warning: Erroneous type documentation for parameter '" + param_name + "' in function '" + name + "' in '" + filepath + "'. Stated type does not match function annotation. This will significantly degrade AI tool use performance.")

                if p["optional"] is not None:
                    param_required = not(p["optional"])

            # finally set the payload
            if param_required:
                required_params.append(param_name)
            properties[param_name] = Property(type=type_to_json_schema(param_type).get("type", "string"),
                                              description=param_description)


        parameters = Parameters(required=required_params,
                                properties=properties)
        new_function = Function(name=name,
                                    description=description,
                                    parameters=parameters)
        if new_function.name not in tools_forbidden:
            tools.append(Tool(function=new_function))



    return (tools, module)

def tryParseToolUse(w, start_string = "```json", end_string = "```", magic_word="Action:"):
    """Process AI output to see if tool use is requested. Returns a dictionary which is {} if parse failed, and the input string with json removed on a successful parse.
    :param w: The input string, e.g. AI generated response.
    :param predicate: Optional boolean filter function which takes tool names as input.
    :return: A pair of (list(dict), str), with the parsed json and the input string with json removed if parse succeeded."""
    m = re.match(magic_word + ".*" + start_string + "(.*)" + end_string + ".*", w, flags=re.DOTALL)
    if not(m):
        return {}, w

    try:
        capture = m.groups(1)[-1]
        tools_requested = json.loads(capture)
    except:
        printerr("warning: Exception while trying to parse AI tool use.\n```" + w + "```")
        printerr(traceback.format_exc())
        return {}, w


    if type(tools_requested) != list:
        printerr("warning: Wrong type of tool request. Parse succeeded but no tool application possible.")
        printerr("Dump: \n" + json.dumps(tools_requested, indent=4))
        
    # parse succeeded, clean the input
    w_clean = w.replace(start_string + capture + end_string, "").replace(magic_word, "")
    return (tools_requested, w_clean)


def tryParseAllowedToolUse(w : str,
                           tools_allowed : dict):
    return tryParseToolUse(w, predicate=lambda tool_name: tool_name in allowed_tools.keys())


def getPositionalArguments(func):
    return [param.name for (k, param) in inspect.signature(func).parameters.items() if param.default == inspect._empty]

def getOptionalArguments(func):
    return [param.name for (k, param) in inspect.signature(func).parameters.items() if param.default != inspect._empty]

def makeToolResult(tool_name, result, tool_call_id) -> ChatMessage:
    """Packages a tool call result as a ChatMessage."""
    return ChatMessage(
        role="tool",
        tool_name= tool_name,
        tool_call_id= tool_call_id,
        content=str(result))
            


def makeToolSystemMsg(tools):
    """Deprecated"""
    w = ""
    w += "    ## Available Tools\nHere is a list of tools that you have available to you:\n\n"
    w += json.dumps(tools, indent= 4)
    w += "\n\n"
    return w


def makeToolInstructionMsg():
    """Deprecated"""
    #FIXME: this is currently designed only for command-r, other llms will use different special tokens, for which we have to extend the templating, probably iwth tool_begin and tool_end 
    w = """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>
Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:  

Action:```json
[
    {
        "tool_name": title of the tool in the specification,
        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters
    }
]
```

<|END_OF_TURN_TOKEN|>"""
    return w


def showToolResult(tool_result, indent=0):
    """Takes a tool result of any type and returns a string that can be passed to an AI. Contains no special tokens. Expects whatever is in the 'output' field of the tool use dictionary. If tool_result is a list or dictionary, this function will be recursively appplied."""
    x = tool_result
    pad = " " * indent
    if type(x) == type(None):
        return pad + "output: None\n"
    elif type(x) == str:
        return pad + x + "\n"
    elif type(x) == list:
        if x == []:
            # FIXME: not sure what's best here
            return "[]\n"
        return (pad + "\n").join([showToolResult(y, indent=indent) for y in x])
    elif type(x) == dict:
        return (pad + "\n").join([k + ": " + showToolResult(v, indent=indent+4) for (k, v) in x.items()])
    # default to json. if you pass something that isn't json serializable to the AI, we crash and it's your own fault
    # FIXME: also this won't respect indent. maybe that's ok
    try:
        w = json.dumps(x, indent=4)
    except:
        printerr("warning: Couldn't show the result of a tool call.\nHere's the result dump:\n" + str(x) + "\n and here's the traceback:\n")
        printerr(traceback.format_exc())
        return ""
    return w
    
