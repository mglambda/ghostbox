
# allows for use of tools with tools.py in char directory.

import os, importlib, inspect, docstring_parser, json, re, traceback
from ghostbox.util import *





# Here's an example what a function/tool dictionary might look like. Taken from cohere's command-r documentation.
# This should work with other models except command-r btw, it's just that I think this is a reasonable json schema.
#    tools = [{
#        "name": "query_daily_sales_report",
#        "description": "Connects to a database to retrieve overall sales volumes and sales information for a given day.",
#        "parameter_definitions": {
#            "day": {
#                "description": "Retrieves sales data for this day, formatted as YYYY-MM-DD.",
#                "type": "str",
#                "required": True
#            }
#        }
#    },
#    {
#        "name": "query_product_catalog",
#        "description": "Connects to a a product catalog with information about all the products being sold, including categories, prices, and stock levels.",
#        "parameter_definitions": {
#            "category": {
#                "description": "Retrieves product information data for all products in this category.",
#                "type": "str",
#                "required": True
#            }
#        }
#    }
# ]

def makeToolDicts(filepath, display_name="tmp_python_module"):
    """Returns a pair of (tool_dict, module)."""
    if not(os.path.isfile(filepath)):
        printerr("warning: Failed to generate tool dictionary for '" + filepath + "' file not found.")
        return ({}, None)


    tools = []
    spec = importlib.util.spec_from_file_location(display_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if name.startswith("_") or not callable(value):
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

        parameters = {}
        sig = inspect.signature(value)
        paramdocs = {p.arg_name : {"type" : p.type_name, "description" : p.description, "optional" : p.is_optional} for p in fulldoc.params}
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
            parameters[param_name] = {"type" : param_type,
                                      "description" : param_description,
                                      "required" : param_required}
        tools.append({"name" : name,
                      "description" : description,
                      "parameter_definitions" : parameters})

    return (tools, module)


def tryParseToolUse(w, predicate=lambda tool_name: True, start_string = "```json", end_string = "```"):
    """Process AI output to see if tool use is requested. Returns a dictionary which is {} if parse failed, and the input string with json removed on a successful parse."""
    m = re.match(".*" + start_string + "(.*)" + end_string + ".*", w, flags=re.DOTALL)
    if not(m):
        return {}, w

    try:
        capture = m.groups(1)[-1]
        tools_requested = json.loads(capture)
    except:
        printerr("warning: Exception while trying to parse AI tool use.\n```" + w + "```")
        printerr(traceback.format_exc())
        return {}, w


    if type(tools_requested) != dict:
        printerr("warning: Wrong type of tool request. Parse succeeded but no tool application possible.")
        printerr("Dump: \n" + json.dumps(tools_requested, indent=4))
        
    # parse succeeded, clean the input
    w_clean = w.replace(start_string + capture + end_string, "")
    
    return {func : params for (func, params) in tools_requested.items() if predicate(func)}, w_clean

def tryParseAllowedToolUse(w : str,
                           tools_allowed : dict):
    return tryParseToolUse(w, predicate=lambda tool_name: tool_name in allowed_tools.keys())


def getPositionalArguments(func):
    return [param.name for (k, param) in inspect.signature(func).parameters.items() if param.default == inspect._empty]

def getOptionalArguments(func):
    return [param.name for (k, param) in inspect.signature(func).parameters.items() if param.default != inspect._empty]

def makeToolResult(tool_name, params, result):
    """Packages a tool call result in a dictionary."""
    return {
        "call" : {
            "name" : tool_name,
            "parameters" : params},
        "output" : result}
            


def makeToolSystemMsg(tools, example=True, instructions=True):
    example_str = """When it is appropriate to use one of your tools, output your tool use with their respective parameters in the json format, like this
```json
{ 
  "<TOOL NAME1>" :
  {
    "<PARAMETER 1>" : <VALUE 1,
	"<PARAMETER 2>" : <VALUE 2>},
  "<TOOL NAME2>" : {}}"""

    w = ""
    w +="# Tools\n"    
    if example:
        w += example_str
    w += "\n## Tools Available\nHere are the tools available to you, as a json dictionary.\n\n"
    w += json.dumps(tools, indent= 2)
    if instructions:
        #w += 'Invoke them by outputting json as described. The json for a tool call should come at the end of your output. The tools will be applied user side, and you will receive a structured json list, with each tool you called and its respective output. Refer to the contents of the "output" field for a tools results.\n'
        w += 'Invoke a tool by outputting json as described. Only output the json and nothing else. The tools will be applied user-side, and their respective output will be at the start of your message. Refer to the "output" field for their respective return values.'
        w += "nly talk to the user about the tools in general terms. Keep it vague and non-technical.\n"
        w += "If a tool allows you to retrieve, read, refer, fetch, pull, inspect, or otherwise gather data and the user requests it, give a simple and terse confirmation, and then output the json. The data will be available after you do this."
    return w
