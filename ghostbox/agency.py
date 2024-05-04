
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


def tryParseToolUse(w, predicate=lambda tool_name: True):
    """Process AI output to see if tool use is requested. Returns a dictionary which is {} if parse failed."""
    m = re.match(".*json(.*)```.*", w, flags=re.DOTALL)
    if not(m):
        return {}

    try:
        tools_requested = json.loads(m.groups(1)[-1])
    except:
        printerr("warning: Exception while trying to parse AI tool use.\n```" + w + "```")
        printerr(traceback.format_exc())
        return {}

    return {func : params for (func, params) in tools_requested.items() if predicate(func)}

def tryParseAllowedToolUse(w : str,
                           tools_allowed : dict):
    return tryParseToolUse(w, predicate=lambda tool_name: tool_name in allowed_tools.keys())


def getPositionalArguments(func):
    return [param.name for (k, param) in inspect.signature(func).parameters.items() if param.default == inspect._empty]

def getOptionalArguments(func):
    return [param.name for (k, param) in inspect.signature(func).parameters.items() if param.default != inspect._empty]



            
        
        
