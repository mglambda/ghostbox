
# allows for use of tools with tools.py in char directory.

import os, importlib, inspect, marshal
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
    if not(os.path.isfile(filepath)):
        printerr("warning: Failed to generate tool dictionary for '" + filepath + "' file not found.")
        return {}


    tools = []
    spec = importlib.util.spec_from_file_location(display_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if name.startswith("_") or not callable(value):
            continue
        doc = inspect.getdoc(value)
    code = marshal.dumps(value.__code__)
    tools.append({"name" : name,
                  "description" : doc})


    return tools
