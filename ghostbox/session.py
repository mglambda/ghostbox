import os, glob, time
from copy import deepcopy
from ghostbox.util import *
from ghostbox.StoryFolder import *
from ghostbox.agency import *


class Session(object):
    special_files = "chat_ai config.json tools.py".split(" ")

    def __init__(
        self, dir=None, chat_user="", chat_ai="", additional_keys=[], tools_forbidden=[]
    ):
        self.dir = dir
        self.fileVars = {
            "chat_user": chat_user,
            "chat_ai": chat_ai,
            "system_msg": "",
            "current_tokens": "0",
            "datetime": getAITime(),
        }
        self.stories = StoryFolder()
        self.tools = []
        self.tools_file = ""
        self.tools_module = None
        if self.dir is not None:
            self._init(additional_keys, tools_forbidden)

    def copy(self):
        # can't deepcopy a module
        tmp = self.tools_module
        self.tools_module = None
        new = deepcopy(self)
        # module becomes a singleton
        self.tools_module = tmp
        new.tools_module = tmp
        return new

    def merge(self, other):
        """Merges some things from a session object other into itself. This generally means keeping story etc, of self, but possibly adding fileVars from other, including overriding our own."""
        self.fileVars = self.fileVars | other.fileVars
        self.dir = other.dir

    def hasVar(self, var):
        return var in self.fileVars

    def getVar(self, var, default=None):
        if var not in self.fileVars:
            if default is None:
                printerr(
                    "warning: session.getVar: Key not defined '"
                    + var
                    + "'. Did you forget to create "
                    + self.dir
                    + "/"
                    + var
                    + "?"
                )
                return ""
            else:
                return default
        return self.expandVars(self.fileVars[var])

    def setVar(self, name, value):
        self.fileVars[name] = value

    def getVars(self):
        return {k: self.expandVars(v) for (k, v) in self.fileVars.items()}

    def getSystem(self, history_retroactive_vars: bool = False):
        if not history_retroactive_vars:
            return self.getVar("system_msg")
        return self.expandVars(self.getVar("system_msg"))
    
    def expandVars(self, w, depth=3):
        """Expands all variables of the form {{VAR}} in a given string w, if VAR is a key in fileVars. By default, will recursively expand replacements to a depth of 3."""
        for i in range(0, depth):
            w_new = replaceFromDict(w, self.fileVars, lambda k: "{{" + k + "}}")
            if w == w_new:
                break
            w = w_new
        return w_new

    def _init(self, additional_keys=[], tools_forbidden=[]):
        if not (os.path.isdir(self.dir)):
            raise FileNotFoundError("Could not find path " + self.dir)

        allfiles = glob.glob(self.dir + "/*") + additional_keys
        for filepath in allfiles:
            filename = os.path.split(filepath)[1]
            if os.path.isfile(filepath) and filename not in self.special_files:
                # self.fileVars[filename] = self.expandVars(open(filepath, "r").read())
                self.fileVars[filename] = open(filepath, "r").read()
                # this is useful but too verbose
                # printerr("Found " + filename)
            elif filename == "tools.py":
                self.tool_file = filepath
                (self.tools, self.tools_module) = makeTools(
                    filepath,
                    display_name=os.path.basename(self.dir) + "_tools",
                    tools_forbidden=tools_forbidden,
                )

        init_msg = self.getVar("initial_msg", "")
        if init_msg:
            self.stories.get().addAssistantText(init_msg)

    def callTool(self, name, params):
        if name not in [tool.function.name for tool in self.tools]:
            return
        try:
            f = getattr(self.tools_module, name)
        except:
            printerr(
                "warning: Failed to call tool '"
                + name
                + "': Not found in module '"
                + self.tools_file
                + "'."
            )
            printerr(traceback.format_exc())
            return

        # we have to build a function call somewhat laboriously because the order of arguments is not guaranteed
        try:
            pargs = [params[arg] for arg in getPositionalArguments(f)]
        except KeyError:
            printerr(
                "warning: Couldn't call tool '"
                + name
                + "': Required positional parameter missing."
            )
            printerr(traceback.format_exc())
            return

        kwargs = {k: v for (k, v) in params.items() if k in getOptionalArguments(f)}

        # here goes nothing
        try:
            result = f(*pargs, **kwargs)
        except:
            printerr(
                "warning: Caught exception when calling tool '"
                + name
                + "'. Here's a dump of the arguments:"
            )
            printerr(json.dumps(params, indent=4))
            printerr("\nAnd here is the full exception:")
            printerr(traceback.format_exc())
            return
        return result

    # new methods
    def get_messages(self, history_retroactive_vars: bool = False) -> List[ChatMessage]:
        """Returns the current active story as a list of chat messages."""
        msgs = self.stories.get().getData()
        # if retroactive is on we expand here, otherwise vars have been expanded when they got added
        if not history_retroactive_vars:
            return msgs

        # ok we have to expand contents
        return [msg.map_content(self.expandVars) for msg in msgs]

    def get_messages_json(self, history_retroactive_vars: bool = False) -> List[Dict[str, Any]]:
        return [msg.model_dump() for msg in self.get_messages(history_retroactive_vars=history_retroactive_vars)]



        
        
