import os, glob
from ghostbox.util import *
from ghostbox.StoryFolder import *
from ghostbox.agency import *

class Session(object):
    def __init__(self, dir=None, chat_user="", additional_keys=[]):
        self.dir = dir
        self.fileVars = {"chat_user" : chat_user, "system_msg" : "" }
        self.stories = StoryFolder()
        self.tools = {}
        if self.dir is not None:
            self._init(additional_keys)

    def merge(self, other):
        """Merges some things from a session object other into itself. This generally means keeping story etc, of self, but possibly adding fileVars from other, including overriding our own."""
        self.fileVars = self.fileVars | other.fileVars
        self.dir = other.dir
            
    def hasVar(self, var):
        return var in self.fileVars
    
    def getVar(self, var):            
        if var not in self.fileVars:
            printerr("warning: session.getVar: Key not defined '" + var + "'. Did you forget to create " + self.dir + "/" + var + "?")
            return ""
        return self.fileVars[var]

    def getVars(self):
        return self.fileVars
    
    def getSystem(self):
        return self.getVar("system_msg")


    def expandVars(self, w):
        """Expands all variables of the form {{VAR}} in a given string w, if VAR is a key in fileVars."""
        return replaceFromDict(w, self.getVars(), lambda k: "{{" + k + "}}")
    
    def _init(self, additional_keys=[]):
        if not(os.path.isdir(self.dir)):
            raise FileNotFoundError("Could not find path " + self.dir)

        allfiles = glob.glob(self.dir + "/*") + additional_keys
        for filepath in allfiles:
            filename = os.path.split(filepath)[1]
            if os.path.isfile(filepath):
                self.fileVars[filename] = self.expandVars(open(filepath, "r").read())
                # this is useful but too verbose
                #printerr("Found " + filename)

        init_msg = self.fileVars.get("initial_msg", "")
        if init_msg:
            self.stories.get().addAssistantText(init_msg)

        if self.hasVar("tools.py"):
            self.tools = makeToolDict(self.getVar("tools.py"))
                
                
            
