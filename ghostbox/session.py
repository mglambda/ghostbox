import os, glob
from ghostbox.util import *
from ghostbox.StoryFolder import *
        
class Session(object):
    def __init__(self, dir=None, chat_user="", additional_keys=[]):
        self.dir = dir
        self.chat_user = chat_user
        self.fileVars = {"chat_user" : chat_user}
        self.stories = StoryFolder()
        if self.dir is not None:
            self._init(additional_keys)

    def getVar(self, var):
        if var not in self.fileVars:
            printerr("warning: session.getVar: Key not defined '" + var + "'. Did you forget to create " + self.dir + "/" + var + "?")
            return ""
        return self.fileVars[var]

    def getVars(self):
        return self.fileVars
    
    
    def getSystem(self):
        return self.getVar("system_msg")
            
    def _showStory(self, w=None, trim_end=False, apply_filter=False):
        if w is None:
            w = self.stories.showStory()
        if trim_end and self.template_end != "":
            if w.endswith(self.template_end):
                return w[:-len(self.template_end)]

        if apply_filter:
            fs = self.fileVars.get("{$template_filter}", "").split(';;;')
            for filter_string in fs:
                w = w.replace(filter_string, "")                
        return w
    
    def _init(self, additional_keys=[]):
        if not(os.path.isdir(self.dir)):
            raise FileNotFoundError("Could not find path " + self.dir)

        allfiles = glob.glob(self.dir + "/*") + additional_keys
        for filepath in allfiles:
            filename = os.path.split(filepath)[1]
            if os.path.isfile(filepath):
                self.fileVars[filename] = open(filepath, "r").read()

                printerr("Found " + filename)

        init_msg = self.fileVars.get("initial_msg", "")
        if init_msg:
            self.stories.get().addAssistantText(init_msg)
