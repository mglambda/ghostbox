import os, glob
from kbcli_util import *


class Session(object):
    def __init__(self, dir=None, chat_user=""):
        self.dir = dir
        self.chat_user = chat_user
        self.memory = ""
        self.note = ""
        self.prompt = ""
        self.initial_prompt = ""
        self.worldinfo = ""
        self.template_system = ""
        self.template = ""
        self.template_end = ""
        self.keys = {}
        self.story = []
        if self.dir is not None:
            try:
                self._init()
            except:
                printerr("Error loading dir " + dir)

    def hasTemplate(self):
        return self.template != ""

    def injectTemplate(self, prompt):
        # prompt is any user provided string, and will be replacing {$user_msg} which is supposed to be in the template somewhere
        w = self.template.replace("{$user_msg}", self.memory + self.getNote() + prompt)
        for (key, v) in self.keys.items():
            w.replace(key, v) #replaces filenames found in dir with their conten, e.g. 'memory' has contents that will be spliced into {$memory}
        return w
        
    def addText(self, w):
        self.story.append(w)

    def getStory(self):
        return "".join(self.story)

    def getNote(self):
        if not(self.note):
            return ""
        return "\n[" + self.note + "]\n"

    def _replaceUser(self, w):
        return w.replace("{$user}", self.chat_user)
    
    def _init(self):
        if not(os.path.isdir(self.dir)):
            return

        for filepath in glob.glob(self.dir + "/*"):
            filename = os.path.split(filepath)[1]
            if os.path.isfile(filepath):
                self.__dict__[filename] = open(filepath, "r").read()
                self.keys["{$" + filename + "}"] = self.__dict__[filename]
                printerr("Found " + filename)
