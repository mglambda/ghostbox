import os
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
        return self.template.replace("{$user_msg}", self.memory + self.getNote() + prompt)
        
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

        templatefile = self.dir + "/template"
        if os.path.isfile(templatefile):
            self.template = open(templatefile, "r").read()
            printerr("Using prompt template from " + templatefile)


        templatesystemfile = self.dir + "/template_system"
        if os.path.isfile(templatesystemfile):
            self.template_system = open(templatesystemfile, "r").read()

            
        templateendfile = self.dir + "/template_end"
        if os.path.isfile(templateendfile):
            self.template_end = open(templateendfile, "r").read()
            
        memoryfile = self.dir + "/memory"
        if os.path.isfile(memoryfile):
            self.memory = self._replaceUser(open(memoryfile, "r").read())
            printerr("Initialized memory from " + memoryfile)
        else:
            printerr("Warning: Memory file not found: " + memoryfile)

        notefile = self.dir + "/note"
        if os.path.isfile(notefile):
            self.note = self._replaceUser(open(notefile, "r").read())
            printerr("Initialized author's note from " + notefile)
        else:
            printerr("Warning: Author's note file not found: " + notefile)

        promptfile = self.dir + "/prompt"
        if os.path.isfile(promptfile):
            self.prompt = self._replaceUser(open(promptfile, "r").read())
            printerr("Ok - prompt file found.") 
        else:
            printerr("Warning: No initial prompt found, missing " + promptfile)

        initpromptfile = self.dir + "/initial_prompt"
        if os.path.isfile(initpromptfile):
            self.initial_prompt = self._replaceUser(open(initpromptfile, "r").read())
            self.addText(self.initial_prompt)
            printerr("Ok - initial prompt file found.") 
        else:
            printerr("Warning: No initial prompt found, missing " + initpromptfile)
            
            
            
        
