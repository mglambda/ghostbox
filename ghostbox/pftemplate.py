import os, glob
from abc import ABC, abstractmethod
from functools import *
from ghostbox.util import *

class PFTemplate(ABC):
    """Abstract base class for prompt format templates, used to turn Story objects into properly formatted strings."""


    @abstractmethod
    def header(self, system_msg, **kwargs):
        pass

    @abstractmethod
    def body(self, story, hint="", **kwargs):
        pass

class FilePFTemplate(PFTemplate):
    """Simple, customizable prompt format templates based on loading dictionaries with certain files."""
    var_decorator = lambda w: "{{" + w + "}}"
    
    def __init__(self, dir):
        self.dir = dir
        self._loadFiles()

    def _loadFiles(self):
        if not(os.path.isdir(self.dir)):
            raise FileNotFoundError("Could not find path " + self.dir)

        allfiles = glob.glob(self.dir + "/*")
        for filepath in allfiles:
            filename = os.path.split(filepath)[1]
            if os.path.isfile(filepath):
                self.__dict__[filename] = open(filepath, "r").read()
                
    def header(self, system_msg, **kwargs):
        return replaceFromDict(self.system.replace("{{system_msg}}", system_msg), kwargs, key_func=FilePFTemplate.var_decorator)

    def body(self, story, append_hint=False, **kwargs):
        def build(w, item):
            # you could do this more modular but why? this way users see the files and the template scheme is obvious. I bet this covers 99% of actual use cases for LLM
            content = replaceFromDict(item["content"], kwargs, key_func=FilePFTemplate.var_decorator)
            if item["role"] == "user":
                return w + self.begin_user + content + self.end_user
            elif item["role"] == "assistant":
                return w + self.begin_assistant + content + self.end_assistant
            # throw if people use weird or no roles with this template
            raise ValueError(item["role"] + " is not a valid role for this template.")
        if append_hint:
            hint = replaceFromDict(self.hint, kwargs, key_func=FilePFTemplate.var_decorator)
        else:
            hint = ""
        return reduce(build, story.getData(), "") + hint

from ghostbox.Story import *

s = Story()
s.addUserText("Hello!")
s.addAssistantText("Yes. How may I help you?")
