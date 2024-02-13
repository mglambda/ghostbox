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
    def body(self, story, append_hint=True, **kwargs):
        pass

    @abstractmethod
    def strip(self, w):
        pass

    @abstractmethod
    def stops(self):
        """Returns a list of strings that may stop generation. This is intended for EOS delimiters, like <|im_end|> etc."""
        pass
    
class FilePFTemplate(PFTemplate):
    """Simple, customizable prompt format templates based on loading dictionaries with certain files.

Files expected:
    system - Will be prepended to every prompt. It should contain '{{system_msg}}', which will be replaced by the actual content of the system prompt when the header method is called.
    begin_user - Contains string that will be prepended to user messages. Be sure to include newlines, if you want them
    end_user - Contains string that will be appended to user message.
    begin_assistant - Contains string that will be prepended to generated AI message. This may be the same as begin_user, or it may differ.
    end_assistant - Contains string that will be appended to the generated AI message.
    hint - Contains a special string that is sometimes appended at the end of a user message. It should contain a string that guides the AI's compleetion, e.g. this may just be '<|im_start|>assistant\n' in the case of chat-ml, which will heavily discourage the LLM from speaking for the user. This is only appended when append_hint is True in the body method.

All methods accept additional **kwargs, which contain replacements for double curly braced strings in the story content and system message. Things like '{{char_name}}' etc.
Example:
    ```
from ghostbox.Story import *
s = Story()
s.addUserText("The {{adjective}}, brown fox jumps over the lazy hedgehog!")
t = FilePFTemplate("templates/chat-ml")
print(t.body(s, append_hint=True, adjective="quick"))
```

Output:
    
```
The quick, brown fox jumps over the lazy hedgehog!<|im_end|><|im_start|>assistant

```
"""    

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

    def body(self, story, append_hint=True, **kwargs):
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
    def stops(self):
        return [self.end_assistant]
        
    
    def strip(self, w):
        #FXIME: only preliminary for testing like this
        targets = [self.begin_user, self.begin_assistant, self.end_user, self.end_assistant]
        return reduce(lambda v, target: v.replace(target, ""), targets, w)
        


class RawTemplate(PFTemplate):
    """This is a dummy template that doesn't do anything. Perfect if you want to experiment."""
    def header(self, system_msg, **kwargs):
        return system_msg


    def body(self, story, append_hint=True, **kwargs):
        return "\n".join([item["content"] for item in story.getData()])

    def strip(self, w):
        return w
