from abc import ABC, abstractmethod
from functools import *
from ghostbox.util import *

class OutputFormatter(ABC):
    """An interface for formatting the chat history. The OutputFormatter interface is used both for formatted text that is presented to the user, as well as formatting text to send to the backend, e.g. to prepend 'Bob:' style chat prompts.
OutputFormatters work on both AI and human text and ensure a certain style, without restricting the LLM's grammar outright. Because the LLM will pick up on your style to varying degrees, depending on various factors, the following law must hold for OutputFormatters:

    ```
f = OutPutFormatter()
f.format(w) == f.format(f.format(w))
```

for all strings w. In other words, f is idempotent under repeated application. We might get a string 'How are you, Sally?' which the formatter turns into 'Bob: How are you, Sally?', but we don't want this to turn into 'Bob: Bob: How are you, Sally?' in case the LLM starts to prepend 'Bob: ' itself."""

    @abstractmethod
    def format(self, w):
        pass

    def compose(xs):
        """Takes a list of objects supporting the OutputFormatter interface and returns a ComposedFormatter that is the result of their composition."""
        return reduce(ComposedFormatter, xs, IdentityFormatter())

    def sequence(xs, w):
        """Takes a list of OutputFormatters xs and applies them to w in sequence. Tip: Read the list from left to right and imagine applying each formatter to the string in succession."""
        return reduce(lambda v, f: f.format(v), xs, w)
    
class IdentityFormatter(OutputFormatter):
    """This formatter returns its input unchanged."""
    def __init__(self):
        pass
    
    def format(self, w):
        return w

class ComposedFormatter(OutputFormatter):
    """Allows composition of two given OutputFormatters.
In general, the following laws hold

```
a = OutputFormatter() #i.e. a has superclass OutputFormatter
e = IdentityFormatter()
    c = ComposedFormatter(a, e)

c.format(w) == a.format(w)
    ```

    for any given string w. In other words, IdentityOutputFormatter is the identity element for composition of formatters. Also

    ```
    a = OutputFormatter()
    b = OutputFormatter()
    c1 = ComposedFormatter(a, b)
    c2 = ComposedFormatter(b, a)

    c1.format(w) != c2.format(w)
    ```

    at least not for all a and b. So in general, composition of formatters does not commute."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def format(self, w):
        return self.a.format(self.b.format(w))
    
class NicknameFormatter(OutputFormatter):
    """This formatter prepends chat names, according to a given decorator. By default, names are decorated with colon, as in 'Bob: '."""
    
    def __init__(self, nickname, decorator_func=lambda w: w + ": "):
        self.decorator_func=decorator_func
        self.nickname = nickname

    def setNickname(self, nickname):
        self.nickname = nickname
        return self
        
    def format(self, w):
        def f(v, line, nick=self.decorator_func(self.nickname)):
            if line.startswith(nick):
                return v + "\n" + line
            else:
                return v + "\n" + nick + line
        return reduce(f, w.split("\n"), "")[1:]

    def unformat(self, w):
        def f(v, line, nick=self.decorator_func(self.nickname)):
            if line.startswith(nick):
                return v + "\n" + line[len(nick):]
            else:
                return v + "\n" + line
        return reduce(f, w.split("\n"), "")[1:]

class NicknameRemover(OutputFormatter):
    """Removes prepended nicknames according to a supplied decorator. By default, this removes stuff like 'Bob: ', and it matches it a little fuzzy to account for spaces."""
    def __init__(self, nickname, decorator_func=lambda w: w + ": "):
        self._f = NicknameFormatter(nickname, decorator_func)
        
    def format(self, w):
        return self._f.unformat(w)
    
class IncompleteSentenceCleaner(OutputFormatter):
    """Removes incomplete sentences at the end of text."""

    def __init__(self,     stopchars = '! . ? :'.split(" ")):
        self.stopchars = stopchars
        
    def format(self, w):
        if w == "":
            return w

        stopchars = self.stopchars
        for i in range(len(w)-1, -1, -1):
            if w[i] in stopchars:
                break

        if i == 0:
            return w
        return w[:i+1]

class WhitespaceFormatter(OutputFormatter):
    """Removes whitespace at the beginning and end of text."""

    def format(self, w):
        return w.strip()
    
class LonelyPromptCleaner(OutputFormatter):
    """Removes lonely occurrences of trailing chat prompts. Like trailing 'Bob: ' etc, since these are hard to spot with incomplete sentence cleaning. You can supply another decorator besides the default '<nick>: ' pattern. Also note that cleaning is a little fuzzy when it comes to spaces."""
    def __init__(self, nickname, decorator_func=lambda w: w + ": "):
        self.decorator_func = decorator_func
        self.nickname = nickname

    def setNickname(self, nickname):
        self.nickname = nickname
        return self
        
    def format(self, w):
        prompt = self.decorator_func(self.nickname)
        ws = w.split("\n")
        return "\n".join(filter(lambda v: not(v in prompt), ws))


class ChatFormatter(OutputFormatter):
    """General purpose chat formatter with some extra functionality. Will prepend nicknames according to a decorator, and cleans up some bogus responses."""
    def __init__(self, nickname, decorator_func=lambda w: w + ": "):
        self.decorator_func = decorator_func
        self.nickname = nickname

    def setNickname(self, nickname):
        self.nickname = nickname

    def format(self, w):
        return OutputFormatter.sequence([IncompleteSentenceCleaner(),
                                         LonelyPromptCleaner(self.nickname, decorator_func=self.decorator_func),
                                         NicknameFormatter(self.nickname, decorator_func=self.decorator_func)],
                                        w)
                                                
DoNothing = IdentityFormatter() # more descriptive name                
DefaultResponseCleaner = OutputFormatter.compose([IncompleteSentenceCleaner()])
CleanResponse = DefaultResponseCleaner
DefaultResponseCleaner.__doc__ =     """Does minimum cleanup. Intended for AI responses."""


    
