from dataclasses import dataclass
from contextlib import contextmanager
from typing import Callable
from ghostbox.main import Plumbing
from ghostbox._argparse import makeDefaultOptions
from ghostbox.util import printerr
from ghostbox import commands
from ghostbox.definitions import *

def from_llamacpp(endpoint="http://localhost:8080", **kwargs):
        return Ghostbox(backend="llamacpp", endpoint=endpoint, **kwargs)

def from_koboldcpp(endpoint="http://localhost:5001", **kwargs):
    return Ghostbox(backend="llama.cpp", endpoint=endpoint, **kwargs)



@dataclass
class ChatMessage:
    role : str
    text : str


@dataclass
class ChatResult:
    # not sure yet
    payload : str


@dataclass
class CompletionResult:
    # also not sure yet
    payload : str

class Ghostbox:
    def __init__(self,
                 endpoint : str,
                backend : LLMBackend,
                **kwargs):
        self.endpoint = endpoint
        self.backend = LLMBackend[backend]
        self.__dict__ |= kwargs
        self.__dict__["_plumbing"] = Plumbing(options = makeDefaultOptions().__dict__ | {k : v for (k, v) in self.__dict__.items() if not(k.startswith("_"))})
        # FIXME: set API defaults here
        
        if self.config_file:
            printerr(commands.loadConfig(self._plumbing, [self.config_file]))


    @contextmanager
    def options(self, options : dict):
        # copy old values
        tmp = {k : v for (k, v) in self._plumbing.options.items() if k in options}
        # this has to be done one by one as setoptions has sideffects
        for (new_k, new_v) in options.items():
            self._plumbing.setOption(new_k, new_v)
        yield self
        # now unwind, also one by one
        for (old_k, old_v) in tmp.items():
            self._plumbing.setOption(old_k, old_v)


    @contextmanager
    def option(self, name, value):
        with self.options({name : value}):
            yield self
            
    def set(option_name : str, value) -> None:
        if option_name in self.__dict__:
            self.__dict__[option_name] = value
        self._plumbing.setOption(option_name, value)

    def get(option_name : str) -> object:
        return self._plumbing.getOption(option_name)

    def __getattr__(self, k):
        return self.__dict__["_plumbing"].getOption(k)

    def __setattr__(self, k, v):
        if k == "_plumbing":
            return

        if k in self.__dict__:
            self.__dict__[k] = v

        if "_plumbing" in self.__dict__:
            self.__dict["_plumbing"].setOption(k, v)


    # diagnostics
    def status(self):
        pass
    
    # these are the payload functions
    def text(self,
             prompt_text : str,
             timeout=None) -> str:
        with self.option("stream", False):
            return self._plumbing.interactBlocking(prompt_text, timeout=timeout)

    def text_async(self,
                   prompt_text : str,
                   callback : Callable[[str], None]) -> None:
        with self.option("stream", False):
            # FIXME: this is tricky as we immediately return and set stream = True again ??? what to do
            self._plumbing.interact(prompt_text, generation_callback=callback)
        return
    
    def text_stream(self,
                    prompt_text : str,
                    chunk_callback : Callable[[str], None],
                    generation_callback : Callable[[str], None] = lambda x: None) -> None:
        with self.option("stream", True):
            self._plumbing.interact(prompt_text, generation_callback=generation_callback, stream_callback=chunk_callback)
        return
    
    def json(self, prompt_text : str) -> dict:
        pass
     
    def json_async(self, prompt_text : str, callback : Callable[[dict], None]) -> None:
        pass
    
    def chat(self, user_message : ChatMessage) -> ChatResult:
        pass

    def chat_async(self, user_message : ChatMessage, callback : Callable[[dict], ChatResult]) -> None:
        pass

    def completion(self, prompt_text : str) -> CompletionResult:
        pass

    def completion_async(self, prompt_text : str, callback : Callable[[CompletionResult], None]) -> None:
        pass
