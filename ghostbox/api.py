from __future__ import annotations
import traceback
from pydantic import BaseModel
from typing import *
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Callable, Dict
from typing_extensions import Self
import json
from ghostbox.main import Plumbing, setup_plumbing
from ghostbox.StoryFolder import StoryFolder
from ghostbox._argparse import makeDefaultOptions
from ghostbox.util import printerr
from ghostbox import commands
from ghostbox.definitions import *
from ghostbox import definitions
from ghostbox.api_internal import *
from ghostbox.agency import Tool, Function, Property, Parameters


def from_generic(endpoint="http://localhost:8080", **kwargs):
    """Returns a Ghostbox instance that connects to an OpenAI API compatible endpoint.
    This generic backend adapter works with many backends, including llama.cpp, llama-box, ollama, as well as online providers, like OpenAI, Anthropic, etc. However, to use features specific to a given backend, that are not part of the OpenAI API, you may need to use a more specific backend.
    Note: Expects ENDPOINT to serve /v1/chat/completions and similar, so e.g. http://localhost:8080/v1/chat/completions should be reachable.
    """
    return Ghostbox(backend=LLMBackend.generic, endpoint=endpoint, **kwargs)


def from_openai_legacy(endpoint="http://localhost:8080", **kwargs):
    """Returns a Ghostbox instance that connects to an OpenAI API compatible endpoint using the legacy /v1/completions interface.
    This generic backend adapter works with many backends, including llama.cpp, llama-box, ollama, as well as online providers, like OpenAI, Anthropic, etc. However, to use features specific to a given backend, that are not part of the OpenAI API, you may need to use a more specific backend.
    Note: There is usually no reason to use this over the generic variant."""
    return Ghostbox(backend=LLMBackend.legacy, endpoint=endpoint, **kwargs)


def from_llamacpp(endpoint="http://localhost:8080", **kwargs):
    """Returns a Ghostbox instance bound to the formidable LLama.cpp. See https://github.com/ggml-org/llama.cpp .
    This uses endpoints described in the llama-server documentation, and will make use of Llama.cpp specific features.
    """
    return Ghostbox(backend=LLMBackend.llamacpp, endpoint=endpoint, **kwargs)


# FIXME: temporarily disabled due to being untested
# ndef from_koboldcpp(endpoint="http://localhost:5001", **kwargs):
#    return Ghostbox(backend="llama.cpp", endpoint=endpoint, **kwargs)


def from_openai_official():
    """Returns a Ghostbox instance that connects to the illustrious OpenAI API at their official servers.
    The endpoint is hardcoded for this one. Use the 'generic' backend to connect to arbitrary URLs using the OpenAI API.
    """
    return Ghostbox(backend=LLMBackend.openai, **kwargs)


class Ghostbox:
    def __init__(self, endpoint: str, backend: LLMBackend, **kwargs):
        kwargs["endpoint"] = endpoint
        kwargs["backend"] = backend.name

        self.__dict__ |= kwargs
        default_options, tags = makeDefaultOptions()
        self.__dict__["_plumbing"] = Plumbing(
            options=default_options.__dict__
            | {
                k: v
                for k, v in self.__dict__.items()
                if not (k.startswith("_")) or k in kwargs.keys()
            },
            tags=tags,
        )

        # override with some API defaults
        # FIXME: only if not specified by user
        self.__dict__["_plumbing"].options |= definitions.api_default_options

        if self.config_file:
            self.load_config(self.config_file)
        setup_plumbing(self._plumbing)


        # for arcane reasons we must startthe tts after everything else
        if self._plumbing.tts_flag:
            self._plumbing.tts_flag = False
            self._plumbing.options["tts"] = False
            printerr(toggle_tts(self._plumbing))


    @contextmanager
    def options(self, **kwargs):
        # copy old values
        tmp = {k: v for (k, v) in self._plumbing.options.items() if k in kwargs}
        # this has to be done one by one as setoptions has sideffects
        for new_k, new_v in kwargs.items():
            self._plumbing.setOption(new_k, new_v)
        yield self
        # now unwind, also one by one
        for old_k, old_v in tmp.items():
            self._plumbing.setOption(old_k, old_v)

    @contextmanager
    def option(self, name, value):
        with self.options({name: value}):
            yield self

    def set(option_name: str, value) -> None:
        if option_name in self.__dict__:
            self.__dict__[option_name] = value
        self._plumbing.setOption(option_name, value)

    def get(self, option_name: str) -> object:
        return self._plumbing.getOption(option_name)

    def set_vars(self, injections: Dict[str, str]) -> Self:
        for k, v in injections.items():
            self._plumbing.session.setVar(k, v)
        return self

    def __getattr__(self, k):
        return self.__dict__["_plumbing"].getOption(k)

    def __setattr__(self, k, v):
        if k == "_plumbing":
            return

        if k in self.__dict__:
            self.__dict__[k] = v

        if "_plumbing" in self.__dict__:
            self.__dict__["_plumbing"].setOption(k, v)

    # diagnostics
    def status(self):
        pass

    def is_busy(self) -> bool:
        """Returns true if an interaction with a backend is currently in progress.
        While busy, all changes to the state of options (e.g. via set or the options context manager) will be buffered and applied when the ghostbox is no longer busy."""
        return self._plumbing._frozen

    # these are the payload functions
    def text(self, prompt_text: str, timeout=None) -> str:
        with self.options(**{"stream": False}):
            return self._plumbing.interactBlocking(prompt_text, timeout=timeout)

    def text_async(self, prompt_text: str, callback: Callable[[str], None]) -> None:
        with self.options(**{"stream": False}):
            # FIXME: this is tricky as we immediately return and set stream = True again ??? what to do
            self._plumbing.interact(prompt_text, user_generation_callback=callback)
        return

    def text_stream(
        self,
        prompt_text: str,
        chunk_callback: Callable[[str], None],
        generation_callback: Callable[[str], None] = lambda x: None,
    ) -> None:
        with self.options(**{"stream": True}):
            self._plumbing.interact(
                prompt_text,
                user_generation_callback=generation_callback,
                stream_callback=chunk_callback,
            )
        return

    def json(self, prompt_text: str) -> dict:
        with self.options(response_format={"type":"json_object"}):
            return self.text(prompt_text)

    def json_async(self, prompt_text: str, callback: Callable[[dict], None]) -> None:
        pass

    def chat(self, user_message: ChatMessage) -> ChatResult:
        pass

    def chat_async(
        self, user_message: ChatMessage, callback: Callable[[dict], ChatResult]
    ) -> None:
        pass

    def completion(self, prompt_text: str) -> CompletionResult:
        pass

    def completion_async(
        self, prompt_text: str, callback: Callable[[CompletionResult], None]
    ) -> None:
        pass

    def start_session(self, filepath: str, keep=False) -> Self:
        printerr(start_session(self._plumbing, filepath))
        return self

    def load_config(self, config_file: str) -> Self:
        printerr(load_config(self._plumbing, config_file))
        # FIXME: update self.__dict__?
        return self

    def tools_inject_dependency(self, symbol_name: str, obj: object) -> Self:
        """Make a python object available in the python tool module of a running ghostbox AI, without having defined it in the tools.py.
        This can be used to inject dependencies from the 'outside'. This is useful in cases where you must make an object available to the AI that can not be acquired during initialization of the tool module, for example, a resource manager, or a network connection.
        Note that the AI will not be aware of an injected dependency, and be unable to reference it. However, you can refer to the symbol_name in the functions you define for the AI, which will be a bound reference as soon as you inject the dependency. The AI may then use the functions that previously didn't refer to the object now bound by symbol_name.
        :param symbol_name: The name of the identifier to be injected. If this is e.g. 'point', point will be bound to obj in tools.py.
        :param obj: An arbitrary python object. A reference to object will be bound to symbol_name and be available in tools.py.
        :return: Ghostbox"""
        module = self._plumbing.session.tools_module
        if module is None:
            printerr(
                "warning: Unable to inject dependency '"
                + symbol_name
                + "'. Tool module not initialized."
            )
            return Self

        # FIXME: should we warn users if they overriade an existing identifier? Let's do ti since if they injected once why do they need to do it again?
        if symbol_name in module.__dict__:
            printerr(
                "warning: While trying to inject a dependency: '"
                + symbol_name
                + "' already exists in tool module."
            )

        module.__dict__[symbol_name] = obj

    def tts_say(self, text: str, interrupt: bool = False) -> Self:
        self._plumbing.communicateTTS(text, interrupt=interrupt)
        return self

    def tts_stop(self) -> Self:
        self._plumbing.stopTTS()
        return self

    def set_char(
        self,
        character_folder: str,
        chat_history: Optional[List[ChatMessage | Dict[str, Any]]] = None,
    ) -> Self:
        """Set an active character_folder, which may be the same one, and optionally set the chat history.
        Note: This will wipe the previous history unless chat_history is None.
        :param character_folder: The new character folder to load.
        :param chat_history: A list of ChatHistory items, valid JSON dictionaries that parse as ChatHistoryItems, or a mix of both. If None, chat history will be retained.
        :return: Ghostbox instance."""
        if character_folder != self._plumbing.getOption("character_folder"):
            printerr(start_session(self._plumbing, character_folder))

        if chat_history is None:
            return self

        self._plumbing.session.stories.reset()
        story = self._plumbing.session.stories.get()
        for item in chat_history:
            if type(item) == ChatMessage:
                story.appendMessage(item)
            else:
                # try to parse the item as ChatMessage
                try:
                    story.addRawJSON(item)
                except:
                    printerr(
                        "warning: Couldn't parse chat history. Not a valid ChatMessage. Skipping message. Traceback below."
                    )
                    printerr(traceback.format_exc())
                    continue

        return self

        return self
