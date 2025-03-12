from __future__ import annotations
import traceback
from pydantic import BaseModel, ValidationError
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
    def is_busy(self) -> bool:
        """Returns true if an interaction with a backend is currently in progress.
        While busy, all changes to the state of options (e.g. via set or the options context manager) will be buffered and applied when the ghostbox is no longer busy.
        """
        return self._plumbing._frozen

    # these are the payload functions
    def text(
        self,
        prompt_text: str,
        timeout: Optional[float] = None,
        options: Dict[str, Any] = {},
    ) -> str:
        """Generate text based on a prompt.
        This function blocks until either the generation finishes, or a provided timeout is reached.
        :param prompt_text: The prompt given to the LLM backend.
        :param timeout: Number of seconds to wait before the generation is canceled.
        :param options: Additional options to pass to ghostbox and possibly the backend, e.g. `{"min_p": 0.01}`. This is an alternative to the `with options` context manager.
        :return: A string that was generated by the backend, based on the provided prompt.
        """
        with self.options(stream=False, **options):
            return self._plumbing.interactBlocking(prompt_text, timeout=timeout)

    def text_async(
        self,
        prompt_text: str,
        callback: Callable[[str], None],
        options: Dict[str, Any] = {},
    ) -> None:
        """Generate text based on a prompt asynchronously.
        This function does not block. It returns immediately and calls the provided callback when the generation finishes.
        :param prompt_text: The prompt given to the LLM backend.
        :param callback: A function that accepts a string. This will be called when the generation finishes, with the generated string as the only parameter.
        :param options: Additional options to pass to ghostbox and possibly the backend, e.g. `{"min_p": 0.01}`. This is an alternative to the `with options` context manager.
        :return: The ghostbox instance.
        """
        with self.options(stream=False, **options):
            # FIXME: this is tricky as we immediately return and set stream = True again ??? what to do
            self._plumbing.interact(prompt_text, user_generation_callback=callback)
        return

    def text_stream(
        self,
        prompt_text: str,
        chunk_callback: Callable[[str], None],
        generation_callback: Callable[[str], None] = lambda x: None,
        options: Dict[str, Any] = {},
    ) -> None:
        """Generate text based on a prompt and stream the response.
        This function does not block. It returns immediately and invokes the provided callbacks when their respective events procure.
        :param prompt_text: The prompt given to the LLM backend.
        :param chunk_callback: This is called for each token that the backend generates, with the token as sole parameter.
        :param generation_callback: This is called exactly once, upon completion of the response, with the entire generation as sole parameter. If you are thinking of concatenating all the tokens in chunk_callback, use this instead.
        :param options: Additional options to pass to ghostbox and possibly the backend, e.g. `{"min_p": 0.01}`. This is an alternative to the `with options` context manager.
        :return: The ghostbox instance.
        """
        with self.options(stream=True, **options):
            self._plumbing.interact(
                prompt_text,
                user_generation_callback=generation_callback,
                stream_callback=chunk_callback,
            )
        return

    @staticmethod
    def _make_json_schema(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if schema is None:
            return {"type": "json_object"}
        return {"type": "json_object", "schema": schema}

    def json(
        self,
        prompt_text: str,
        schema: Optional[Dict] = None,
        timeout: Optional[float] = None,
        options: Dict[str, Any] = {},
    ) -> str:
        """Given a prompt, returns structured output as a string that is json deserializable.
        Output is structured but somewhat unpredictable, unless you provide a json schema. If you are thinking about using pydantic objects and using their model_json_schema method, consider using the ghostbox.new method directly.
        :param prompt_text: The prompt text as natural language.
        :param schema: A dict representing a json schema, which will further restrict the generation.
        :param timeout: Number of seconds to wait before generation is canceled.
        :param options: Additional options that will be passed to the backend, e.g. `{"min_p": 0.01}`.
        :return: A string that contains json, hopefully adequately satisfying the prompt.
        Example use:
        noises = json.loads(box.json("Can you list some animal noises? Please give key/value pairs."))
        noises is now e.g. {"dog": "woof", "cat":"meow", ...}
        """
        with self.options(response_format=self._make_json_schema(schema), **options):
            return self.text(prompt_text, timeout=timeout)

    def json_async(
        self,
        prompt_text: str,
        callback: Callable[[dict], None],
        schema: Optional[Dict[str, Any]] = None,
        options: Dict[str, Any] = {},
    ) -> None:
        """This is an asyncrhonous version of the json method. See its documentation for more. This method returns nothing but expects an additional callback parameter, which will be called with the generated json string as argument once the backend is done generating.
        This function does not block, but returns immediately."""
        with self.options(response_format=self._make_json_schema(schema), **options):
            self.text_async(prompt_text, callback=callback)
        return

    def new(
        self,
        pydantic_class,
        text_prompt: str,
        timeout: Optional[float] = None,
        retries: int = 0,
        options: Dict[str, Any] = {},
    ) -> Any:
        """Given a subclass of pydantic.BaseModel, returns a python object of that type, with its fields filled in by the LLM backend with adherence to a given text prompt.
        This function will block until either generation finishes or a provided timeout is reached.
        This function may raise a pydantic error if the object cannot be validated. Although the LLM will be forced to adhere to the pydantic data model, this can still happen occasionally, for example, in the case of refusals. Either use the retries argument, or wrap a call to new in a try block accordingly.
        :param pydantic_class: The type of object that should be created.
        :param text_prompt: Prompt given to the LLM that should aid in object creation. This will influence how the pydantic object's fields will be filled in. Depending on the model used, certain prompts may lead to refusal, even with object creation, so be careful.
        :param timeout: A timeout in seconds after which generation is canceled.
        :param retries: In case of a validation error, how often should the generation be retried? Since sampling, especially with temperature > 0.0, is not deterministic, retries can eventually yield valid results. A value of 0 means no retries will be performed and the function raises an error on invalid data or refusal. A value of -1 means retry forever or until the timeout is reached.
        :param options: Additional options to pass to ghostbox and possibly the backend, e.g. `{"min_p": 0.01}`. This is an alternative to the `with options` context manager.
        :return: A valid python object of the provided pydantic type, with its fields filled in.

        Example:

        ## animal.py
        ```python
        from pydantic import BaseModel
        import ghostbox, json

        class Animal(BaseModel):
            name: str
            cute_name: str
            number_of_legs: int
            friendly: bool
            favorite_foods: List[str]

        box = ghostbox.from_generic(...)
        cat = box.new(Animal, "Please generate a cute housecat.")
        print(json.dumps(cat.model_dump(), indent=4))
        ```

        ## Output

        ```bash
        $ python animal.py
        {
            "name" : "Cat",
            "cute_name" : "Dr. Kisses",
            "number_of_legs" : 4,
            "friendly" : true,
            "favorite_foods" : ["Tuna", "Cat Treats", "Water from the toilet"]
        }
        ```
        """
        with self.options(stream=False, **options):
            while True:
                try:
                    return pydantic_class(
                        **json.loads(
                            self.json(
                                text_prompt,
                                schema=pydantic_class.model_json_schema(),
                                timeout=timeout,
                            )
                        )
                    )
                except ValidationError as e:
                    if retries == 0:
                        raise e
                    retries -= 1

    # managing ghostbox operation
    def start_session(self, filepath: str, keep=False) -> Self:
        """Start a completely new session with a given character folder.
        This function wipes all history and context variables. It's a clean slate. If you want to switch characters while retaining context, use set_char instead.
        :param filepath: Path to a character folder.
        :return: The ghostbox instance.
        """
        printerr(start_session(self._plumbing, filepath))
        return self

    def load_config(self, config_file: str) -> Self:
        """Loads a config file and applies the option/value pairs in it to this ghostbox instance.
        A config file contains a valid json string that describes an options dictionary.
        Configs loaded this way are intended to be user profiles or similar. AI character folders have their own config.json files that, by convention, are loaded automatically. If you want to simply configure an AI char, you should use their dedicated config file as it generally does *the right thing* when it comes to annoying stuff of overriding options and option precedence etc.
        Example:

        ## alice_config.json

        ```json
        {
            "chat_user" : "alice",
            "tts": true,
            "tts_model": "kokoro",
            "tts_voice": "af_sky",
            "log_time": false
        }
        ```
        """
        printerr(load_config(self._plumbing, config_file))
        # FIXME: update self.__dict__?
        return self

    def tools_inject_dependency(self, symbol_name: str, obj: object) -> Self:
        """Make a python object available in the python tool module of a running ghostbox AI, without having defined it in the tools.py.
        This can be used to inject dependencies from the 'outside'. This is useful in cases where you must make an object available to the AI that can not be acquired during initialization of the tool module, for example, a resource manager, or a network connection.
        Note that the AI will not be aware of an injected dependency, and be unable to reference it. However, you can refer to the symbol_name in the functions you define for the AI, which will be a bound reference as soon as you inject the dependency. The AI may then use the functions that previously didn't refer to the object now bound by symbol_name.
        :param symbol_name: The name of the identifier to be injected. If this is e.g. 'point', point will be bound to obj in tools.py.
        :param obj: An arbitrary python object. A reference to object will be bound to symbol_name and be available in tools.py.
        :return: Ghostbox
        Here's an example use of dependency injection:
        ## human_resources.py
        ```python
        box = ghostbox.from_generic(...)
        # imaginary database connection e.g.
        # we don't want this in the tools.py of employee_assistant
        database_handle = EmployeeDatabase.open_connection(user="bob", password="bruce_schneier")
        # stuff happens, maybe we start a read/evaluate loop
        # ...

        # start our assistant
        box.start_session("employee_assistant")
        #without this, the tool calls would fail
        box.tools_inject_dependency("DBHANDLE", database_handle)
        with options{use_tools=True):
            good_employees = box.text("Can you give me a quick summary of all the employees in the database that have overperformed last motnh?")
        # do something with good_employees
        # ...
        ```

        ## employee_assistant/tools.py
        ```python
        # many tool definitions here
        # ...

        # then the one the AI will probably choose for our request above
        def query_database(sql_query: str) -> Dict:
            # this would fail without dependency injection
            # since DBHANDLE is not defined in the module
            results = DBHANDLE.query(sql)
            # do stuff with results to bring it into a form the AI likes
            final = some_repackaging(results)
            return final
        ```

        At the point above where tools_inject_dependency is called, the DBHANDLE identified in the tools.py module becoems defined. Without the injection, referencing it would raise an exception.
        """
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

    def tts_say(self, text: str, interrupt: bool = True) -> Self:
        """Speak something with the underlying TTS engine.
        You can modify the tts behaviour by wrapping this call in a `with options(...)` context. For possible options, see the ghostbox documentation of the various tts_* options.
        :param text: The text to be spoken.
        :param interrupt: If true, this call will interrupt speech output that is currently in progress (the default). Otherwise, text will be queue and spoken in sequence.
        :return: The ghostbox instance.
        """
        self._plumbing.communicateTTS(text, interrupt=interrupt)
        return self

    def tts_stop(self) -> Self:
        """Stops speech output that is in progress.
        :return: The ghostbox instance.
        """
        self._plumbing.stopTTS()
        return self

    def set_char(
        self,
        character_folder: str,
        chat_history: Optional[List[ChatMessage | Dict[str, Any]]] = None,
    ) -> Self:
        """Set an active character_folder, which may be the same one, and optionally set the chat history.
        This method differs from start_session in that it doesn't wipe various vars that may be set in the session, and preserves the chat history by default.
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

    def history(self) -> List[ChatMessage]:
        """Returns the current chat history for this ghostbox instance.
        :return: The chat history.
        """
        return self._plumbing.session.stories.get().getData()
