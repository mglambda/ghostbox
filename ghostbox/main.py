import requests, json, os, io, re, base64, random, sys, threading, signal, tempfile, string, uuid, textwrap
from queue import Queue, Empty
from typing import *
import feedwater
from functools import *
from colorama import just_fix_windows_console, Fore, Back, Style
from huggingface_hub import snapshot_download, try_to_load_from_cache
from lazy_object_proxy import Proxy  # type: ignore
import argparse
from argparse import Namespace
from ghostbox.commands import *
from ghostbox.autoimage import *
from ghostbox.output_formatter import *
from ghostbox.util import *
from ghostbox import util
from ghostbox import agency
from ghostbox._argparse import *
from ghostbox import streaming
from ghostbox.session import Session
from ghostbox.pftemplate import *
from ghostbox.backends import *
from ghostbox import backends
from ghostbox.client import GhostboxClient
import ghostbox


def showHelpCommands(prog, argv):
    """
    List commands, their arguments, and a short description."""

    w = ""
    candidate = "/" + "".join(argv).strip()
    failure = True
    for cmd_name, f in sorted(cmds, key=lambda p: p[0]):
        if candidate != "/" and cmd_name != candidate:
            continue
        failure = False
        if f.__doc__ is None:
            docstring = cmds_additional_docs.get(cmd_name, "")
        else:
            docstring = str(f.__doc__)
        w += cmd_name + " " + docstring + "\n"
    printerr(w, prefix="")

    if failure:
        return "Command not found. See /help for more."
    return ""


def show_help_option_tag(prog, tag, markdown: bool = False) -> str:
    w = ""
    if not (markdown):
        cv = str(prog.getOption(tag.name))
        w += (
            tag.name
            + "\n"
            + "\n".join(textwrap.wrap(tag.help))
            + "\n"
            + "\n".join(textwrap.wrap(tag.show_description()))
            + "\nIts current value is "
            + (cv if cv else '""')
            + "."
        )
        if tag.default_value is not None:
            dv = str(tag.default_value)
            w += " Its default value is " + (dv if dv else '""')
        if tag.service:
            w += "\nSetting it to True will start the corresponding service."
    else:
        # we generate markdown for the docs
        w += (
            f"## `{tag.name}`"
            + "\n"
            + tag.help
            + "\n\n```\n"
            + "\n".join(textwrap.wrap(tag.show_description()))
        )
        if tag.default_value is not None:
            dv = str(tag.default_value)
            w += "\nIts default value is " + (dv if dv else '""')
        if tag.service:
            w += "\nSetting it to True will start the corresponding service."  # this is for the docs
        w += "\n```\n"
        # avoid printerr prefix
        print(w)
        return ""
    return w


def showHelp(prog, argv):
    """[TOPIC] [-v|--verbose] [--markdown]
    List help on various topics. Use -v to see even more information."""
    w = ""
    if argv != [] and argv[-1] == "--markdown":
        markdown = True
        del argv[-1]
    else:
        markdown = False

    if argv == []:
        # list topics
        w += "[Commands]\nUse these with a preceding slash like e.g. '/retry'. Do /help COMMANDNAME for more information on an individual command.\n\n"
        command_names = sorted([tuple[0].replace("/", "") for tuple in cmds])
        for i in range(len(command_names)):
            command_name = command_names[i]
            if i % 3 == 0:
                start = "\n  "
            else:
                start = "\t"

            w += start + command_name

        w += "\n\n[Options]\nSet these with '/set OPTIONNAME VALUE', where value is a valid python expression.\n List options and their values with /ls [OPTIONNAME].\n"
        tags = prog.getTags()
        options = sorted(
            tags.items(), key=lambda item: (item[1].group.name, item[1].name)
        )
        last_group = ""
        for i in range(len(options)):
            option, tag = list(options)[i]
            if last_group != tag.group.name:
                w += "\n\n  [" + tag.group.name + "]\n"
                last_group = tag.group.name
            option = (
                wrapColorStyle(option, "", Style.BRIGHT)
                if tag.very_important
                else option
            )
            if i % 3 == 0:
                w += "\n" + option
            else:
                w += "\t" + option

        return w

    topic = argv[0]
    if topic in prog.getTags():
        tag = prog.getTags()[topic]
        if tag.is_option:
            return show_help_option_tag(prog, tag, markdown=markdown)
    elif topic == "commands":
        # list all commands with help
        return showHelpCommands(prog, [])
    elif topic == "options":
        # list full options with description
        return "\n\n".join(
            filter(
                lambda w: w != "",
                [
                    show_help_option_tag(prog, tag, markdown=markdown)
                    for tag in sorted(prog.getTags().values(), key=lambda t: t.name)
                    if tag.is_option
                ],
            )
        )
    else:
        # fix it's a command
        return showHelpCommands(prog, [topic])
    # else:
    # not found
    # return "Topic not found in help. Please choose one of the topics below.\n" + showHelp(prog, [])
    return w


# these can be typed in at the CLI prompt
cmds = [
    ("/help", showHelp),
    ("/helpcommands", showHelpCommands),
    ("/start", newSession),
    ("/switch", switch),
    ("/quit", exitProgram),
    ("/test", testQuestion),
    ("/restart", lambda prog, argv: newSession(prog, [])),
    ("/print", printStory),
    ("/next", nextStory),
    ("/prev", previousStory),
    ("/story", gotoStory),
    ("/retry", retry),
    ("/rephrase", rephrase),
    ("/drop", dropEntry),
    ("/new", newStory),
    ("/clone", cloneStory),
    ("/log", lambda prog, w: printStory(prog, w, stderr=True, apply_filter=False)),
    ("!", transcribe),
    ("/transcribe", transcribe),
    ("/audio", toggleAudio),
    ("/image_watch", toggleImageWatch),
    ("/image", image),
    ("/time", showTime),
    ("/status", showStatus),
    ("/detokenize", detokenize),
    ("/tokenize", tokenize),
    ("/raw", showRaw),
    ("/lastresult", debuglast),
    ("/lastrequest", lastRequest),
    ("/ttsdebug", ttsDebug),
    ("/tts", toggleTTS),
    ("/set", setOption),
    ("/unset", lambda prog, argv: setOption(prog, [argv[0]])),
    ("/template", changeTemplate),
    ("/saveoptions", saveConfig),
    ("/saveconfig", saveConfig),
    ("/loadconfig", loadConfig),
    ("/save", saveStoryFolder),
    ("/load", loadStoryFolder),
    ("/varfile", varfile),
    ("/lstemplates", showTemplates),
    ("/lsoptions", showOptions),
    ("/lschars", showChars),
    ("/lsvoices", showVoices),
    ("/lsvars", showVars),
    ("/mode", toggleMode),
    ("/hide", hide),
    ("/cont", doContinue),
    ("/continue", doContinue),
]

# the mode formatters dictionary is a mapping from mode names to tuples of formatters, wrapped in a lambda that supplies a dictionary with keyword options to the formatters.
# The tuple contains indices :
# 0 - Formats text sent to the console display
# 1 - Formats text sent to the TTS backend
# 2 - Formats text to be saved as user message in the chat history
# 3 - formats text to be saved as AI message in the chat history
mode_formatters = {
    "default": lambda d: (DoNothing, DoNothing, DoNothing, CleanResponse),
    "chat": lambda d: (
        DoNothing,
        NicknameRemover(d["chat_ai"]),
        NicknameFormatter(d["chat_user"]),
        ChatFormatter(d["chat_ai"]),
    ),
}


class Plumbing(object):
    def __init__(self, options={}, initial_cli_prompt="", tags={}):
        # the printerr stuff needs to happen early
        self._printerr_buffer = []
        self._initial_printerr_callback = lambda w: self._printerr_buffer.append(w)
        util.printerr_callback = self._initial_printerr_callback
        if not (options["stderr"]):
            util.printerr_disabled = True
        # some general callbacks that are veryuseful
        self._on_interaction = None  # Callable[[], None]
        self._on_interaction_finished = None  # Callable[[], None]

        # this is for the websock clients
        self.stderr_token = "[|STDER|]:"
        self._stdout_ringbuffer = ""
        self._stdout_ringbuffer_size = 1024
        self._frozen = False
        self._freeze_queue = Queue()
        self.options = options
        self.tags = tags
        self.backend = None
        self.template = None
        self.initializeBackend(self.getOption("backend"), self.getOption("endpoint"))
        self.session = Session(chat_user=options.get("chat_user", ""))
        # FIXME: make this a function returning backend.getlAStResult()
        self.lastResult = {}
        self._lastInteraction = 0
        self.tts_flag = False
        self.initial_print_flag = False
        self.initial_cli_prompt = initial_cli_prompt
        self.stream_queue = []
        self.stream_sentence_queue = []
        self._stream_only_once_token_bag = set()
        self.images = {}
        # flag to show wether image data needs to be resent to the backend
        self._images_dirty = False
        self._lastPrompt = ""
        self._dirtyContextLlama = False
        self._stop_generation = threading.Event()
        # indicates generation work being done, really only used to print the goddamn cli prompt
        self._busy = threading.Event()
        self._busy.set()
        self._triggered = False
        self._smartShifted = False
        self._systemTokenCount = None
        self.continue_with = ""
        self.tts = None
        self.tts_config = None
        self.multiline_buffer = ""
        if self.getOption("json_grammar"):
            self.setOption("grammar", getJSONGrammar())
            del self.options["json"]
        elif self.getOption("grammar_file"):
            self.loadGrammar(self.getOption("grammar_file"))
        else:
            self.setOption("grammar", "")
        # template
        self.loadTemplate(self.getOption("prompt_format"), startup=True)

        # whisper stuff. We do this with a special init function because it's lazy
        self._on_transcription = None  # Callable[[str],str] or None
        self._on_activation = None  # Callable[[], None] or None
        self._on_transcription_generation = None  # Callable[[str], None]
        self.whisper = self._newTranscriber()
        self.ct = None
        self._defaultSIGINTHandler = signal.getsignal(signal.SIGINT)
        self._transcription_suspended = False

        # imagewatching
        self.image_watch = None

        # http server
        self.http_server = None
        self.http_thread = None

        # websock server
        self.websock_thread = None
        self.websock_server = None
        self.websock_clients = []
        self.websock_server_running = threading.Event()
        self.websock_msg_queue = Queue()

        # formatters is to be idnexed with modes
        self.setMode(self.getOption("mode"))
        self.running = True
        self._startCLIPrinter()

    def getTags(self):
        return self.tags | {
            param.name: param
            for param in backends.sampling_parameter_tags.values()
            if param.name in self.getBackend().sampling_parameters().keys()
        }

    def initializeBackend(self, backend, endpoint):
        api_key = self.getOption("api_key")
        kwargs = {"logger": self.verbose}
        if backend == LLMBackend.llamacpp.name:
            self.backend = LlamaCPPBackend(endpoint, **kwargs)
        elif backend == LLMBackend.openai.name:
            if not api_key:
                printerr(
                    "error: OpenAI API key is required for the OpenAI backend. Did you forget to provide --api_key?"
                )
                # this is rough but we are in init phase so it's ok
                sys.exit()
            self.backend = OpenAIBackend(api_key, **kwargs)
            self.setOption("prompt_format", "auto")
        elif backend == LLMBackend.generic.name:
            self.backend = OpenAIBackend(api_key, endpoint=endpoint, **kwargs)
            self.setOption("prompt_format", "auto")
        elif backend == LLMBackend.legacy.name:
            if (
                self.getOption("prompt_format")
                == PromptFormatTemplateSpecialValue.auto.name
            ):
                printerr(
                    "warning: Setting prompt_format to 'raw' as using 'auto' as prompt_format with the legacy backend will yield server errors. This backend exists specifically to *not* apply templates server side. You can manually reset this if you want."
                )
                # doing it without setOption as backend isn't fully initialized yet
                self.options["prompt_format"] = (
                    PromptFormatTemplateSpecialValue.raw.name
                )

            self.backend = OpenAILegacyBackend(api_key, endpoint=endpoint, **kwargs)
        else:
            # Handle other backends...
            pass

        # fill in the default sampler parameters that were not shown in the command line arguments, but are still supported
        for param in self.backend.sampling_parameters().values():
            if param.name not in self.options.keys():
                self.setOption(param.name, param.default_value)

    def getBackend(self):
        return self.backend

    def justUsedTools(self) -> bool:
        # FIXME: not sure if this is the place to do it/right way to do it
        # if the last message in the history has role 'tools', it means
        # we just send results, so we don't use tools for this interaction
        try:
            return self.session.stories.get().getData()[-1].role == "tool"
        except:
            printerr("warning: Tried to check for tools use with empty history.")
        return False

    def _getTTSSpecialMsg(self) -> str:
        """Returns special tags for the TTS or empty string."""
        w = ""
        tags = self.tryGetTTSSpecialTags()
        if tags != []:
            w += (
                "\nUse the following tags in conversation when appropriate (they will be translated into audio by a TTS model): "
                + ", ".join(tags)
            )

        return w

    def makeGeneratePayload(self, text):
        d = {}

        # gather set sampling parameters and warn user for unsupported ones
        all_params = backends.sampling_parameters.keys()
        supported_params = self.getBackend().sampling_parameters().keys()
        for param in all_params:
            if param in self.options:
                if param not in supported_params and not (
                    self.getOption("force_params")
                ):
                    if self.getOption("warn_unsupported_sampling_parameter"):
                        printerr(
                            "warning: Sampling parameter '"
                            + param
                            + "' is not supported by the "
                            + str(self.getBackend().getName())
                            + " backend. Set force_params to true to send it anyway."
                        )
                    else:
                        continue
                else:
                    d[param] = self.getOption(param)

        # some special ones
        for k, v in special_parameters.items():
            d[k] = v if not (self.getOption(k)) else self.getOption(k)

        # don't override grammar if it's null, rather not send it at all
        if "grammar" in d:
            if not (d["grammar"]):
                del d["grammar"]

        # just throw toools in for backends that can use them, unless user disabled
        if self.getOption("use_tools"):
            if not (self.justUsedTools()):
                d["tools"] = [tool.model_dump() for tool in self.session.tools]
                # FIXME: necessary currently with llamacpp
                if "grammar" in d:
                    del d["grammar"]

        if (
            self.getOption("backend") == LLMBackend.generic.name
            or self.getOption("backend") == LLMBackend.openai.name
            or self.getOption("prompt_format")
            == PromptFormatTemplateSpecialValue.auto.name
        ):
            # openai chat/completion needs the 'messages' key
            # we also do this for llama in "auto" template mode
            d["system"] = self.session.getSystem()
            if self.getOption("tts_modify_system_msg"):
                d["system"] += self._getTTSSpecialMsg()

            d["messages"] = [
                {"role": "system", "content": d["system"]}
            ] + copy.deepcopy(self.session.stories.get().to_json())
            # FIXME: I don't even know if this does something right now
            self.images_dirty = False

        else:
            # all others get the text
            # FIXME: there is some deep mishandling here because this uses the text parameter while oai endpoitns use the story. must investigate this
            d["prompt"] = text
        return d

    def isContinue(self):
        return self.getOption("continue")

    def resetContinue(self):
        self.setOption("continue", False)

    def _newTranscriber(self):
        # makes a lazy WhisperTranscriber, because model loading can be slow
        # yes this is hacky but some platforms (renpy) can't handle torch, which the whisper models rely on
        def delayWhisper():
            from ghostbox.transcribe import WhisperTranscriber

            return WhisperTranscriber(
                model_name=self.getOption("whisper_model"),
                silence_threshold=self.getOption("audio_silence_threshold"),
                input_func=lambda: printerr("Started Recording. Hit enter to stop."),
            )

        return Proxy(delayWhisper)

    def continueWith(self, newUserInput):
        # FIXME: the entire 'continue' architecture is a trashfire. This should be refactored along with other modeswitching/input rewriting stuff in the main loop
        self.setOption("continue", "1")
        self.continue_with = newUserInput

    def popContinueString(self):
        self.setOption("continue", False)
        tmp = self.continue_with
        self.continue_with = ""
        return tmp

    def loadGrammar(self, grammar_file):
        if os.path.isfile(grammar_file):
            w = open(grammar_file, "r").read()
            self.setOption("grammar", w)
        else:
            self.setOption("grammar", "")
            printerr(
                "warning: grammar file "
                + grammar_file
                + " could not be loaded: file not found."
            )

    def guessPromptFormat(self):
        """Uses any trick it can do guess the prompt format template. Returns a string like 'chat-ml', 'alpaca', etc."""
        # see if we can find llm_layers
        try:
            data = loadLayersFile()
        except:
            printerr("warning: Couldn't load layers file " + getLayersFile())

        try:
            models = dirtyGetJSON(self.getOption("endpoint") + "/v1/models").get(
                "data", []
            )
            # hope it's just one
            model = os.path.basename(models[0]["id"])
        except:
            printerr(traceback.format_exc())
            printerr("Failed to guess prompt format. Defaulting.")
            return "raw"

        # check if model is in layers file
        for d in data:
            if "name" not in d.keys():
                continue
            if d["name"].lower() == model.lower():
                if d["prompt_format"]:
                    # success
                    return d["prompt_format"]

        # FIXME: at this point it's not in the layers file, but we still have a model name. consider googling it on hugginface and grepping the html
        printerr(
            "Failed to guess prompt format after exhausting all options 😦. Defaulting."
        )
        return "raw"

    def loadTemplate(self, name, startup=False):
        # this is so llamacpp uses the /chat/completions endpoint if we use auto templating
        # has no effect on other backends
        self.getBackend().configure(
            {
                "llamacpp_use_chat_completion_endpoint": (
                    True
                    if name == PromptFormatTemplateSpecialValue.auto.name
                    else False
                )
            }
        )

        # special cases
        if name == PromptFormatTemplateSpecialValue.auto.name:
            if startup and (self.template is not None):
                # don't set this twice
                return
            printerr(
                "Prompt format template set to 'auto': Formatting is handled server side."
            )
            self.template = RawTemplate()
            return
        if name == "guess":
            name = self.guessPromptFormat()

        allpaths = [p + "/" + name for p in self.getOption("template_include")]
        for path in allpaths:
            path = os.path.normpath(path)
            if not (os.path.isdir(path)):
                failure = (
                    "Could not find prompt format template '"
                    + name
                    + "'. Did you supply a --template_include option?"
                )
                continue
            failure = False
            try:
                template = FilePFTemplate(path)
                break
            except FileNotFoundError as e:
                failure = e

        if failure:
            printerr("warning: " + str(failure) + "\nDefaulting to 'raw' template.")
            self.template = RawTemplate()
            self.options["prompt_format"] = "raw"
            return
        # actually load template
        # first unload old stops
        self.options["stop"] = list(
            filter(lambda w: w not in self.template.stops(), self.options["stop"])
        )

        self.template = template
        for w in template.stops():
            if not (w):
                continue
            self.appendOption("stop", w)
        self.options["prompt_format"] = name
        printerr("Using '" + name + "' as prompt format template.")

    def loadConfig(self, json_data, override=True, protected_keys: List[str] = []):
        """Loads a config file provided as json into options. Override=False means that command line options that have been provided will not be overriden by the config file."""
        d = json.loads(json_data)
        if type(d) != type({}):
            return "error loading config: Not a dictionary."
        if not (override):
            # drop keys in the config that can be found in the command line arguments
            # have to do the one letter options manually
            letterMap = {
                "u": "chat_user",
                "c": "character_folder",
                "V": "tts_voice",
                "T": "prompt_format",
            }
            for arg in sys.argv:
                if not (arg.startswith("-")):
                    continue
                key = stripLeadingHyphens(arg)
                # "no-" is for e.g. '--no-tts'
                if key in d or "no-" + key in d:
                    del d[key]
                for letter, full_arg in letterMap.items():
                    if key.startswith(letter):
                        if full_arg in d:
                            del d[full_arg]

        # now actually load the options, with a partial ordering
        items = sorted(
            d.items(), key=cmp_to_key(lambda a, b: -1 if a[0] == "mode" else 1)
        )
        for k, v in items:
            if not (k in protected_keys):
                self.setOption(k, v)
        return ""

    def showCLIPrompt(self):
        if self.isMultilineBuffering():
            return ""

        if self.getOption("cli_prompt") == "":
            return ""

        f = IdentityFormatter()
        if self.getOption("color"):
            f = ColorFormatter(self.getOption("cli_prompt_color")) + f

        return self.session.expandVars(f.format(self.getOption("cli_prompt")))

    def getMode(self):
        w = self.getOption("mode")
        if not (self.isValidMode(w)):
            return "default"
        return w

    def isValidMode(self, mode):
        return mode in mode_formatters

    def setMode(self, mode):
        # FIXME: this isn't affected by freeze, but since most things go through setOption we should be fine, right? ... right?
        if not (self.isValidMode(mode)):
            return

        self.options["mode"] = mode
        if mode.startswith("chat"):
            userPrompt = mkChatPrompt(self.getOption("chat_user"))
            self.setOption("cli_prompt", "\n" + userPrompt)
            self.appendOption("stop", userPrompt, duplicates=False)
            self.appendOption("stop", userPrompt.strip(), duplicates=False)

        else:  # default
            self.options["cli_prompt"] = self.initial_cli_prompt

    def getOption(self, key):
        return self.options.get(key, None)

    def optionDiffers(self, name, newValue):
        if name not in self.options:
            return True
        return self.getOption(name) != newValue

    def getFormatters(self):
        mode = self.getOption("mode")
        if not (self.isValidMode(mode)):
            mode = "default"
            printerr("warning: Unsupported mode '" + mode + "'.. Using 'default'.")
        return mode_formatters[mode](self.options | self.session.getVars())

    def getDisplayFormatter(self):
        return self.getFormatters()[0]

    def getTTSFormatter(self):
        return self.getFormatters()[1]

    def getUserFormatter(self):
        return self.getFormatters()[2]

    def getAIColorFormatter(self):
        if self.getOption("color"):
            return ColorFormatter(
                self.getOption("text_ai_color"), self.getOption("text_ai_style")
            )
        return IdentityFormatter()

    def getAIFormatter(self, with_color=False):
        return self.getAIColorFormatter() + self.getFormatters()[3]

    def addUserText(self, w):
        if w and self.getOption("history"):
            if self.getOption("history_force_alternating_roles"):
                # we don't want 2 consecutive user messages
                # and in this particular case we can just extend the last message.
                w = self._maybe_extend_last_user_message(w)

            self.session.stories.get().addUserText(
                self.getUserFormatter().format(w), image_context=self.images
            )

    def addAIText(self, w):
        """Adds text toe the AI's history in a cleanly formatted way according to the AI formatter. Returns the formatted addition or empty string if nothing was added.."""
        if w == "":
            return ""

        if self.getOption("continue"):
            continuation = self.getAIFormatter().format(w)
            self.session.stories.get().extendAssistantText(continuation)
            return continuation

        # hint may have been sent to the backend but we have to add it to the story ourselves.
        if self.getOption("hint_sticky"):
            hint = self.getOption("hint")
        else:
            hint = ""

        if (rformat := self.getOption("response_format")) and (
            rformat["type"] != "text"
        ):
            # don't use formatters if user expects structured data
            addition = hint + w
        else:
            addition = self.getAIFormatter().format(hint + w)

        if self.getOption("history"):
            self.session.stories.get().addAssistantText(addition)
        return addition

    def addSystemText(self, w):
        """Add a system message to the chat log, i.e. add a message with role=system. This is mostly used for tool/function call results."""
        if w == "":
            return

        # FIXME: we're currently rawdogging system msgs. is this correct?
        self.session.stories.get().addSystemText(w)

    def applyTools(self, w: str = "", json={}) -> Tuple[List[ChatMessage], ChatMessage]:
        """Takes an AI generated string w and optionally the json result from an OpenAI API compatible tool request. Tries to detect a tool request in both. If detected, will apply the tools and then return their results as structured data, as well as the fully parsed original tool call.
        Ideally the JSON result makes tool use obvious through the field
        json["choices"][0]["finish_reason"] == "tool_calls"
        However, not all models will do this consistently and those who can may fail at higher temperatures.
        So as a fallback we check the raw w for json data that corresponds to tools use.
        :param w: Unstructured AI generation string that may contain a tool request.
        :param json: Structured json that was returned by the AI that may contain a tool request.
        :return: A pair of tool results, and the original tool requesting message, both as json dicts. If no tool request was detected, both are empty dicts.
        """
        import json as j

        nothing = ([], {})

        if not (self.getOption("use_tools")):
            self.verbose("Not applying tools because use_tools=False.")
            return nothing

        try:
            if json["choices"][0]["finish_reason"] != "tool_calls":
                self.verbose(
                    f"Not using tools: finish_reason={json["choices"][0]["finish_reason"]}"
                )
                return nothing
            tool_request = json["choices"][0]["message"]
            tool_calls = tool_request["tool_calls"]
        except:
            tool_calls = []
            printerr(traceback.format_exc())
        # FIXME: add heuristic for scanning w for possible tool request
        # ...
        # here, we know it was a tool request and we have tool_calls
        self.verbose(f"Using tools with calls: {tool_calls}")
        if tool_calls == []:
            # either none requested or something went wrong. anyhow.
            return nothing

        tool_results = []
        try:
            for tool_call in tool_calls:
                if self.getOption("verbose"):
                    printerr(j.dumps(tool_call, indent=4))

                tool = tool_call["function"]
                try:
                    params = j.loads(tool["arguments"])
                except:
                    if self.getOption("verbose"):
                        printerr(
                            "warning: Couldn't pass tool arguments as json: "
                            + tool["arguments"]
                        )
                        continue
                maybeResult = self.session.callTool(tool["name"], params)
                tool_results.append(
                    agency.makeToolResult(tool["name"], maybeResult, tool_call["id"])
                )
        except:
            printerr("warning: Caught the following exception while applying tools.")
            printerr(traceback.format_exc())

        if tool_results == []:
            return nothing

        # NOTE: the responses from the server come back more complex than what is later supposed to be sent in the chat history
        # specifically, a message with role "assistant", conten null, and tool_calls=... should *not* have the tool c<calls be a layered dictionary with type and function key, it's just a plain list of dicts of names and arguments
        # I have no idea why this is inconsistent but that's how it is, and that's the reason for the repackaging below.
        tool_request_msg = ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    function=FunctionCall(
                        name=fcall["function"]["name"],
                        arguments=fcall["function"]["arguments"],
                    )
                )
                for fcall in tool_request["tool_calls"]
            ],
        )

        return tool_results, tool_request_msg

    def _formatToolResults(self, results):
        """Based on a list of dictionaries, returns a string that represents the tool outputs to an LLM."""
        # FIXME: currently only intended for command-r
        w = ""
        # w += "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
        for tool in results:
            if len(results) > 1:
                # tell the LLM what tool has which results
                w += "## " + tool["tool_name"] + "\n\n"
            w += agency.showToolResult(tool["output"])
        # w += "<|END_OF_TURN_TOKEN|>"
        return w

    def appendOption(self, name, value, duplicates=True):
        if name not in self.options:
            printerr("warning: unrecognized option '" + name + "'")
            return

        xs = self.getOption(name)
        if type(xs) != type([]):
            printerr(
                "warning: attempt to append to '" + name + "' when it is not a list."
            )
            return

        if not (duplicates):
            if value in xs:
                return
        self.options[name].append(value)

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False
        while not (self._freeze_queue.empty()):
            (name, value) = self._freeze_queue.get(block=False)
            self.setOption(name, value)
            if self.getOption("verbose"):
                printerr("unfreezing " + name + " : " + str(value))

    def setOption(self, name, value):
        # we freeze state during some parts of execution, applying options after we unfreeze
        if self._frozen:
            self._freeze_queue.put((name, value))
            return

        # mode gets to call dibs
        if name == "mode":
            self.setMode(value)
            return

        oldValue = self.options.get(name, None)
        differs = self.optionDiffers(name, value)
        self.options[name] = value
        # for some options we do extra stuff
        if (
            name == "tts_voice"
            or name == "tts_volume"
            or name == "tts_tortoise_quality"
        ) and self.getOption("tts"):
            # we don't want to restart tts on /restart
            if differs:
                # printerr("Restarting TTS.")
                self.tts_flag = True  # restart TTS
        elif name == "no-tts":
            self.setOption("tts", not (value))
        elif name == "whisper_model":
            self.whisper = self._newTranscriber()
        # elif name == "cli_prompt":
        # self.initial_cli_prompt = value
        elif name == "tts_websock":
            if value:
                self.setOption("tts_output_method", TTSOutputMethod.websock.name)
            else:
                self.setOption("tts_output_method", TTSOutputMethod.default.name)
            if differs and self.getOption("tts"):
                self.tts_flag = True  # restart TTS
        elif name == "max_context_length":
            self._dirtyContextLlama = False
        elif name == "prompt_format":
            self.loadTemplate(value)
        elif name == "chat_user":
            # userpormpt might be in stopwords, which we have to refresh
            prompt = mkChatPrompt(oldValue)
            badwords = [prompt, prompt.strip()]
            self.options["stop"] = list(
                filter(lambda w: w not in badwords, self.getOption("stop"))
            )

            # and this will add the new username if it's chat mode
            self.setMode(self.getMode())
        elif name == "stop":
            # user may change this in a char config.json. This may be buggy, but let's make sure at least the template is in.
            if self.template:
                self.options[name] = self.template.stops()
            else:
                self.options[name] = []
            self.options[name] += value
        elif name == "chat_ai":
            self.session.setVar(name, value)
        elif name == "http" and differs:
            if value:
                self._initializeHTTP()
            else:
                self._stopHTTP()
        elif name == "websock" and differs:
            if value:
                self.initializeWebsock()
            else:
                self.stopWebsock()
        elif name == "image_watch" and differs:
            if value:
                self.startImageWatch()
            else:
                self.stopImageWatch()
        elif name == "audio" and differs:
            if value:
                self.startAudioTranscription()
            else:
                self.stopAudioTranscription()
        elif name == "stderr":
            util.printerr_disabled = not (value)
        elif name == "use_tools" and value:
            printerr(
                "warning: Streaming is currently not supported with tool use. Setting stream = False."
            )
            self.setOption("stream", False)
        elif name == "stream":
            if value and self.getOption("tool_use"):
                printerr(
                    "warning: Streaming is currently not supported with tool use. Setting stream = False."
                )
                self.setOption("stream", False)
                printerr
        return ""

    def _ctPauseHandler(self, sig, frame):
        printerr("Recording paused. CTRL + c to resume, /text to stop.")
        self.ct.pause()
        signal.signal(signal.SIGINT, self._ctResumeHandler)

    def _ctResumeHandler(self, sig, frame):
        printerr("Recording resumed. CTRL + c to interrupt.")
        self.ct.resume()
        signal.signal(signal.SIGINT, self._ctPauseHandler)

    def _imageWatchCallback(self, image_path, image_id):
        newStory(self, [])
        w = self.getOption("image_watch_msg")
        if w == "":
            return

        self.loadImage(image_path, image_id)
        # FIXME: what if loadImage fails?
        (modified_w, hint) = self.modifyInput(w)
        self.addUserText(modified_w)
        image_watch_hint = self.getOption("image_watch_hint")
        self.addAIText(self.communicate(self.buildPrompt(hint + image_watch_hint)))
        # print(self.showCLIPrompt(), end="")

    def _print_generation_callback(self, result_str):
        """Pass this as callback to self.interact for a nice simple console printout of generations."""
        if result_str == "":
            return

        if not (self.getOption("stream")) and self.getOption("stdout"):
            self.print(
                self.getAIFormatter(with_color=self.getOption("color")).format(
                    result_str
                ),
                end="",
            )
        # self.print(self.showCLIPrompt(), end="", tts=False)
        return

    def suspendTranscription(self):
        """Renders transcription unresponsive until an activation phrase is heard or unsuspendTranscription is called."""
        self._transcription_suspended = True
        self.verbose("Suspending audio interaction.")

    def unsuspendTranscription(self):
        """Resumes interacting by audio after a suspension."""
        self._transcription_suspended = False
        self.verbose("Resuming audio interaction.")

    def modifyTranscription(self, w):
        """Checks wether an incoming user transcription by the whisper model contains activation phrases or is within timing etc. Returns the modified transcription, or the empty string if activation didn't trigger."""

        # want to match fuzzy, so strip of all punctuation etc.
        # FIXME: no need to do this here, again and again
        phrase = (
            self.getOption("audio_activation_phrase")
            .translate(str.maketrans("", "", string.punctuation))
            .strip()
            .lower()
        )
        if not (phrase):
            # user doesn't require phrase. proceed. unless we are suspended
            if not (self._transcription_suspended):
                return w

        # ok we require an activation phrase, but are we within the grace period where none is required?
        if (t := self.getOption("audio_activation_period_ms")) > 0:
            # grace period doesn't matter if we have been suspended.
            if (time_ms() - self._lastInteraction) <= t and not (
                self._transcription_suspended
            ):
                # FIXME: returning here means there might be a phrase in the input even when phrase_keep is false. Maybe it doesn't matter?
                return w

        # now we need an activation phrase or it ain't happenin
        # strip the transcription
        test = w.translate(str.maketrans("", "", string.punctuation)).strip().lower()
        try:
            n = test.index(phrase)
        except ValueError:
            # no biggie, phrase wasn't found. we're done here
            return ""

        # ok it was found
        self.unsuspendTranscription()
        if self.getOption("audio_activation_phrase_keep"):
            return w
        # FIXME: this is 100% not correct, but it may be good enough
        return w.replace(phrase, "").replace(
            self.getOption("audio_activation_phrase"), ""
        )

    def printActivationPhraseWarning(self):
        n = self.getOption("warn_audio_activation_phrase")
        if not (n):
            return
        w = (
            "warning: Your message was received but triggered no response, because the audio activation phrase was not found in it. Hint: The activation phrase is '"
            + self.getOption("audio_activation_phrase")
            + "'."
        )
        if n != -1:
            w += " This warning will only be shown once."
            self.setOption("warn_audio_activation_phrase", False)
        printerr(w)

    def _transcriptionCallback(self, w):
        """Supposed to be called whenever the whisper model has successfully transcribed a phrase."""
        if self.getOption("audio_show_transcript"):
            self.print(w, tts=False)

        w = self.modifyTranscription(w)
        if not (w):
            self.printActivationPhraseWarning()
            return

        if self._on_transcription is not None:
            w = self._on_transcription(w)

        self.interact(
            w,
            user_generation_callback=self._on_transcription_generation,
            generation_callback=self._print_generation_callback,
        )

    def _transcriptionOnThresholdCallback(self):
        """Gets called whenever the continuous transcriber picks up audio above the threshold."""
        if self.getOption("audio_interrupt"):
            self.stopAll()

        if self._on_activation is not None:
            self._on_activation()

    def _streamCallback(self, token, user_callback=None, only_once=None):
        if only_once not in self._stream_only_once_token_bag:
            # this is so that we can print tokens/sentences without interrupting the TTS every time
            # except that we want to interrupt the TTS exactly once -> when we start streaming
            # If you do this elsewhere, i.e. communicate(), we risk a race condition
            self._stream_only_once_token_bag.add(only_once)
            if self.getOption("tts_interrupt"):
                self.stopTTS()

        if user_callback is None:
            user_callback = lambda x: x
        f = lambda w: self.print(
            self.getAIColorFormatter().format(w), end="", flush=True, interrupt=False
        )

        self.stream_queue.append(token)
        method = self.getOption("stream_flush")
        if method == "token":
            f(token)
            user_callback(token)

        elif method == "sentence" or method == "flex":
            self.stream_sentence_queue.append(token)
            if "\n" in token:
                w = "".join(self.stream_sentence_queue)
            else:
                w = IncompleteSentenceCleaner().format(
                    "".join(self.stream_sentence_queue)
                )
                if w.strip() == "":
                    # not a complete sentence yet, let's keep building it
                    return

                if method == "flex" and len(w) < self.getOption(
                    "stream_flush_flex_value"
                ):
                    # sentence isn't long enough yet
                    return

            # w is a complete sentence, or a full line
            self.stream_sentence_queue = []
            f(w)
            user_callback(w)

    def flushStreamQueue(self):
        w = "".join(self.stream_queue)
        self.stream_queue = []
        self.stream_sentence_queue = []
        return w

    def isAudioTranscribing(self):
        return self.ct is not None and self.ct.running

    def startImageWatch(self):
        dir = self.getOption("image_watch_dir")
        printerr("Watching for new images in " + dir + ".")
        self.image_watch = AutoImageProvider(dir, self._imageWatchCallback)

    def startAudioTranscription(self):
        printerr("Beginning automatic transcription. CTRL + c to pause.")
        if self.ct:
            self.ct.stop()
        self.ct = self.whisper.transcribeContinuously(
            callback=self._transcriptionCallback,
            on_threshold=self._transcriptionOnThresholdCallback,
            websock=self.getOption("audio_websock"),
            websock_host=self.getOption("audio_websock_host"),
            websock_port=self.getOption("audio_websock_port"),
        )
        signal.signal(signal.SIGINT, self._ctPauseHandler)

    def stopImageWatch(self):
        if self.image_watch:
            printerr("Stopping watching of images.")
            self.image_watch.stop()

    def stopAudioTranscription(self):
        if self.ct:
            printerr("Stopping automatic audio transcription.")
            self.ct.stop()
        self.ct = None
        signal.signal(signal.SIGINT, self._defaultSIGINTHandler)

    def initializeTTS(self):
        tts_program = self.getOption("tts_program")
        candidate = os.getcwd() + "/" + tts_program
        if os.path.isfile(candidate):
            tts_program = candidate

        if not (tts_program):
            return "Cannot initialize TTS: No TTS program set."

        # download models if need be
        # ghostbox-tts does this as well, but it's stupidly tricky to get it's stderr to print to our stderr only for model download
        # so we do it here to give user some feedback on dl
        if tts_program == "ghostbox-tts":
            if (
                self.getOption("tts_model") == TTSModel.orpheus.name
                and self.getOption("tts_orpheus_model") == ""
            ):
                # the default, which is reasonable for most people, is to use the 4bit quant
                # if users want some more specific stuff they're going to have to work for it a little bit I guess
                downloaded = False
                if (
                    gguf_file := try_to_load_from_cache(
                        "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
                        "orpheus-3b-0.1-ft-q4_k_m.gguf",
                    )
                ) is not None:
                    if os.path.isfile(gguf_file):
                        downloaded = True

                if not (downloaded):
                    snapshot_download("isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF")

        # pick a voice in case of random
        if self.getOption("tts_voice") == "random":
            # no setOption to avoid recursion
            voices = getVoices(self)
            if voices == []:
                return "error: Cannot initialize TTS: No voices available."
            self.options["tts_voice"] = random.choice(voices)
            printerr("Voice '" + self.getOption("tts_voice") + "' was chosen randomly.")

        if self.tts is not None:
            # restarting
            try:
                if not (self.tts.is_running()):
                    self.tts.close()
            except ProcessLookupError:
                printerr(
                    "warning: TTS process got lost somehow. Probably not a big deal."
                )

        # let's make the path issue absolutely clear. We only track tts_voice_dir, but to the underlying tts program, we expose the tts_voice_abs_dir environment variable, which contains the absolute path to the voice dir
        # FIXME: rewrite the entire path architecture
        tts_voice_abs_dir = self.tryGetAbsVoiceDir()
        self.tts = feedwater.run(
            tts_program,
            env=envFromDict(
                self.options
                | {
                    "tts_voice_abs_dir": tts_voice_abs_dir,
                    "ONNX_PROVIDER": "CUDAExecutionProvider",
                }
            ),
        )
        self.setOption("stream_flush", "flex")
        if self.getOption("verbose"):
            printerr(
                " Automatically set stream_flush to 'flex'. This is recommended with TTS. Manually reset it to 'token' if you really want."
            )

        # new
        # we try to get the tts config
        # it contains special options specific to a model
        # unfortunately it will only be available after the model has finished initializing.
        # so we do it on a seperate thread
        self.spawnUpdateTTSConfig()
        return "TTS initialized."

    def spawnUpdateTTSConfig(self) -> None:
        """Spawns a small worker that tries to update the TTS config by communicating with the TTS backend.
        This may fail or simply go on forever. If the worker succeeds, the thread is terminated.
        Invoking this method is non-blocking, but will set self.tts_config to None."""
        self.tts_config = None

        def update_config():
            retries = 0
            max_retries = 3

            while self.tts_config is None:
                time.sleep(3)
                if self.tts is None:
                    continue

                try:
                    # this can fail e.g. if the tts isn't ready yet
                    # but will also fail with a broken pipe if the underlying process is already closed or the ghostbox deleted
                    # hence we give up after max_retries
                    self.tts.write_line("<dump_config>")
                    lines = self.tts.get()
                except:
                    retries += 1
                    if retries >= max_retries:
                        break
                    continue

                for i in range(len(lines) - 1, -1, -1):
                    line = lines[i]
                    if line.startswith("config: "):
                        try:
                            self.tts_config = json.loads(line[8:])
                        except:
                            printerr("warning: Error while reading tts config.")
                            self.verbose(traceback.format_exc())
                            # give up and don't keep trying
                            self.tts_config = {}

        t = threading.Thread(target=update_config, daemon=True)
        t.start()

    def tryGetTTSSpecialTags(self) -> List[str]:
        """Returns a list of special tags that the underlying tts model may or may not have defined.
        These are things like <laugh> or <cough>, etc.
        Returns empty list if no tags can be ascertained."""
        if self.tts_config is None:
            return []

        return self.tts_config.get("special_tags", [])

    def tryGetAbsVoiceDir(self):
        # this is sort of a heuristic. The problem is that we allow multiple include dirs, but have only one voice dir. So right now we must pick the best from a number of candidates.
        if os.path.isabs(self.getOption("tts_voice_dir")) and os.path.isdir(
            self.getOption("tts_voice_dir")
        ):
            return self.getOption("tts_voice_dir")

        winner = ""
        ok = False
        for path in self.getOption("include"):
            file = path + "/" + self.getOption("tts_voice_dir")
            if os.path.isdir(file):
                winner = file
                ok = True
                break

        abs_dir = os.path.abspath(winner)
        if not (ok):
            printerr(
                "warning: Couldn't cleanly determine tts_voice_dir. Guessing it is '"
                + abs_dir
                + "'."
            )
        return abs_dir

    def stopTTS(self):
        # FIXME: not implemented for all TTS clients
        if self.tts is None:
            return

        self.tts.write_line("<clear>")

    def communicateTTS(self, w, interrupt=False):
        if not (self.getOption("tts")):
            return ""

        if interrupt:
            self.stopTTS()

        # strip color codes - this would be nicer by just disabling color, but the fact of the matter is we sometimes want color printing on console and tts at the same time. At least regex is fast.
        w = stripANSI(w)

        # strip whitespace, we especially don't want to send pure whitespace like ' \n' or '  ' to a tts, this is known to crash some of them. It also shouldn't change the resulting output.
        w = w.strip()

        if self.tts is None or not (self.tts.is_running()):
            self.setOption("tts", False)
            printerr(
                "error: TTS is dead. You may attempt to restart with /tts. Check errors with /ttsdebug ."
            )
            return ""
        self.tts.write_line(w)
        return w

    def verbose(self, w: str):
        """Prints to stderr but only in verbose mode."""
        if self.getOption("verbose"):
            printerr(w)

    def console(self, w):
        """Print something to stderr. This exists to be used in tools.py via dependency injection."""
        printerr(w)

    def console_me(self, w):
        """Prints w to stderr, prepended by the AI name. This exists because it is a very common pattern to be used in tools.py"""
        printerr(self.getOption("chat_ai") + w)

    def _ringbuffer(self, w: str) -> str:
        """Takes a string w and returns it unchanged, but copies it to a local ringbuffer.
        The purpose of this is solely to let the CLI worker check if we have a newline at the end of the output.
        Note: not actually a ringbuffer."""
        self._stdout_ringbuffer = (self._stdout_ringbuffer + w)[
            : self._stdout_ringbuffer_size
        ]
        return w

    def print(
        self,
        w,
        end="\n",
        flush=False,
        color="",
        style="",
        tts=True,
        interrupt=None,
        websock=True,
    ):
        # either prints, speaks, or both, depending on settings
        if w == "":
            return

        if self.getOption("websock"):
            self.websockSend(w)

        if self.getOption("quiet"):
            return

        if tts and self.getOption("tts") and w != self.showCLIPrompt():
            self.communicateTTS(
                self.getTTSFormatter().format(w) + end,
                interrupt=(
                    self.getOption("tts_interrupt") if interrupt is None else interrupt
                ),
            )
            if not (self.getOption("tts_subtitles")):
                return

        if not (self.getOption("stdout")):
            return

        if not (color) and not (style):
            print(
                self._ringbuffer(self.getDisplayFormatter().format(w)),
                end=end,
                flush=flush,
            )
        else:
            print(
                self._ringbuffer(
                    style
                    + color
                    + self.getDisplayFormatter().format(w)
                    + Fore.RESET
                    + Style.RESET_ALL
                ),
                end=end,
                flush=flush,
            )

    def replaceForbidden(self, w):
        for forbidden in self.getOption("forbid_strings"):
            w = w.replace(forbidden, "")
        return w

    def bufferMultilineInput(self, w):
        if self.getOption("multiline"):
            # does not expect backslash at end
            self.multiline_buffer += w + "\n"
        else:
            # expects strings with \ at the end
            self.multiline_buffer += w[:-1] + "\n"

    def isMultilineBuffering(self):
        return self.multiline_buffer != ""

    def flushMultilineBuffer(self):
        w = self.multiline_buffer
        self.multiline_buffer = ""
        return w

    def expandDynamicFileVars(self, w):
        """Expands strings of the form `{[FILEPATH]}` by replacing them with the contents of FILE1 if it is found and readable."""
        if not (self.getOption("dynamic_file_vars")):
            return w

        pattern = r"\{\[(.*?)\]\}"
        matches = re.findall(pattern, w)

        for match in matches:
            var = "{[" + match + "]}"
            # users will probably mess around with this so we wanna be real careful
            common_error_msg = "error: Unable to expand '" + var + "'. "
            try:
                content = open(match, "r").read()
                w = w.replace(var, content)
            except FileNotFoundError:
                printerr(common_error_msg + "File not found.")
            except PermissionError:
                printerr(common_error_msg + "Operation not permitted.")
            except IsADirectoryError:
                printerr(common_error_msg + "'" + match + "' is a directory.")
            except UnicodeDecodeError:
                printerr(common_error_msg + "Unable to decode the file.")
            except IOError:
                printerr(common_error_msg + "Bad news: IO error.")
            except OSError:
                printerr(common_error_msg + "Operating system has signaled an error.")
            except:
                printerr(
                    common_error_msg
                    + "An unknown exception has occurred. Here's the full traceback."
                )
                printerr(traceback.format_exc())
        return w

    def modifyInput(prog, w):
        """Takes user input (w), returns pair of (modified user input, and a hint to give to the ai."""
        if (
            prog.isContinue() and prog.continue_with == ""
        ):  # user entered /cont or equivalent
            if prog.getMode().startswith("chat"):
                # prevent AI from talking for us
                if prog.showStory().endswith("\n"):
                    return ("", prog.adjustForChat("")[1])
            return ("", "")

        if (
            prog.continue_with != ""
        ):  # user input has been replaced with something else, e.g. a transcription
            w = prog.popContinueString()

        # dynamic vars must come before other vars for security reasons
        w = prog.expandDynamicFileVars(w)
        w = prog.session.expandVars(w)
        (w, ai_hint) = prog.adjustForChat(w)

        # tool_hint = agency.makeToolInstructionMsg() if prog.getOption("use_tools") else ""

        # user may also provide a hint. unclear how to best append it, we put it at the end
        user_hint = prog.session.expandVars(prog.getOption("hint"))
        if user_hint and prog.getOption("warn_hint"):
            printerr(
                "warning: Hint is set. Try /raw to see what you're sending. Use /set hint '' to disable the hint, or /set warn_hint False to suppress this message."
            )

        return (w, ai_hint + user_hint)

    def getSystemTokenCount(self):
        """Returns the number of tokens in system msg. The value is cached per session. Note that this adds +1 for the BOS token."""
        if self._systemTokenCount is None:
            # self._systemTokenCount = len(self.getBackend().tokenize(self.session.getSystem())) + 1
            self._systemTokenCount = (
                len(self.getBackend().tokenize(self.showSystem())) + 1
            )
        return self._systemTokenCount

    def getTemplate(self):
        return self.template

    def getRawTemplate(self):
        return RawTemplate()

    def showSystem(self):
        # vars contains system_msg and others that may or may not be replaced in the template
        vars = self.session.getVars().copy()
        if (
            self.getOption("use_tools")
            and (tool_hint := self.getOption("tools_hint")) != ""
        ):
            vars["system_msg"] += "\n" + tools_hint
        return self.getTemplate().header(**vars)

    def showStory(self, story_folder=None, append_hint=True) -> str:
        """Returns the current story as a unformatted string, injecting the current prompt template."""
        # new and possibly FIXME: we need to add another hint from agency.makeToolInstructionMsg when using tools, so we disable the hint here
        if self.getOption("use_tools"):
            append_hint = False

        if story_folder is None:
            sf = self.session.stories
        else:
            sf = story_folder
        if self.isContinue():
            # user hit enter and wants ai to keep talking. this is kind of like using the entire last reply as a hint -> no templating needed
            return self.getRawTemplate().body(
                sf.get(), append_hint, **self.session.getVars()
            )
        return self.getTemplate().body(sf.get(), append_hint, **self.session.getVars())

    def formatStory(self, story_folder=None, with_color=False):
        """Pretty print the current story (or a provided one) in a nicely formatted way. Returns pretty story as a string."""
        if story_folder is None:
            sf = self.session.stories
        else:
            sf = story_folder

        ws = []
        for item in sf.get().getData():
            if type(w := item.content) != str:
                continue
            if item.role == "assistant":
                ws.append(
                    (self.getAIColorFormatter() + self.getDisplayFormatter()).format(w)
                )
                continue
            ws.append(self.getDisplayFormatter().format(w))
        return "\n".join(ws)

    def buildPrompt(self, hint=""):
        """Takes an input string w and returns the full history (including system msg) + w, but adjusted to fit into the context given by max_context_length. This is done in a complicated but very smart way.
        returns - A string ready to be sent to the backend, including the full conversation history, and guaranteed to carry the system msg.
        """
        # FIXME: this is not smart. and it needs to be reworked badly. IIRC a lot of this was useful in the early days, now (march 2025) this kind of stuff is more obsolete because contexts are bigger and templates are handled server side
        # FIXME: it also just doesn't work since we changed how chat msgs are stored
        # another FIXME: also tokenizing needs to be fixed in the backends
        self._smartShifted = False  # debugging
        backend = self.getBackend()

        w = self.showStory() + hint
        gamma = self.getOption("max_context_length")
        k = self.getSystemTokenCount()
        n = self.getOption("max_length")
        # budget is the number of tokens we may spend on story history
        budget = gamma - (k + n)

        if budget < 0:
            # shit the bed
            return self.showSystem() + w

        if not (self.getOption("smart_context")):
            # currently we just dump FIXME: make this version keep the system prompt at least
            return self.showSystem() + w

        # now we need a smart story history that fits into budget
        sf = self.session.stories.copyFolder(only_active=True)
        while len(
            backend.tokenize(self.showStory(story_folder=sf) + hint)
        ) > budget and not (sf.empty()):

            # drop some items from the story, smartly, and without changing original
            self._smartShifted = True
            item = sf.get().pop(0)
            # FIXME: this can be way better, needs moretesting!
        return self.showSystem() + self.showStory(story_folder=sf) + hint

    def adjustForChat(self, w):
        """Takes user input w and returns a pair (w1, v) where w1 is the modified user input and v is a hint for the AI to be put at the end of the prompt."""
        v = ""
        if self.getMode() == "chat":
            w = mkChatPrompt(self.getOption("chat_user")) + w
            v = mkChatPrompt(self.session.getVar("chat_ai"), space=False)
        return (w, v)

    def communicate(self, prompt_text, stream_callback=None):
        """Sends prompt_text to the backend and returns results."""
        backend = self.getBackend()
        payload = self.makeGeneratePayload(prompt_text)
        self._lastPrompt = prompt_text
        if self.getOption("warn_trailing_space"):
            if prompt_text.endswith(" "):
                printerr(
                    "warning: Prompt ends with a trailing space. This messes with tokenization, and can cause the model to start its responses with emoticons. If this is what you want, you can turn off this warning by setting 'warn_trailing_space' to False."
                )

        # FIXME: this is only necessary until llamacpp implements streaming tool calls
        if self.getOption("stream"):
            # FIXME: this is the last hacky bit about formatting
            if self.getOption("chat_show_ai_prompt") and self.getMode().startswith(
                "chat"
            ):
                self.print(
                    self.session.getVar("chat_ai") + ": ",
                    end="",
                    flush=True,
                    interrupt=False,
                )
            if backend.generateStreaming(
                payload,
                lambda token, only_once=uuid.uuid4(): self._streamCallback(
                    token, user_callback=stream_callback, only_once=only_once
                ),
            ):
                printerr("error: " + backend.getLastError())
                return ""
            backend.waitForStream()
            self.setLastJSON(backend.getLastJSON())
            return self.flushStreamQueue()
        else:
            result = backend.handleGenerateResult(backend.generate(payload))
            self.setLastJSON(backend.getLastJSON())

        if not (result):
            printerr(
                "error: Backend yielded no result. Reason: " + backend.getLastError()
            )
            self.verbose(
                "Additional information (last request): \n"
                + json.dumps(backend.getLastRequest(), indent=4)
            )
            return ""
        return result

    def interact(
        self,
        w: str,
        user_generation_callback=None,
        generation_callback=None,
        stream_callback=None,
        blocking=False,
        timeout=None,
    ) -> None:
        """This is as close to a main loop as we'll get. w may be any string or user input, which is sent to the backend. Generation(s) are then received and handled. Certain conditions may cause multiple generations. Strings returned from the backend are passed to generation_callback.
        :param w: Any input string, which will be processed and may be modified, eventually being passed to the backend.
        :param generation_callback: A function that takes a string as input. This will receive generations as they arrive from the backend. Note that a different callback handles streaming responses. If streaming is enabled and this function prints, you will print twice. Hint: check for getOption('stream') in the lambda.
        :return: Nothing. Use the callback to process the results of an interaction, or use interactBlocking.
        """
        # internal state does not change during an interaction
        if generation_callback is None:
            generation_callback = self._print_generation_callback

        self._updateDatetime()
        self.freeze()

        def loop_interact(w):
            while self._busy.is_set():
                # another interaction might be happening
                # respect the jinja template 🔥🔥🔥
                time.sleep(0.1)

            self._stop_generation.clear()
            communicating = True
            if self._on_interaction is not None:
                self._on_interaction()

            self._busy.set()
            if self.getOption("history_force_alternating_roles"):
                self._ensureAlternatingRoles()
            (modified_w, hint) = self.modifyInput(w)
            self.addUserText(modified_w)
            while communicating:
                # this only runs more than once if there is auto-activation, e.g. with tool use
                if (
                    generated_w := self.communicate(
                        self.buildPrompt(hint), stream_callback=stream_callback
                    )
                ) == "":
                    # empty string usually means something went wrong.
                    # communicate will have already printed to stderr
                    # it is important that we don't loop forever here though, so we bail
                    communicating = False

                # if the generated string has tool calls, we apply them here
                tool_results, tool_call = self.applyTools(
                    generated_w, json=self.lastResult
                )

                output = ""
                if tool_results != []:
                    # need to add both the calls and result to the history
                    self.session.stories.get().addMessages([tool_call] + tool_results)
                    (w, hint) = ("", "")
                    if self.getOption("verbose"):
                        output = json.dumps(tool_results)
                else:
                    if self.getMode() != "chat":
                        output = self.addAIText(generated_w)
                    else:
                        # formatting is a bit broken in chat mode. Actually chat mode is a bit broken
                        output = generated_w
                    communicating = False

            # done communicating
            if user_generation_callback is not None:
                user_generation_callback(output)
            generation_callback(output)
            self.unfreeze()
            self._lastInteraction = time_ms()
            if self.getOption("history_force_alternating_roles"):
                self._ensureAlternatingRoles()

            self._busy.clear()
            if self._on_interaction_finished is not None:
                self._on_interaction_finished()

        t = threading.Thread(target=loop_interact, args=[w])
        t.start()
        if blocking:
            t.join(timeout=timeout)
        if self.getOption("log_time"):
            printerr(showTime(self, []))

    def interactBlocking(self, w: str, timeout=None) -> str:
        temp = ""

        def f(v):
            nonlocal temp
            temp = v

        self.interact(w, user_generation_callback=f, timeout=timeout, blocking=True)
        return temp

    def _stopInteraction(self):
        """Stops an ongoing generation, regardless of wether it is streaming, blocking, or async.
        This is a low level function. Consider using stopAll instead.
        Note: Currently only works for streaming generations."""
        streaming.stop_streaming.set()
        self._stop_generation.set()

    def stopAll(self):
        """Stops ongoing generation and interrupts the TTS.
        The side effects of stopping generation depend on the method used, i.e. streaming generation will yield some partially generated results, while a blocking generation may be entirely discarded.
        Note: Currently only works for streaming generations."""
        self._stopInteraction()
        self.stopTTS()

    def _aggressively_fix_history(self, history) -> None:
        """Ensures role alternates between assistant and user with extreme prejudice. Currently untested for tool calls etc."""
        # this is supposed to be called from _ensure_...
        if len(history) < 2:
            return

        # history must start with user msg
        # we did say 'aggressive'
        if history[0].role != "user":
            history.insert(0, ChatMessage(role="user", content="Hello."))

        # we know that history has at least 2 items
        # and starts with a user message
        i = 1
        while i < len(history):
            # we consider pairs throughout history
            before_msg = history[i - 1]
            current_msg = history[i]
            if before_msg.role == current_msg.role:
                # if roles happen to be the same, we merge them
                before_msg.content = before_msg.content + "\n" + current_msg.content
                del history[i]
            # if they are different, it's fine
            i += 1

    def _maybe_extend_last_user_message(self, new_message) -> str:
        """Part of the entire history_force_alternating_roles suite of methods.
        If the last message in history is of user role, this method deletes it and then returns new_message with the deleted message's contents prepended.
        """
        history = self.session.stories.get().data
        if len(history) == 0:
            return new_message

        msg = history[-1]
        if msg.role == "user":
            prepended_message = msg.content + "\n" + new_message
            del history[-1]
            return prepended_message
        return new_message

    def _ensureAlternatingRoles(self) -> None:
        """Rewrites chat history to ensure that 'assistant' and 'user' roles alternate."""
        # FIXME: this whole approach is kind of smelly
        # this only became necessary because llama.cpp started throwing errors and refusing to handle requests
        # with some models and jinja templates enabled, when roles don't alternate properly.
        # well, jinja is becoming the norm and works well when it doesn't do this kind of pedantry
        # so I just wanted a quick fix since it's extremely annoying when this comes up while I'm experimenting on something else entirely
        # I personally disagree with the entire philosophybehind this, and I think you should give the user the choice to make suboptimal or even broken queries.
        # ofc I understand the opposite view as well, where you force users to make good choices. I just think that approach doesn't fit llama.cpp, which is kind of DIY to the core
        # ultimately the two views are irreconcilable, and I think llama.cpp should add a --jinja-strict toggle
        history = self.session.stories.get().data
        # history is a list of chatmessages
        if history == []:
            return

        if len(history) == 1:
            # this is sort of a weird case
            # it can happen sometimes with old chars where we used to have initial_msg
            # so if it's an assistant message, we just pretend there was a user.
            msg = history[0]
            if msg.role == "assistant":
                fake_msg = ChatMessage(
                    role="user",
                    content="[System Message: User is initiating chat but has not send any message yet.]",
                )
            else:
                # only a user msg
                # this is even weirder
                fake_msg = ChatMessage(role="assistant", content="")

            history.insert(0, fake_msg)
            return

        # we have n messages in history
        # FIXME: right now, we defer to the ultra aggressive version, but this is no long term solution
        if True:
            # wrapped in if to not confuse black
            self._aggressively_fix_history(history)
            return
        else:
            # This is the less aggressive version, only fixing the last 2 msgs
            # for now, we will only consider the last two
            last = history[-1]
            before = history[-2]
            if last.role != before.role:
                # ok, history is safe
                # consider also that we might have tool calls
                # though I don't know maybe the jinja pope will throw a hissy fit about that as well
                # but I think this is a reasonably safe case
                return

            # here, we know the roles are equal, which is a problem
            if last.role == "assistant" or last.role == "user":
                # this feels wrong but oh well
                # we are going to merge the two messages
                before.content = before.content + "\n" + last.content
                history.pop(-1)
                return

    def hasImages(self):
        return bool(self.images) and self._images_dirty

    def loadImage(self, url, id):
        url = os.path.expanduser(url.strip())
        if not (os.path.isfile(url)):
            printerr("warning: Could not load image '" + url + "'. File not found.")
            return

        self.images[id] = ImageRef(url=url, data=loadImageData(url))
        self._images_dirty = True

    def setLastJSON(self, json_result):
        self.lastResult = json_result
        try:
            current_tokens = str(self.lastResult["usage"]["total_tokens"])
        except:
            # fail silently, it's not that important
            current_tokens = "?"

        self.session.setVar("current_tokens", current_tokens)

    def _updateDatetime(self) -> None:
        """Sets the datetime special var in the current session."""
        self.session.setVar("datetime", getAITime())

    def backup(self):
        """Returns a data structure that can be restored to return to a previous state of the program."""
        # copy strings etc., avoid copying high resource stuff or tricky things, like models and subprocesses
        return (self.session.copy(), copy.deepcopy(self.options))

    def restore(self, backup):
        (session, options) = backup
        self.session = session
        for k, v in options.items():
            self.setOption(k, v)

    def _stopHTTP(self):
        self.http_server.close()

    def _initializeHTTP(self):
        """Spawns a simple web server on its own thread.
        This will only serve the html/ folder included in ghostbox, along with the js files. This includes a minimal UI, and capabilities for streaming TTS audio and transcribing from user microphone input.
        Note: By default, this method will override terminal, audio and tts websock addresses. Use --no-http_override to suppress this behaviour.
        """
        import http.server
        import socketserver

        host, port = self.getOption("http_host"), self.getOption("http_port")

        # take care of terminal, audio and tts
        if self.getOption("http_override"):
            config = {
                "audio_websock_host": host,
                "audio_websock_port": port + 1,
                "audio_websock": True,
                "tts_websock_host": host,
                "tts_websock_port": port + 2,
                "tts_websock": True,
                "websock_host": host,
                "websock_port": port + 100,
                "websock": True,
            }

            for opt, value in config.items():
                self.setOption(opt, value)

        handler = partial(
            http.server.SimpleHTTPRequestHandler,
            directory=ghostbox.get_ghostbox_html_path(),
        )

        def httpLoop():
            with socketserver.TCPServer((host, port), handler) as httpd:
                printerr(
                    f"Starting HTTP server. Visit http://{host}:{port} for the web interface."
                )
                self.http_server = httpd
                httpd.serve_forever()

        self.http_thread = threading.Thread(target=httpLoop)
        self.http_thread.daemon = True
        self.http_thread.start()

    def initializeWebsock(self):
        """Starts a simple websocket server that sends and receives messages, behaving like a terminal client."""
        printerr("Initializing websocket server.")
        self.websock_server_running.set()
        self.websock_thread = threading.Thread(
            target=self._runWebsockServer, daemon=True
        )
        self.websock_thread.start()

        self.websock_regpl_thread = threading.Thread(
            target=regpl, args=[self, self._websockPopMsg], daemon=True
        )
        self.websock_regpl_thread.start()

        # handle printerr
        # FIXME: currently, we are not buffering while we send to a websock server, oh well
        util.printerr_callback = lambda w: self.websockSend(self.stderr_token + w)

    def stopWebsock(self):
        printerr("Stopping websocket server.")
        self.websock_server_running.clear()
        self.websock_clients = []
        self._printerr_buffer = ["Resetting stderr buffer."]
        util.printerr_callback = self._initial_printerr_callback

    def websockSend(self, msg: str) -> None:
        from websockets import ConnectionClosedError

        for i in range(len(self.websock_clients)):
            client = self.websock_clients[i]
            try:
                client.send(msg)
            except ConnectionClosedError:
                printerr(
                    "error: Unable to send message to "
                    + str(client.remote_address)
                    + ": Connection closed."
                )
                del self.websock_clients[i]

    def _websockPopMsg(self) -> str:
        """Pops a message of the internal websocket queue and returns it.
        This function blocks until a message is found on the queue."""
        while self.websock_server_running.is_set():
            try:
                return self.websock_msg_queue.get(timeout=1)
            except Empty:
                # timeout was hit
                # future me: don't remove this: get will super block all OS signals so we have to occasionally loop around or program will be unresponsive
                continue
            except:
                print(
                    "Exception caught while blocking. Shutting down gracefully. Below is the full exception."
                )
                printerr(traceback.format_exc())
                self.stopWebsock()

    def _runWebsockServer(self):
        import websockets
        import websockets.sync.server as WS

        def handler(websocket):
            remote_address = websocket.remote_address
            # printerr("[WEBSOCK] Got connection from " + str(remote_address))
            # update the new client with console log
            for msg in self._printerr_buffer:
                try:
                    websocket.send(self.stderr_token + msg)
                except ConnectionClosedError:
                    printerr(
                        "error: Connection with "
                        + str(remote_address)
                        + " closed during initial history update."
                    )
                    return

            self.websock_clients.append(websocket)
            try:
                while self.websock_server_running.is_set():
                    msg = websocket.recv()
                    if type(msg) == str:
                        self.websock_msg_queue.put(msg)
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websock_clients.remove(websocket)

        self.websock_server_running.set()
        self.websock_server = WS.serve(
            handler, self.getOption("websock_host"), self.getOption("websock_port")
        )
        printerr(
            "WebSocket server running on ws://"
            + self.getOption("websock_host")
            + ":"
            + str(self.getOption("websock_port"))
        )
        self.websock_server.serve_forever()

    def _startCLIPrinter(self) -> None:
        """Checks wether we are busy. If we are busy and then aren't, we print the CLI prompt."""
        # we don't want to use the self.print, nor printerr for this as both of them have side effects
        # this really is just for terminal users
        print_cli = lambda prefix="": print(
            prefix + self.showCLIPrompt(), file=sys.stderr, end="", flush=True
        )

        def cli_printer():
            while self.running:
                self._busy.wait()
                while self._busy.is_set():
                    # don't blow up potato cpu
                    # user can wait for their coveted cli for 10ms
                    time.sleep(0.1)
                    continue

                if self.getOption("quiet"):
                    continue

                # bingo
                if self._triggered:
                    # this means someone wrote a command etc so user pressed enter and we don't need a newline
                    print_cli(prefix="")
                    self._triggered = False
                    continue

                # AI generated a bunch of text. there may or may not be a newline. llama actually has this in the result json but we roll our own
                if self._stdout_ringbuffer.endswith("\n"):
                    print_cli()
                else:
                    print_cli(prefix="\n")

        self._cli_printer_thread = threading.Thread(target=cli_printer, daemon=True)
        self._cli_printer_thread.start()

    def triggerCLI(self, check: Optional[str] = None) -> None:
        """Signals that the user has pressed enter, usually executing a command.
        The entire purpose of this is to track wether we need to print a newline before printing the CLI prompt.
        :param check: If none, is checked for a newline at the end, determining wether to print a newline.
        """
        if check is None:
            self._triggered = True
        else:
            self._triggered = True if check.endswith("\n") else False

        # off to the races lulz
        self._busy.set()
        self._busy.clear()

    def ttsIsSpeaking(self) -> bool:
        """Returns true if the tts is currently speaking.
        If querying this is not supported by the tts_program, this method will always return False.
        """
        if self.getOption("tts_program") != "ghostbox-tts":
            return False

        if not (self.getOption("tts")):
            return False

        # to figure out if ghostbox-tts is speaking, we send a '<is_speaking> message
        # the return in stdout should be is_speaking: True'
        self.tts.write_line("<is_speaking>")
        # FIXME: find a better way for tts and ghostbox to  interact
        start_t = time.time()
        limit = 60.0
        while time.time() - start_t < limit:
            lines = self.tts.get()
            for line in lines:
                if "is_speaking: " in line:
                    return "is_speaking: True" in line
            time.sleep(0.1)
        printerr("warning: TTS is unresponsive when querying for is_speaking state.")
        return False


def main():
    just_fix_windows_console()
    tagged_parser = makeTaggedParser(backends.default_params)
    parser = tagged_parser.get_parser()
    args = parser.parse_args()
    if args.client:
        # we do something completely different
        client = GhostboxClient(
            remote_host=args.remote_host, remote_port=args.remote_port, logging=args.verbose
        )
        while client.running:
            try:
                w = input()
                if w == "/quit":
                    client.shutdown()
                else:
                    client.write_line(w)
            except EOFError:
                client.shutdown()
        return

    prog = Plumbing(
        options=args.__dict__,
        initial_cli_prompt=args.cli_prompt,
        tags=tagged_parser.get_tags(),
    )
    setup_plumbing(prog, args)

    if (prompt := prog.getOption("prompt")) is not None:

        def input_once():
            prog.running = False
            return prompt

        regpl(prog, input_function=input_once)
    else:
        regpl(prog)


def setup_plumbing(
    prog: Plumbing, args: Namespace = Namespace(), protected_keys: List[str] = []
) -> None:
    # the following is setup, though it is subtly different from Plumbing.init, so beware
    if userConfigFile():
        prog.setOption("user_config", userConfigFile())
        printerr(
            loadConfig(
                prog, [userConfigFile()], override=False, protected_keys=protected_keys
            )
        )

    if prog.getOption("config_file"):
        printerr(loadConfig(prog, [prog.options["config_file"]]))

    try:
        # this can fail if called from api
        if "-u" in sys.argv or "--chat_user" in sys.argv:
            prog.setOption("chat_user", args.chat_user)
    except AttributeError:
        # we don't have arg.xxx -> fine
        pass

    if prog.getOption("character_folder"):
        printerr(newSession(prog, []))

    if prog.getOption("hide"):
        hide(prog, [])

    if prog.getOption("tts"):
        prog.tts_flag = True

    if prog.getOption("image_watch"):
        del prog.options["image_watch"]
        prog.setOption("image_watch", True)

    if prog.getOption("http"):
        del prog.options["http"]
        prog.setOption("http", True)

    if prog.getOption("websock") and not (prog.websock_server_running.is_set()):
        del prog.options["websock"]
        prog.setOption("websock", True)

    if prog.getOption("audio") and prog.ct is None:
        del prog.options["audio"]
        prog.setOption("audio", True)


def regpl(prog: Plumbing, input_function: Callable[[], str] = input) -> None:
    """Read user input, evaluate, generate LLM response, print loop."""
    skip = False
    prog.triggerCLI()
    prog._busy.clear()

    # check if someone started the cli without a char and try to be helpful
    if not (prog.getOption("character_folder")):
        printerr(
            "warning: Running ghostbox without a character folder. Use `/start <charname>` to actually load an AI character.\n  For example `/start ghostbox-helper` will load a helpful AI that can explain ghostbox to you.\n  Other choices are: "
            + ", ".join(all_chars(prog))
        )

    while prog.running:
        last_state = prog.backup()
        try:
            # have to do TTS here for complex reasons; flag means to reinitialize tts, which can happen e.g. due to voice change
            if prog.tts_flag:
                prog.tts_flag = False
                prog.options["tts"] = False
                printerr(toggleTTS(prog, []))

            if prog.initial_print_flag:
                prog.initial_print_flag = False
                print("\n\n" + prog.formatStory(), end="")

            w = input_function()

            # check for multiline
            # this works different wether we have multiline mode enabled, or are doing ad-hoc multilines
            if prog.getOption("multiline"):
                # multiline mode
                if w != prog.getOption("multiline_delimiter"):
                    prog.bufferMultilineInput(w)
                    continue
                else:
                    # :-1 for trailing newline
                    w = prog.flushMultilineBuffer()[:-1]
            else:
                # ad hoc multilines
                if w.endswith("\\") and not (w.endswith("\\\\")):
                    prog.bufferMultilineInput(w)
                    continue
                elif prog.isMultilineBuffering():
                    w = prog.flushMultilineBuffer() + w

            # for convenience when chatting
            if w == "":
                # New: changed from
                # w = "/cont"
                # to the below because /cont wasn't used much and continue doesn't work very well with OAI API which is getting more prevalent
                prog.stopAll()
                prog.triggerCLI()
                continue

            # expand session vars, so we can do e.g. /tokenize {{system_msg}}
            w = prog.session.expandVars(w)

            for cmd, f in cmds:
                # FIXME: the startswith is dicey because it now makes the order of cmds defined above relevant, i.e. longer commands must be specified before shorter ones.
                if w.startswith(cmd):
                    # execute a / command
                    v = f(prog, w.split(" ")[1:])
                    printerr(v)
                    prog.triggerCLI(check=v)
                    if not (prog.isContinue()):
                        # skip means we don't send a prompt this iteration, which we don't want to do when user issues a command, except for the /continue command
                        skip = True
                    break  # need this to not accidentally execute multiple commands like /tts and /ttsdebug

            if skip:
                skip = False
                continue

            # this is the main event

            # for CLI use, we want a new generation to first stop all ongoing generation and TTS
            prog.stopAll()
            prog.interact(w, generation_callback=prog._print_generation_callback)

            prog.resetContinue()
        except KeyboardInterrupt:
            prog.running = False
            sys.exit
        except:
            printerr("error: Caught unhandled exception in main()")
            printerr(traceback.format_exc())
            try:
                prog.restore(last_state)
            except:
                printerr(
                    "error: While trying to recover from an exception, another exception was encountered. This is very bad."
                )
                printerr(traceback.format_exc())
                printerr(saveStoryFolder(prog, []))
                sys.exit()
            printerr("Restored previous state.")


if __name__ == "__main__":
    main()
