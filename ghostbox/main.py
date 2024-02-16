import requests, json, os, io, re, base64, random, sys, threading, subprocess, signal, tempfile
from functools import *
from colorama import just_fix_windows_console, Fore, Back, Style
from lazy_object_proxy import Proxy
import argparse
from ghostbox.commands import *
from ghostbox.autoimage import *
from ghostbox.output_formatter import *
from ghostbox.util import *
from ghostbox._argparse import *
from ghostbox.streaming import streamPrompt
from ghostbox.session import Session
from ghostbox.transcribe import WhisperTranscriber
from ghostbox.pftemplate import *
from ghostbox.backends import *
from ghostbox import backends


def showHelp(prog, argv):
    """
    List commands, their arguments, and a short description."""
    
    w = ""
    for (cmd_name, f) in cmds:
        if f.__doc__ is None:
            docstring = cmds_additional_docs.get(cmd_name, "")
        else:
            docstring = str(f.__doc__) 
        w += cmd_name + " " + docstring + "\n"
    printerr(w, prefix="")
    return ""

# these can be typed in at the CLI prompt
cmds = [
    ("/help", showHelp),
    ("/start", newSession),
    ("/switch", switch),
    ("/quit", exitProgram),
    ("/restart", lambda prog, argv: newSession(prog, [])),
    ("/print", printStory) ,
    ("/next", nextStory),
    ("/prev", previousStory),
    ("/story", gotoStory),
    ("/retry", retry),
    ("/rephrase", rephrase),
    ("/drop", dropEntry),
    ("/new", newStory),
    ("/clone", cloneStory),
    ("/log", lambda prog, w: printStory(prog, w, stderr=True, apply_filter=False)),
    ("!",  transcribe),
    ("/transcribe", transcribe),
    ("/audio", toggleAudio),
    ("/image_watch", toggleImageWatch),    
    ("/image", image),
    ("/time", showTime),
    ("/status", showStatus),
    ("/detokenize", detokenize),
    ("/tokenize", tokenize),
    ("/raw", showRaw),
    ("/debuglast", debuglast),
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
        ("/continue", doContinue)
]

# the mode formatters dictionary is a mapping from mode names to pairs of OutputFormatters (DISPLAYFORMATTER, TXTFORMATTER), where DISPLAYFORMATTER is applied to output printed to the screen in that mode, and TXTFORMATTER is applied to the text that is saved in the story. The pair is wrapped in a lambda that supplies a keyword dictionary.
mode_formatters = {
    "default" : lambda d: (DoNothing, DoNothing, DoNothing, CleanResponse),
    "chat" : lambda d: (DoNothing, NicknameRemover(d["chat_ai"]), NicknameFormatter(d["chat_user"]), ChatFormatter(d["chat_ai"]))
}

class Program(object):
    def __init__(self, options={}, initial_cli_prompt=""):
        self.options = options        
        self.backend = None
        self.initializeBackend(self.getOption("backend"), self.getOption("endpoint"))
        self.session = Session(chat_user=options.get("chat_user", ""))
        self.lastResult = {}
        self.tts_flag = False
        self.initial_print_flag = False
        self.initial_cli_prompt = initial_cli_prompt
        self.stream_queue = []
        self.stream_sentence_queue = []
        self. images = {}
        self._lastPrompt = ""
        self._dirtyContextLlama = False
        self._smartShifted = False
        self._systemTokenCount = None
        self.continue_with = ""
        self.tts = None
        self.multiline_buffer = ""
        if self.getOption("json"):
            self.setOption("grammar", getJSONGrammar())
            del self.options["json"]
        elif self.getOption("grammar_file"):
            self.loadGrammar(self.getOption("grammar_file"))
        else:
            self.setOption("grammar", "")
        # template
        self.loadTemplate(self.getOption("prompt_format"))
            
            # whisper stuff. We do this with a special init function because it's lazy
        self.whisper = self._newTranscriber()
        self.ct = None
        self._defaultSIGINTHandler = signal.getsignal(signal.SIGINT)        

        #imagewatching
        self.image_watch = None
        
        # formatters is to be idnexed with modes
        self.setMode(self.getOption("mode"))
        self.running = True

    def initializeBackend(self, backend, endpoint):
        self.backend = LlamaCPPBackend(endpoint)

    def getBackend(self):
        return self.backend
    
    def makeGeneratePayload(self, text):
        d = backends.default_params.copy()
        for key in d.keys():
            if key in self.options:
                d[key] = self.getOption(key)
        
        d["prompt"] = text
        # this one is temporarily disabled because of argparse so we have to do it here
        d["stream"] = self.getOption("stream")

        # these 2 have unintuitive names so we explicitly mention them here
        #d["n_ctx"] = self.getOption("max_context_length"), # this is sort of undocumented in llama.cpp server
        d["n_predict"] = self.getOption("max_length")
            
        if self.hasImages():
            d["image_data"] = [packageImageDataLlamacpp(d["data"], id) for (id, d) in self.images.items()]
        return d
            
    def _newTranscriber(self):
        # makes a lazy WhisperTranscriber, because model loading can be slow
        return Proxy(lambda: WhisperTranscriber(model_name = self.getOption("whisper_model"),
                                                silence_threshold = self.getOption("audio_silence_threshold"),
                                                input_func=lambda: printerr("Started Recording. Hit enter to stop.")))

    def    continueWith(self, newUserInput):
        # FIXME: the entire 'continue' architecture is a trashfire. This should be refactored along with other modeswitching/input rewriting stuff in the main loop
        self.setOption("continue","1")
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
            printerr("warning: grammar file " + grammar_file + " could not be loaded: file not found.")

    def loadTemplate(self, name):
        allpaths = [p + "/" + name for p in self.getOption("template_include")]
        for path in allpaths:
            path = os.path.normpath(path)
            if not(os.path.isdir(path)):
                failure = "Could not find template '" + name + "'. Did you supply a --template_include option?"
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
            self.options["prompt_format"] = 'raw'
            return
        # actually load template
        # first unload old stops
        self.options["stop"] = list(filter(lambda w: w not in self.template.stops(), self.options["stop"]))
            
        self.template = template
        for w in template.stops():
            if not(w):
                continue
            self.appendOption("stop", w)
        self.options["prompt_format"] = name
        printerr("Using '" + name + "' as prompt format template.")
                
    def loadConfig(self, json_data, override=True):
        """Loads a config file provided as json into options. Override=False means that command line options that have been provided will not be overriden by the config file."""
        d = json.loads(json_data)
        if type(d) != type({}):
            return "error loading config: Not a dictionary."
        if not(override):
            # drop keys in the config that can be found in the command line arguments
            # have to do the one letter options manually
            letterMap = {"u" : "chat_user", "c" : "character_folder", "V" : "tts_voice", "T" : "prompt_format"}
            for arg in sys.argv:
                if not(arg.startswith("-")):
                    continue
                key = stripLeadingHyphens(arg)
                # "no-" is for e.g. '--no-tts'
                if key in d or "no-"+key in d:
                    del d[key]
                for (letter, full_arg) in letterMap.items():
                    if key.startswith(letter):
                        if full_arg in d:
                            del d[full_arg]

        # now actually load the options, with a partial ordering
        items = sorted(d.items(), key=cmp_to_key(lambda a, b: -1 if a[0] == "mode" else 1))
        for (k, v) in items:
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

        return f.format(self.getOption("cli_prompt"))
            
    def getMode(self):
        w = self.getOption("mode")
        if not(self.isValidMode(w)):
            return 'default'
        return w

    def isValidMode(self, mode):
        return mode in mode_formatters

    def setMode(self, mode):
        if not(self.isValidMode(mode)):
            return
        
        self.options["mode"] = mode
        if mode.startswith("chat"):
            userPrompt = mkChatPrompt(self.getOption("chat_user"))
            self.setOption("cli_prompt", "\n" + userPrompt)
            self.appendOption("stop", userPrompt, duplicates=False)
            self.appendOption("stop", userPrompt.strip(), duplicates=False)            

        else: # default
            self.options["cli_prompt"] = self.initial_cli_prompt
               
    def getOption(self, key):
        return self.options.get(key, False)

    def optionDiffers(self, name, newValue):
        if name not in self.options:
            return True
        return self.getOption(name) == newValue

    def getFormatters(self):
        mode = self.getOption("mode")
        if not(self.isValidMode(mode)):
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
            return ColorFormatter(self.getOption("text_ai_color"), self.getOption("text_ai_style"))
        return IdentityFormatter()
    
    def getAIFormatter(self, with_color=False):
        return self.getAIColorFormatter() + self.getFormatters()[3]        
    
    def addUserText(self, w):
        if w:
            self.session.stories.get().addUserText(self.getUserFormatter().format(w))

    def addAIText(self, w):
        if w:
            if self.getOption("continue"):
                self.session.stories.get().extendAssistantText(w)
            else:
                self.session.stories.get().addAssistantText(self.getAIFormatter().format(w))

    def appendOption(self, name, value, duplicates=True):
        if name not in self.options:
            printerr("warning: unrecognized option '" + name + "'")
            return

        xs = self.getOption(name)
        if type(xs) != type([]):
            printerr("warning: attempt to append to '" + name + "' when it is not a list.")
            return

        if not(duplicates):
            if value in xs:
                return
        self.options[name].append(value)
            
            
            
                
    def setOption(self, name, value):
        # mode gets to call dibs
        if name == "mode":
            self.setMode(value)
            return
        
        oldValue = self.getOption(name)
        differs = self.optionDiffers(name, value)
        self.options[name] = value
        # for some options we do extra stuff
        if (name == "tts_voice" or name == "tts_volume") and self.getOption("tts"):
            # we don't want to restart tts on /restart
            if differs:
                self.tts_flag = True #restart TTS
        elif name == "no-tts":
            self.setOption("tts", not(value))
        elif name == "whisper_model":
            self.whisper = self._newTranscriber()
        #elif name == "cli_prompt":
            #self.initial_cli_prompt = value
        elif name == "max_context_length":
            self._dirtyContextLlama = False
        elif name =="prompt_format":
            self.loadTemplate(value)
        elif name == "chat_user":
            # userpormpt might be in stopwords, which we have to refresh
            prompt = mkChatPrompt(oldValue)
            badwords = [prompt, prompt.strip()]
            self.options["stop"] = list(filter(lambda w: w not in badwords, self.getOption("stop")))

            # and this will add the new username if it's chat mode
            self.setMode(self.getMode())

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
        (modified_w, hint) = self.modifyInput(w)
        self.addUserText(modified_w)
        image_watch_hint = self.getOption("image_watch_hint")
        self.addAIText(self.communicate(self.buildPrompt(hint + image_watch_hint)))
        print(self.showCLIPrompt(), end="")

    def _transcriptionCallback(self, w):
        (modified_w, hint) = self.modifyInput(w)
        self.addUserText(modified_w)
        if self.getOption("audio_show_transcript"):
            print(w)
        self.addAIText(self.communicate(self.buildPrompt(hint)))
        print(self.showCLIPrompt(), end="")

    def _streamCallback(self, token):
        self.stream_queue.append(token)
        method = self.getOption("stream_flush")
        if method == "token":
            self.print(self.getAIColorFormatter().format(token), end="", flush=True)
        elif method == "sentence":
            self.stream_sentence_queue.append(token)            
            w = IncompleteSentenceCleaner().format("".join(self.stream_sentence_queue))
            if w.strip() == "":
                # not a complete sentence yet, let's keep building it
                return
            # w is a complete sentence
            self.stream_sentence_queue = []
            self.print(self.getAIColorFormatter().format(w), end="", flush=True)

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
        self.ct = self.whisper.transcribeContinuously(callback=self._transcriptionCallback)
        signal.signal(signal.SIGINT, self._ctPauseHandler)

    def stopImageWatch(self):
        if self.image_watch:
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
            
        voice_dir = self.getOption("tts_voice_dir")
        voicefile = self.getOption("tts_voice")
        
        if not(tts_program):
            return "Cannot initialize TTS: No TTS program set."

        voice_args = [""]
        if voicefile:
            file = voice_dir + "/" + voicefile
            if os.path.isfile(file):
                voice_args = ["-V", file]
            else:
                #FIXME: this crashes if the file doesn't exist. maybe that's ok
                voice_args = ["-V", voicefile]


        if self.tts is not None:
            # since shell=true spawns child processes that may still be running , we have to terminate by sending kill signal to entire process group
            # FIXME: this doesn't work on windows
            os.killpg(os.getpgid(self.tts.pid), signal.SIGTERM)

                    
        cmd = [tts_program] + voice_args + ["--volume=" + str(self.getOption("tts_volume"))]
        cmdstring = " ".join(cmd)
        self.tts = subprocess.Popen(cmdstring,
                                    text=True,
                                    stdin=subprocess.PIPE,
                                    shell=True,
                                    preexec_fn=os.setsid,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        return ""

    def communicateTTS(self, w):
        if not(self.getOption("tts")):
            return ""

        # strip color codes - this would be nicer by just disabling color, but the fact of the matter is we sometimes want color printing on console and tts at the same time. At least regex is fast.
        w = stripANSI(w)

        # strip whitespace, we especially don't want to send pure whitespace like ' \n' or '  ' to a tts, this is known to crash some of them. It also shouldn't change the resulting output.
        w = w.strip()
        
        # this is crazy
        self.tts.stdin.flush()
        self.tts.stdout.flush()
        self.tts.stderr.flush()        
        self.tts.stdin.write(w + "\n")
        self.tts.stdin.flush()
        self.tts.stdout.flush()
        self.tts.stderr.flush()
        return w

    def print(self, w, end="\n", flush=False, color="", style=""):
        # either prints, speaks, or both, depending on settings
        if w == "":
            return
        
        if self.getOption("tts"):
            self.communicateTTS(self.getTTSFormatter().format(w) + end)
            if not(self.getOption("tts_subtitles")):
                return

        if not(color) and not(style):
            print(self.getDisplayFormatter().format(w), end=end, flush=flush)
        else:
            print(style + color + self.getDisplayFormatter().format(w) + Fore.RESET + Style.RESET_ALL, end=end, flush=flush)

    def replaceForbidden(self, w):
        for forbidden in self.getOption("forbid_strings"):
            w = w.replace(forbidden, "")
        return w

    def bufferMultilineInput(self, w):
        #expects strings with \ at the end
        self.multiline_buffer += w[:-1] + "\n"

    def isMultilineBuffering(self):
        return self.multiline_buffer != ""
            
    def flushMultilineBuffer(self):
        w = self.multiline_buffer
        self.multiline_buffer = ""
        return w
    
    def modifyInput(prog, w):
        """Takes user input (w), returns pair of (modified user input, and a hint to give to the ai."""
        if prog.getOption("continue") and prog.continue_with == "": # user entered /cont or equivalent
            #setOption(prog, ["continue", "False"])
            if prog.getMode().startswith("chat"):
                # prevent AI from talking for us
                if prog.showStory().endswith("\n"):
                    return ("", prog.adjustForChat("")[1])
            return ("", "")

        if prog.continue_with != "":# user input has been replaced with something else, e.g. a transcription
            w = prog.popContinueString()

        w = prog.session.expandVars(w)
        (w, ai_hint) = prog.adjustForChat(w)
        # user may also provide a hint. unclear how to best append it, we put it at the end
        user_hint = prog.session.expandVars(prog.getOption("hint"))
        if user_hint and prog.getOption("warn_hint"):
            printerr("warning: Hint is set. Try /raw to see what you're sending. Use /set hint '' to disable the hint, or /set warn_hint False to suppress this message.")
            
        return (w, ai_hint + user_hint)

    def getSystemTokenCount(self):
        """Returns the number of tokens in system msg. The value is cached per session. Note that this adds +1 for the BOS token."""
        if self._systemTokenCount is None:
            self._systemTokenCount = len(self.getBackend().tokenize(self.session.getSystem())) + 1
        return self._systemTokenCount

    def getTemplate(self):
        return self.template

    def getRawTemplate(self):
        return RawTemplate()
    
    def showSystem(self):
        return self.getTemplate().header(**self.session.getVars())

    def showStory(self, story_folder=None, append_hint=True):
        """Returns the current story as a unformatted string, injecting the current prompt template.""" 
        if story_folder is None:
            sf = self.session.stories
        else:
            sf = story_folder
        if self.getOption("continue"):
            # user hit enter and wants ai to keep talking. this is kind of like using the entire last reply as a hint -> no templating needed
            return self.getRawTemplate().body(sf.get(), append_hint, **self.session.getVars())
        return self.getTemplate().body(sf.get(), append_hint, **self.session.getVars())

    def formatStory(self, story_folder=None, with_color=False):
        """Pretty print the current story (or a provided one) in a nicely formatted way. Returns pretty story as a string."""
        if story_folder is None:
            sf = self.session.stories
        else:
            sf = story_folder

        ws = []
        for item in sf.get().getData():
            w = item["content"]
            if item["role"] == "assistant":
                ws.append((self.getAIColorFormatter() + self.getDisplayFormatter()).format(w))
                continue
            ws.append(self.getDisplayFormatter().format(w))
        return "\n".join(ws)

            
            


    def buildPrompt(self,hint=""):
        """Takes an input string w and returns the full history (including system msg) + w, but adjusted to fit into the context given by max_context_length. This is done in a complicated but very smart way.
returns - A string ready to be sent to the backend, including the full conversation history, and guaranteed to carry the system msg."""
        # problem: the llm can only process text equal to or smaller than the context window
        # dumb solution (ds): make a ringbuffer, append at end, throw away the beginning until it fits into context window
        # problem with dumb solution: the system msg gets thrown out and the AI forgets the basics of who it is
        # slightly less dumb solution (slds): keep the system_msg at all costs, throw first half of the rest away when context is exceeded this is llama.cpp solution, but only if you supply n_keep = tokens of system_msg koboldcpp does this too, but they seem to be a bit smarter about it and make it more convenient. Advantage of this approach is that you will make better use of the cache, since at least half of the prompt after the system msg is guaranteed to be in cache.
        # problem with slds: This can cut off the story at awkward moments, especially if it's in the middle of sentence or prompt format relevant tokens, which can really throw some models off, especially in chat mode where we rely on proper formatting a lot
        # ghostbox (brilliant) solution (gbs): use metadata in the story history to semantically determine good cut-off points. usually, this is after an AI message, since those are more often closing-the-action than otherwise. Use template files to ensure syntactic correctness (e.g. no split prompt format tokens).
        # The problem with this (gbs) is that it is not making as good use of the cache, since if an earlier part of the prompt changed everything after it gets invalidated. However prompt eval is the easy part. Clearly though, there is a trade off. Maybe give user a choice between slds and gbs as trade off between efficiency vs quality?
        # Also: this is currently still quite dumb, actually, since we don't take any semantics into account
        # honorable mention of degenerate cases: If the system_msg is longer than the context itself, or users pull similar jokes, it is ok to shit the bed and let the backend truncate the prompt.
        self._smartShifted = False #debugging
        backend = self.getBackend()
        
        w = self.showStory() + hint
        gamma = self.getOption("max_context_length")        
        k = self.getSystemTokenCount()
        n = self.getOption("max_length")
        # budget is the number of tokens we may spend on story history        
        budget = gamma - (k + n)

        if budget < 0:
            #shit the bed
            return self.showSystem() + w

        # now we need a smart story history that fits into budget
        sf = self.session.stories.copyFolder(only_active=True)
        while len(backend.tokenize(self.showStory(story_folder=sf) + hint)) > budget and not(sf.empty()):

        # drop some items from the story, smartly, and without changing original
            self._smartShifted = True
            item = sf.get().pop(0)
            #FIXME: this can be way better, needs moretesting!
        return self.showSystem() + self.showStory(story_folder=sf) + hint
        
    def adjustForChat(self, w):
        """Takes user input w and returns a pair (w1, v) where w1 is the modified user input and v is a hint for the AI to be put at the end of the prompt."""
        v = ""
        if self.getMode() == "chat":
            w = mkChatPrompt(self.getOption("chat_user")) + w
            v = mkChatPrompt(self.session.getVar("chat_ai"), space=False)

        return (w, v)
    
    def communicate(self, prompt_text):
        """Sends prompt_text to the backend and prints results."""
        backend = self.getBackend()
        payload = self.makeGeneratePayload(prompt_text)
        self._lastPrompt = prompt_text
        if self.getOption("warn_trailing_space"):
            if prompt_text.endswith(" "):
                printerr("warning: Prompt ends with a trailing space. This messes with tokenization, and can cause the model to start its responses with emoticons. If this is what you want, you can turn off this warning by setting 'warn_trailing_space' to False.")

        if self.getOption("stream"):
            # FIXME: this is the last hacky bit about formatting
            if self.getOption("chat_show_ai_prompt") and self.getMode().startswith("chat"):
                self.print(self.session.getVar("chat_ai") + ": ", end="", flush=True)
                    
            if backend.generateStreaming(payload, self._streamCallback):
                printerr("error: " + backend.getLastError())
                return ""
            backend.waitForStream()
            self.setLastJSON(backend.getLastJSON())
            return self.flushStreamQueue()
        else:
            result = backend.handleGenerateResult(backend.generate(payload))
            self.setLastJSON(backend.getLastJSON())            

        if not(result):
            printerr("error: " + backend.getLastError())
            return ""
        # FIXME: we're currently formatting the AI string twice. Here and in addAIText. that's not a big deal, though                
        self.print(self.getAIFormatter(with_color=self.getOption("color")).format(result), end="")
        return result            

    def hasImages(self):
        return bool(self.images)
        
    def loadImage(self, url, id):
        url = os.path.expanduser(url)
        if not(os.path.isfile(url)):
            printerr("warning: Could not load image '" + url + "'. File not found.")
            return

        self.images[id] = {"url" : url,
                           "data" : loadImageData(url)}

    def setLastJSON(self, json_result):
        self.lastResult = json_result
        if self.getOption("backend") == "llama.cpp":
            # llama does not allow to set the context size by clients, instead it dictates it server side. however i have not found a way to query it directly, it just gets set after the first request
            self.setOption("max_context_length", json_result["generation_settings"]["n_ctx"])
            self._dirtyContextLlama = True # context has been set by llama

    def backup(self):
        """Returns a data structure that can be restored to return to a previous state of the program."""
# copy strings etc., avoid copying high resource stuff or tricky things, like models and subprocesses
        return (copy.deepcopy(self.session), copy.deepcopy(self.options))

    def restore(self, backup):
        (session, options) = backup
        self.session = session
        for (k, v) in options.items():
            self.setOption(k, v)
            

                
def main():
    just_fix_windows_console()
    parser = makeArgParser(backends.default_params)
    args = parser.parse_args()
    prog = Program(options=args.__dict__, initial_cli_prompt=args.cli_prompt)
    if userConfigFile():    
        prog.setOption("user_config", userConfigFile())
        printerr(loadConfig(prog, [userConfigFile()], override=False))
    
    if prog.getOption("config_file"):
        printerr(loadConfig(prog, [prog.options["config_file"]]))

    # FIXME: this might also have to be done for other variables in the future, at which point we refactor and generalize
    if "-u" in sys.argv or "--chat_user" in sys.argv:
        prog.setOption("chat_user", args.chat_user)
        
    if args.character_folder:
        printerr(        newSession(prog, []))

    if prog.getOption("hide"):
        hide(prog, [])
        # FIXME: this shouldn't be deleted so that it stays persistent when user does /start or /restart, but it's also a useless option. implement a emchanism for hidden options?
    #del prog.options["hide"]

    if prog.getOption("tts"):
        prog.tts_flag = True
    
    if prog.getOption("audio"):
        prog.startAudioTranscription()
    del prog.options["audio"]

    if prog.getOption("image_watch"):
        prog.startImageWatch()
        del prog.options["image_watch"]
        
    
    skip = False        
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

            w = input(prog.showCLIPrompt())
            # check for multiline
            if w.endswith("\\") and not(w.endswith("\\\\")):
                prog.bufferMultilineInput(w)
                continue
            elif prog.isMultilineBuffering():
                w = prog.flushMultilineBuffer() + w

            # for convenience when chatting
            if w == "":
                w = "/cont"

            # expand session vars, so we can do e.g. /tokenize {{system_msg}}
            w = prog.session.expandVars(w)

            for (cmd, f) in cmds:
                #FIXME: the startswith is dicey because it now makes the order of cmds defined above relevant, i.e. longer commands must be specified before shorter ones. 
                if w.startswith(cmd):
                    v = f(prog, w.split(" ")[1:])
                    printerr(v)
                    if not(prog.getOption("continue")):
                        # skip means we don't send a prompt this iteration, which we don't want to do when user issues a command, except for the /continue command
                        skip = True
                    break #need this to not accidentally execute multiple commands like /tts and /ttsdebug

            if skip:
                skip = False
                continue

            # this is the main event
            (modified_w, hint) = prog.modifyInput(w)
            prog.addUserText(modified_w)
            prog.addAIText(prog.communicate(prog.buildPrompt(hint)))
            setOption(prog, ["continue", "False"])
        except:
            printerr("error: Caught unhandled exception in main()")
            printerr(traceback.format_exc())
            try:
                prog.restore(last_state)
            except:
                printerr("error: While trying to recover from an exception, another exception was encountered. This is very bad.")
                printerr(traceback.format_exc())
                printerr(saveStoryFolder(prog, []))
                sys.exit()
            printerr("Restored previous state.")
                      
                

                
                

                

if __name__ == "__main__":
    main()
