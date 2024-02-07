import requests, json, os, io, re, base64, random, sys, threading, subprocess, signal
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
    ("/quit", exitProgram),
    ("/restart", lambda prog, argv: newSession(prog, [])),
    ("/print", printStory) ,
    ("/next", nextStory),
    ("/prev", previousStory),
    ("/story", gotoStory),
    ("/retry", retry),
    ("/resend", lambda prog, argv: retry(prog, argv, lambda item: item["text"] != "" and item["user_generated"] == True)),
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
    ("/tokenize", tokenize),
    ("/raw", showRaw),
    ("/debuglast", debuglast),
    ("/ttsdebug", ttsDebug),    
    ("/tts", toggleTTS),
    ("/set", setOption),
    ("/unset", lambda prog, argv: setOption(prog, [argv[0]])),
    ("/saveoptions", saveConfig),
    ("/saveconfig", saveConfig),    
    ("/loadconfig", loadConfig),
    ("/save", saveStoryFolder),
    ("/load", loadStoryFolder),
    ("/varfile", varfile),
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
    
#defined here for convenience. Can all be changed through cli parameters or with / commands
DEFAULT_PARAMS = {                "rep_pen": 1.1, "temperature": 0.7, "top_p": 0.92, "top_k": 0, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 320, "rep_pen_slope": 0.7, "sampler_order": [6, 0, 1, 3, 4, 2, 5], "quiet": True, "use_default_badwordsids": True}

class Program(object):
    def __init__(self, options={}, initial_cli_prompt=""):
        self.session = Session(chat_user=options.get("chat_user", ""))
        self.template = FilePFTemplate("templates/chat-ml")
        self.lastResult = {}
        self.tts_flag = False
        self.initial_print_flag = False
        self.initial_cli_prompt = initial_cli_prompt
        self.streaming_done = threading.Event()
        self.stream_queue = []
        self. images = {}
        self._lastPrompt = ""
        self._dirtyContextLlama = False
        self._smartShifted = False
        self._systemTokenCount = None
        self.continue_with = ""
        self.tts = None
        self.multiline_buffer = ""
        self.options = options
        if self.getOption("json"):
            self.setOption("grammar", getJSONGrammar())
            del self.options["json"]
        elif self.getOption("grammar_file"):
            self.loadGrammar(self.getOption("grammar_file"))
        else:
            self.setOption("grammar", "")

            # whisper stuff. We do this with a special init function because it's lazy
        self.whisper = self._newTranscriber()
        self.ct = None
        self._defaultSIGINTHandler = signal.getsignal(signal.SIGINT)        

        #imagewatching
        self.image_watch = None
        
        # formatters is to be idnexed with modes
        self.setMode(self.getOption("mode"))
        self.running = True

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
            



        
    def loadConfig(self, json_data, override=True):
        d = json.loads(json_data)
        if type(d) != type({}):
            return "error loading config: Not a dictionary."

        if not(override):
            # drop keys in the config that can be found in the command line arguments
            for arg in sys.argv:
                key = stripLeadingHyphens(arg)
                if key in d:
                    del d[key]


        # used to be
        #self.options = self.options | d
        # which is nice, but unfortunately we have to go through setOption
        for (k, v) in d.items():
            self.setOption(k, v)
        return ""

    def showCLIPrompt(self):
        if self.isMultilineBuffering():
            return ""
        return self.getOption("cli_prompt")
    
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
            self.options["cli_prompt"] = "\n" + mkChatPrompt(self.getOption("chat_user"))            
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
        return mode_formatters[mode](self.options)

    def getDisplayFormatter(self):
        return self.getFormatters()[0]


    def getTTSFormatter(self):
        return self.getFormatters()[1]

    def getUserFormatter(self):
        return self.getFormatters()[2]

    def getAIFormatter(self):
        return self.getFormatters()[3]
    
    def addUserText(self, w):
        if w:
            self.session.stories.get().addUserText(self.getUserFormatter().format(w))

    def addAIText(self, w):
        if w:
            self.session.stories.get().addAssistantText(self.getAIFormatter().format(w))

    def setOption(self, name, value):
        self.options[name] = value
        # for some options we do extra stuff
        if name == "tts_voice" or name == "tts_volume":
            # we don't want to restart tts on /restart
            if self.optionDiffers(name, value):
                self.tts_flag = True #restart TTS
        elif name == "whisper_model":
            self.whisper = self._newTranscriber()
        elif name == "cli_prompt":
            self.initial_cli_prompt = value
        elif name == "max_context_length":
            self._dirtyContextLlama = False

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
        #FIXME: this is only temporary until we fix the context being exceeded with llamacpp. This happens very quickly with image data, and so we frequently restart the session to get the system message in.
        newSession(self, [])
        
        
        w = self.getOption("image_watch_msg")
        if w == "":
            return
        
        self.loadImage(image_path, image_id)
        (modified_w, hint) = self.modifyInput(w)
        self.addUserText(modified_w)
        self.addAIText(self.communicate(self.buildPrompt(hint)))
        print(self.showCLIPrompt(), end="")

        
    def _transcriptionCallback(self, w):
        (modified_w, hint) = self.modifyInput(w)
        self.addUserText(modified_w)
        if self.getOption("audio_show_transcript"):
            print(w)
        self.addAIText(self.communicate(self.buildPrompt(hint)))
        print(self.showCLIPrompt(), end="")
        
        
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
            
            

        
        
    def getPrompt(self, text):
        self._lastPrompt = text
        
        if self.getOption("warn_trailing_space"):
            if text.endswith(" "):
                printerr("warning: Prompt ends with a trailing space. This messes with tokenization, and can cause the model to start its responses with emoticons. If this is what you want, you can turn off this warning by setting 'warn_trailing_space' to False.")
        backend = self.getOption("backend")
        if backend == "koboldcpp":
            d = {"prompt": text + "",
                 "grammar" : self.getOption("grammar"),
                 "n": 1,
                 "max_context_length": self.getOption("max_context_length"),
                 "max_length": self.options["max_length"]}
        elif backend == "llama.cpp":
            d = {"prompt": text,
                 "grammar" : self.getOption("grammar"),
                 #"n_ctx": self.getOption("max_context_length"), # this is sort of undocumented in llama.cpp server
                 "cache_prompt" : True,
                 "n_predict": self.options["max_length"]}

            if self.hasImages():
                d["image_data"] = [packageImageDataLlamacpp(d["data"], id) for (id, d) in self.images.items()]
        else:
            printerr("error: backend not recognized.")
            d = {}

            
        for paramname in DEFAULT_PARAMS.keys():
            d[paramname] = self.options[paramname]
        return d
            
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

        # this is crazy
        self.tts.stdin.flush()
        self.tts.stdout.flush()
        self.tts.stderr.flush()        
        self.tts.stdin.write(w + "\n")
        self.tts.stdin.flush()
        self.tts.stdout.flush()
        self.tts.stderr.flush()
        return w

    def print(self, w, end="\n", flush=False):
        # either prints, speaks, or both, depending on settings
        if self.getOption("tts"):
            self.communicateTTS(self.getTTSFormatter().format(w) + end)
            if not(self.getOption("tts_subtitles")):
                return

        print(self.getDisplayFormatter().format(w), end=end, flush=flush)

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
            setOption(prog, ["continue", "False"])
            return ("", "")

        if prog.continue_with != "":# user input has been replaced with something else, e.g. a transcription
            w = prog.popContinueString()


        w = prog.session.expandVars(w)
        (w, ai_hint) = prog.adjustForChat(w)
        
        return (w, ai_hint)

    def getSystemTokenCount(self):
        """Returns the number of tokens in system msg. The value is cached per session. Note that this adds +1 for the BOS token."""
        if self._systemTokenCount is None:
            self._systemTokenCount = len(self.tokenize(self.session.getSystem())) + 1
        return self._systemTokenCount


    def getTemplate(self):
        return self.template
    
    
    def showSystem(self):
        return self.getTemplate().header(**self.session.getVars())

    def showStory(self, story_folder=None, append_hint=True):
        """Returns the current story as a unformatted string, injecting the current prompt template.""" 
        if story_folder is None:
            sf = self.session.stories
        else:
            sf = story_folder
        return self.getTemplate().body(sf.get(), append_hint, **self.session.getVars())

    def formatStory(self, story_folder=None):
        """Pretty print the current story (or a provided one) in a nicely formatted way. Returns pretty story as a string."""
        if story_folder is None:
            sf = self.session.stories
        else:
            sf = story_folder
        return "\n".join([self.getDisplayFormatter().format(item["content"]) for item in sf.get().getData()])

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
        w = self.showStory() + hint 
        gamma = self.getOption("max_context_length")        
        k = self.getSystemTokenCount()
        wc = len(self.tokenize(w))
        n = self.getOption("max_length")
        budget = gamma - (k + wc + n)
        
        if budget < 0:
            #shit the bed
            return self.showSystem() + w

        # now we need a smart story history that fits into budget
        sf = self.session.stories.copyFolder(only_active=True)
        while len(self.tokenize(self.showStory(story_folder=sf) + hint)) > budget and not(sf.empty()):
        # drop some items from the story, smartly, and without changing original
            self._smartShifted = True
            item = sf.get().pop(0)
            #FIXME: this can be way better, needs moretesting!
        return self.showSystem() + self.showStory(story_folder=sf) + hint
        
    def adjustForChat(self, w):
        """Takes user input w and returns a pair (w1, v) where w1 is the modified user input and v is the beginning of the AI message. Both of these carry adjustments for chat mode, such as adding 'Bob: ' and similar. This has no effect if there is no chat mode set, and v is empty string in this case."""
        v = ""
        if self.getMode() == "chat":
            w = mkChatPrompt(self.getOption("chat_user")) + w
            v = mkChatPrompt(self.getOption("chat_ai"), space=False)
        elif self.getMode() == "chat-thoughts":
            w = mkChatPrompt(self.getOption("chat_user")) + w
            # have to start with Internal Thought: etc, FIXME: refactor
            v = mkChatPrompt(self.session.getVar("thoughtstr"), space=False)
        return (w, v)
    
    def communicate(prog, prompt_text):
        """Sends prompt_text to the backend and prints results."""
        #turn raw text into json for the backend
        prompt = prog.getPrompt(prompt_text)
        if prog.getOption("streaming"):
            r = streamPrompt(prog, prog.getOption("endpoint") + "/api/extra/generate/stream", json=prompt)
            #FIXME: find a better way to do this (print of ai prompt with colon etc.)
            if prog.getOption("chat_show_ai_prompt"):
                print(mkChatPrompt(prog.getOption("chat_ai")), end="")
            prog.streaming_done.wait()
            prog.streaming_done.clear()
            r.json = lambda ws=prog.stream_queue: {"results" : [{"text" : "".join(ws)}]}
            prog.stream_queue = []
        else:
            r = requests.post(prog.getOption("endpoint") + prog.getGenerateApi(), json=prompt)

        if r.status_code == 200:
            results = prog.getResults(r)
            if prog.getOption("streaming"):
                # already printed it piecemeal, so skip this step
                return results
            else:
                # FIXME: we're currently formatting the AI string twice. Here and in addAIText. that's not a big deal, though
                prog.print(prog.getAIFormatter().format(results), end="")
                return results
        else:
            printerr(str(r.status_code))
            return ""

    def getGenerateApi(self):
        backend = self.getOption("backend")
        if backend == "llama.cpp":
            return "/completion"
        else:
            return "/api/v1/generate"


    def getHealthEndpoint(self):
        backend = self.getOption("backend")
        if backend == "llama.cpp":
            return self.getOption("endpoint") + "/health"
        else:
            return self.getOption("endpoint") + "/api/v1/health"

    def getHealthResult(self):
        r = requests.get(self.getHealthEndpoint())
        if r.status_code == 200:
            return r.json()["status"]
        return "error " + str(r.status_code)
        
    def getResults(self, r):
        backend = self.getOption("backend")
        self.setLastResult(r.json())        
        if backend == "llama.cpp":
            return r.json()['content']
        else:
            return r.json()['results'][0]["text"]

    def hasImages(self):
        return bool(self.images)
        
    def loadImage(self, url, id):
        url = os.path.expanduser(url)
        if not(os.path.isfile(url)):
            printerr("warning: Could not load image '" + url + "'. File not found.")
            return

        self.images[id] = {"url" : url,
                           "data" : loadImageData(url)}

    def setLastResult(self, result):
        self.lastResult = result
        if self.getOption("backend") == "llama.cpp":
            # llama does not allow to set the context size by clients, instead it dictates it server side. however i have not found a way to query it directly, it just gets set after the first request
            self.setOption("max_context_length", result["generation_settings"]["n_ctx"])
            self._dirtyContextLlama = True # context has been set by llama
            
            
        

    def tokenize(self, w):
        # returns list of tokens
        if self.getOption("backend") != "llama.cpp":
            # this doesn't work
            return []

        r = requests.post(self.getOption("endpoint") + "/tokenize", json= {"content" : w})
        if r.status_code == 200:
            return r.json()["tokens"]
        return []
    
def main():
    parser = makeArgParser(DEFAULT_PARAMS)
    args = parser.parse_args()
    prog = Program(options=args.__dict__, initial_cli_prompt=args.cli_prompt)
    if userConfigFile():    
        prog.setOption("user_config", userConfigFile())
        printerr(loadConfig(prog, [userConfigFile()]))
    
    if prog.getOption("config_file"):
        printerr(loadConfig(prog, [prog.options["config_file"]]))
    
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

if __name__ == "__main__":
    main()
