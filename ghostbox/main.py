import requests, json, os, io, re, base64, random, sys, threading, subprocess, signal
import argparse
from ghostbox.commands import *
from ghostbox.util import *
from ghostbox._argparse import *
from ghostbox.streaming import streamPrompt
from ghostbox.session import Session


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

    
#defined here for convenience. Can all be changed through cli parameters or with / commands
DEFAULT_PARAMS = {                "rep_pen": 1.1, "temperature": 0.7, "top_p": 0.92, "top_k": 0, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 320, "rep_pen_slope": 0.7, "sampler_order": [6, 0, 1, 3, 4, 2, 5], "quiet": True, "use_default_badwordsids": True}

class Program(object):
    def __init__(self, options={}, initial_cli_prompt=""):
        self.session = Session(chat_user=options.get("chat_user", ""))
        self.tts_flag = False
        self.initial_print_flag = False
        self.initial_cli_prompt = initial_cli_prompt
        self.streaming_done = threading.Event()
        self.stream_queue = []
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
        
        # formatters is to be idnexed with modes
        self._formatters = {
            "default" : self._defaultFormatter,
            "chat" : self._chatFormatter}
        self.setMode(self.getOption("mode"))
        self.running = True


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
                
        self.options = self.options | d
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
        return mode in self._formatters

    def setMode(self, mode):
        if not(self.isValidMode(mode)):
            return
        
        self.options["mode"] = mode
        if mode == "chat":
            self.options["cli_prompt"] = "\n" + mkChatPrompt(self.getOption("chat_user"))            
        else: # default
            self.options["cli_prompt"] = self.initial_cli_prompt
               
    def getOption(self, key):
        return self.options.get(key, False)

    def setOption(self, name, value):
        self.options[name] = value
        # for some options we do extra stuff
        if name == "tts_voice" or name == "tts_volume":
            self.tts_flag = True #restart TTS
        
    
    def getPrompt(self, conversation_history, text, system_msg = ""): # For KoboldAI Generation
        d = {"prompt": conversation_history + text + "",
             "grammar" : self.getOption("grammar"),
             "memory" : system_msg, # koboldcpp special feature: will prepend this to the prompt, overwriting prompt history if necessary
             "n": 1,
             "max_context_length": self.getOption("max_context_length"),
             "max_length": self.options["max_length"]}
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
            self.communicateTTS(w + end)
            if not(self.getOption("tts_subtitles")):
                return

        print(w, end=end, flush=flush)

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
        
    
    def formatGeneratedText(self, w):
        return self._formatters.get(self.getMode(), self._defaultFormatter)(self.replaceForbidden(w))

    def _defaultFormatter(self, w):
        display =trimIncompleteSentence(w)
        raw = display + self.session.template_end
        return (display, raw)

    def _chatFormatter(self, w):
        # point of this is to 1. always start the prompt with e.g. Gary: if the AI is called gary. This helps the LLM stay in character, but it should be hidden from the user in chat mode
        # 2. filter out any accidental generations of Bob: if the user is called bob, i.e. don't let the LLM talk for the user.
        ai_chat_prompt = mkChatPrompt(self.getOption("chat_ai"))
        user_chat_prompt = mkChatPrompt(self.getOption("chat_user"))
        display = filterLonelyPrompt(ai_chat_prompt, trimIncompleteSentence(trimChatUser(self.getOption("chat_user"), w))).strip()
        txt = display + self.session.template_end
        if self.getOption("chat_show_ai_prompt"):
            display = ai_chat_prompt + display        
        return (display, txt)
    


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
    del prog.options["hide"]
    skip = False        
    while prog.running:
        # have to do TTS here for complex reasons; flag means to reinitialize tts, which can happen e.g. due to voice change
        if prog.tts_flag:
            prog.tts_flag = False            
            prog.options["tts"] = False
            printerr(toggleTTS(prog, []))

        if prog.initial_print_flag:
            prog.initial_print_flag = False
            print("\n\n" + prog.session.showStory(apply_filter=True), end="")
        
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

        if prog.getOption("continue"):
            setOption(prog, ["continue", "False"])
            w = "" # get rid of /cont etc
            prompt = prog.getPrompt(prog.session.showStory(trim_end=True), "", system_msg = prog.session.getSystem())                
        else:
            v = ""
            if prog.getMode() == "chat":
                w = mkChatPrompt(prog.getOption("chat_user")) + w
                v = mkChatPrompt(prog.getOption("chat_ai"))
            w = prog.session.injectTemplate(w) + v
            prompt = prog.getPrompt(prog.session.showStory(), w, system_msg = prog.session.getSystem())



        if prog.getOption("streaming"):
            r = streamPrompt(prog, prog.getOption("endpoint") + "/api/extra/generate/stream", json=prompt)
            prog.streaming_done.wait()
            prog.streaming_done.clear()
            r.json = lambda ws=prog.stream_queue: {"results" : [{"text" : "".join(ws)}]}
            prog.stream_queue = []
        else:
            r = requests.post(prog.getOption("endpoint") + "/api/v1/generate", json=prompt)

        if r.status_code == 200:
            results = r.json()['results']
            (displaytxt, txt) = prog.formatGeneratedText(results[0]["text"])
            prog.session.addUserText(w)
            prog.session.addAIText(txt)

            if prog.getOption("streaming"):
                # already printed it piecemeal, so skip this step
                continue
            else:
                prog.print(displaytxt, end="")
        else:
            print(str(r.status_code))
        
if __name__ == "__main__":
    main()


