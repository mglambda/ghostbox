#!/bin/python
import requests, json, os, io, re, base64, random, sys, threading, subprocess
import argparse
from commands import *
from kbcli_util import *
from kbcli_streaming import streamPrompt
from session import Session

# these can be typed in at the CLI prompt
cmds = [
    ("/start", newSession),
    ("/restart", lambda prog, argv: newSession(prog, [])),
    ("/print", printStory) ,
    ("/next", nextStory),
    ("/prev", previousStory),
    ("/story", gotoStory),
    ("/load", loadStoryFolder),
    ("/save", saveStoryFolder),
    ("/retry", retry),
    ("/drop", dropEntry),
    ("/new", newStory),
    ("/log", lambda prog, w: printStory(prog, w, stderr=True)),
    ("/ttsdebug", ttsDebug),    
    ("/tts", toggleTTS),
    ("/set", setOption),
    ("/unset", lambda prog, argv: setOption(prog, [argv[0]])),
    ("/lsoptions", showOptions),
    ("/chatmode", toggleChatMode),
    ("/cont", doContinue),
        ("/continue", doContinue)
]

#defined here for convenience. Can all be changed through cli parameters or with / commands
DEFAULT_PARAMS = {                "rep_pen": 1.1, "temperature": 0.7, "top_p": 0.92, "top_k": 0, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 320, "rep_pen_slope": 0.7, "sampler_order": [6, 0, 1, 3, 4, 2, 5], "quiet": True, "use_default_badwordsids": True}

class Program(object):
    def __init__(self, options={}, initial_cli_prompt=""):
        self.session = Session(chat_user=options.get("chat_user", ""))
        self.initial_cli_prompt = initial_cli_prompt
        self.mode = "default"
        self.streaming_done = threading.Event()
        self.stream_queue = []
        self.tts = None
        self.options = options
        if self.getOption("chat_user"):
            toggleChatMode(self, [self.getOption("chat_user")])
        if self.getOption("tts"):
            printerr(self.initializeTTS())
            
    def getMode(self):
        return self.mode
        
    def getOption(self, key):
        return self.options.get(key, False)
        
    def getPrompt(self, conversation_history, text, system_msg = ""): # For KoboldAI Generation
        d = {"prompt": conversation_history + text + "",
                "memory" : system_msg, # koboldcpp special feature: will prepend this to the prompt, overwriting prompt history if necessary
                "n": 1,
                "max_context_length": 2048, #1024,
                "max_length": self.options["max_length"]}
        for paramname in DEFAULT_PARAMS.keys():
            d[paramname] = self.options[paramname]
        return d
            
    #        "max_context_length": 1024, "max_length": 256, "n": 1, "rep_pen": 1.8, "rep_pen_range": 2048, "rep_pen_slope": 0.7, "temperature": 0.7, "tfs": 1, "top_a": 0, "top_k": 0, "top_p": 0.9, "typical": 1, "sampler_order": [6, 0, 1, 3, 4, 2, 5], "singleline": False, "sampler_seed": 69420, "sampler_full_determinism": False, "frmttriminc": False, "frmtrmblln": False}    

    def initializeTTS(self):
        tts_program = self.getOption("tts_program")
        if not(tts_program):
            return "Cannot initialize TTS: No TTS program set."

        self.tts = subprocess.Popen([os.getcwd() + "/" + tts_program], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
        return ""

    def communicateTTS(self, w):
        if not(self.getOption("tts")):
            return ""

        self.tts.stdin.write(w)
        self.tts.stdin.flush()
        return w

    def print(self, w, end="\n", flush=False):
        # either prints, speaks, or both, depending on settings
        if self.getOption("tts"):
            self.communicateTTS(w + end)
            if not(self.getOption("tts_subtitles")):
                return

        print(w, end=end, flush=flush)

        
    
def main():
    parser = argparse.ArgumentParser(description="kbcli - koboldcpp Command Line Interface", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-I", '--include', action="append", default=["chars/"], help="Include paths that will be searched for character folders named with the /start command or the --character_folder command line argument.")
    parser.add_argument("-c", '--character_folder', type=str, default="", help="character folder to load at startup. The folder may contain templates, as well as arbitrary text files that may be injected in the templates. See the examples for more. Path is attempted to be resolved relative to the include paths, if any are provided.")    
    parser.add_argument("--endpoint", type=str, default="http://localhost:5001", help="Address of koboldcpp http endpoint.")
    parser.add_argument("--max_length", type=int, default=300, help="Number of tokens to request from koboldcpp for generation.")
    parser.add_argument("--chat_user", type=str, default="", help="Username you wish to be called when chatting. Setting this automatically enables chat mode. It will also replace occurrences of {chat_user} anywhere in the character files.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Enable streaming mode.")
    parser.add_argument("--streaming_flush", action=argparse.BooleanOptionalAction, default=False, help="When True, flush print buffer immediately in streaming mode (print token-by-token). When set to false, waits for newline until generated text is printed.")
    parser.add_argument("--cli_prompt", type=str, default="\n 🧠 ", help="String to show at the bottom as command prompt. Can be empty.")
    parser.add_argument("--tts", action=argparse.BooleanOptionalAction, default=False, help="Enable text to speech on generated text.")
    parser.add_argument("--tts_program", type=str, default="tts.sh", help="Path to a TTS (Text-to-speech) program to verbalize generated text. The TTS program should read lines from standard input.")
    parser.add_argument("--tts_subtitles", action=argparse.BooleanOptionalAction, default=False, help="Enable printing of generated text while TTS is enabled.") 

    
    for (param, value) in DEFAULT_PARAMS.items():
        parser.add_argument("--" + param, type=type(value), default=value, help="Passed on to koboldcpp. Change during runtime with /set " + param + ".")
    
    args = parser.parse_args()
    prog = Program(options=args.__dict__, initial_cli_prompt=args.cli_prompt)
    if args.character_folder:
        printerr(        newSession(prog, []))
    skip = False
    
    while True:
        w = input(prog.getOption("cli_prompt"))
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
        elif prog.session.hasTemplate():
            w = prog.session.injectTemplate(w)
            prompt = prog.getPrompt(prog.session.showStory(), w, system_msg = prog.session.getSystem())
        else:
            prompt = prog.getPrompt(prog.session.memory + prog.session.getNote() + prog.session.showStory() , w + prog.session.prompt)

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
            displaytxt = filterLonelyPrompt(prog.session.prompt, trimIncompleteSentence(trimChatUser(prog.getOption("chat_user"), results[0]["text"])))
            txt = prog.session.prompt + displaytxt + prog.session.template_end
            prog.session.addText(w)
            prog.session.addText(txt)

            if prog.getOption("streaming"):
                # already printed it piecemeal, so skip this step
                continue
            else:
                prog.print(displaytxt)
        else:
            print(str(r.status_code))
        
if __name__ == "__main__":
    main()

