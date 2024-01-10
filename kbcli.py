#!/bin/python
import requests, json, os, io, re, base64, random, sys, threading
import argparse
from commands import *
from kbcli_util import *
from kbcli_streaming import streamPrompt
from session import Session

# these can be typed in at the CLI prompt
cmds = [
    ("/start", newSession),
    ("/print", printStory) ,
    ("/log", lambda prog, w: printStory(prog, w, stderr=True)),
    ("/set", setOption),
    ("/unset", lambda prog, w: setOption(prog, "")),
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
        self.options = options
        if self.getOption("chat_user"):
            toggleChatMode(self, [self.getOption("chat_user")])
        
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





def main():
    parser = argparse.ArgumentParser(description="kbcli - koboldcpp Command Line Interface", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--character_folder", type=str, default="", help="character folder to load at startup. The folder may contain templates, as well as arbitrary text files that may be injected in the templates. See the examples for more.")
    parser.add_argument("--endpoint", type=str, default="http://localhost:5001", help="Address of koboldcpp http endpoint.")
    parser.add_argument("--max_length", type=int, default=300, help="Number of tokens to request from koboldcpp for generation.")
    parser.add_argument("--chat_user", type=str, default="", help="Username you wish to be called when chatting. Setting this automatically enables chat mode. It will also replace occurrences of {chat_user} anywhere in the character files.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Enable streaming mode.")
    parser.add_argument("--streaming_flush", action=argparse.BooleanOptionalAction, default=False, help="When True, flush print buffer immediately in streaming mode (print token-by-token). When set to false, waits for newline until generated text is printed.")
    parser.add_argument("--cli_prompt", type=str, default="\n 🧠 ", help="String to show at the bottom as command prompt. Can be empty.")
    for (param, value) in DEFAULT_PARAMS.items():
        parser.add_argument("--" + param, type=type(value), default=value, help="Passed on to koboldcpp. Change during runtime with /set " + param + ".")
    
    args = parser.parse_args()
    CHAT_USER = ""
    prog = Program(options=args.__dict__, initial_cli_prompt=args.cli_prompt)
    if args.character_folder:
        printerr(        newSession(prog, []))
    skip = False
    
    while True:
        w = input(prog.getOption("cli_prompt"))
        if prog.getMode() == "chat":
            # for convenience when chatting
            if w == "":
                w = "/cont"
            
            
        for (cmd, f) in cmds:
            if w.startswith(cmd):
                v = f(prog, w.split(" ")[1:])
                printerr(v)
                if not(prog.getOption("continue")):
                    # skip means we don't send a prompt this iteration, which we don't want to do when user issues a command, except for the /continue command
                    skip = True
            
        if skip:
            skip = False
            continue

        if prog.getOption("continue"):
            setOption(prog, ["continue", "False"])
            w = "" # get rid of /cont etc
            prompt = prog.getPrompt(prog.session.getStory(trim_end=True), "", system_msg = prog.session.getSystem())                
        elif prog.session.hasTemplate():
            w = prog.session.injectTemplate(w)
            prompt = prog.getPrompt(prog.session.getStory(), w, system_msg = prog.session.getSystem())
        else:
            prompt = prog.getPrompt(prog.session.memory + prog.session.getNote() + prog.session.getStory() , w + prog.session.prompt)

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
            prog.session.addText(w + txt)

            if prog.getOption("streaming"):
                # already printed it piecemeal, so skip this step
                continue
            else:
                print(displaytxt)
        else:
            print(str(r.status_code))
        
if __name__ == "__main__":
    main()

