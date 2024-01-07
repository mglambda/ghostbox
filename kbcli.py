#!/bin/python
import requests, json, os, io, re, base64, random, sys, threading
from commands import *
from kbcli_util import *
from kbcli_streaming import streamPrompt
from session import Session

argw = " ".join(sys.argv)
w = getArgument("--max_tokens", argw)
if w:
    MAX_TOKENS = int(w)
else:
    MAX_TOKENS = 256

w = getArgument("--user", argw)
if w:
    CHAT_USER = w
else:
    CHAT_USER = ""
INPUT_DELIMITER = "\n\n" # inserted between user input and response, but only when tehre is a CHAT_USER
ENDPOINT = "http://localhost:5001"

cmds = [
    ("/start", newSession),
    ("/print", printStory) ,
    ("/log", lambda prog, w: printStory(prog, w, stderr=True)),
    ("/set", setOption),
    ("/unset", lambda prog, w: setOption(prog, "")),
    ("/lsoptions", showOptions),
    ("/cont", doContinue),
        ("/continue", doContinue)
]
    


def split_text(text):
    parts = re.split(r'\n[a-zA-Z]', text)
    return parts



def getPrompt(conversation_history, text, system_msg = ""): # For KoboldAI Generation
    return {"prompt": conversation_history + text + "",
            "memory" : system_msg, # koboldcpp special feature: will prepend this to the prompt, overwriting prompt history if necessary
            "n": 1,
            "max_context_length": 2048, #1024,
            "max_length": MAX_TOKENS,
            "rep_pen": 1.1, "temperature": 0.7, "top_p": 0.92, "top_k": 0, "top_a": 0, "typical": 1, "tfs": 1, "rep_pen_range": 320, "rep_pen_slope": 0.7, "sampler_order": [6, 0, 1, 3, 4, 2, 5], "quiet": True, "use_default_badwordsids": True}            
#        "max_context_length": 1024, "max_length": 256, "n": 1, "rep_pen": 1.8, "rep_pen_range": 2048, "rep_pen_slope": 0.7, "temperature": 0.7, "tfs": 1, "top_a": 0, "top_k": 0, "top_p": 0.9, "typical": 1, "sampler_order": [6, 0, 1, 3, 4, 2, 5], "singleline": False, "sampler_seed": 69420, "sampler_full_determinism": False, "frmttriminc": False, "frmtrmblln": False}    


def getStory():
    url = ENDPOINT + "/API/v1/story"
    print(url)
    return requests.get(url)

class Program(object):
    def __init__(self, chat_user=""):
        self.chat_user = chat_user
        self.session = Session(chat_user=chat_user)
        self.streaming_done = threading.Event()
        self.stream_queue = []
        self.options = {"streaming" : "True"}

    def getOption(self, key):
        return self.options.get(key, False)
        

    
prog = Program(chat_user=CHAT_USER)
skip = False
while True:
    w = input()
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
    if CHAT_USER:
        if w == "":
            w = INPUT_DELIMITER
        else:
            w = INPUT_DELIMITER + CHAT_USER + ": " + w + INPUT_DELIMITER

    if prog.getOption("continue"):
        setOption(prog, ["continue", ""])
        prompt = getPrompt(prog.session.getStory(trim_end=True), "", system_msg = prog.session.template_system)                
    elif prog.session.hasTemplate():
        w = prog.session.injectTemplate(w)
        prompt = getPrompt(prog.session.getStory(), w, system_msg = prog.session.template_system)
    else:
        prompt = getPrompt(prog.session.memory + prog.session.getNote() + prog.session.getStory() , w + prog.session.prompt)



    if prog.getOption("streaming"):
        r = streamPrompt(prog, ENDPOINT + "/api/extra/generate/stream", json=prompt)
        prog.streaming_done.wait()
        prog.streaming_done.clear()
        r.json = lambda ws=prog.stream_queue: {"results" : [{"text" : "".join(ws)}]}
        prog.stream_queue = []
    else:
        r = requests.post(ENDPOINT + "/api/v1/generate", json=prompt)
    if r.status_code == 200:
        results = r.json()['results']
        displaytxt = filterLonelyPrompt(prog.session.prompt, trimIncompleteSentence(trimChatUser(CHAT_USER, results[0]["text"])))
        txt = prog.session.prompt + displaytxt + prog.session.template_end
        prog.session.addText(w + txt)

        if prog.getOption("streaming"):
            # already printed it piecemeal, so skip this step
            continue
        elif CHAT_USER:
            # we filter again because we don't want prompts in the spoken dialogue for chats, though it must be in the saved story if it was generated
            displaytxt = filterPrompt(prog.session.prompt, displaytxt)
            print(displaytxt)
        else:
            print(displaytxt)
    else:
        print(str(r.status_code))
    
