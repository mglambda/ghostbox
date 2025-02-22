import subprocess, tempfile
import datetime
import os
#from TTS.tts.layers.xtts.tokenizer import split_sentence

def getAccumulatorFile(filepath=""):
    if filepath:
        f = open(filepath, "w")
    else:
        f = tempfile.NamedTemporaryFile(suffix=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".wav", delete=False)
    return f

def maybeGetTag(w):
    # returns pair of (tag, tagargs) as (string,. list of strings)
    delim = "#"
    if w.startswith(delim):
        ws = w.split(" ")
        tag = ws[0][len(delim):]
        args = ws[1:]
        return (tag, args)
    return ("", [])
        


    



def dir2(x):
    for k in dir(x):
        if k.startswith("_"):
            continue
        print(str(k))


def fixDivZero(w):
    # so, weirdly, the TTS will crash when it encounters xml tags like <test> or <begin> or really anythingin <> brackets. (division by zero crash)
    # easy fix is just to replace the symbols with spelled out words, since at this point we only care about the spoken part anyway. Fucks up languages other than english, but oh well
    return w.replace("<", " less than ").replace(">", " greater than ")
