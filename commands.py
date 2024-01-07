from session import Session
from kbcli_util import *
def newSession(program, argv):
    program.session = Session(dir=argv, chat_user=program.chat_user)
    w = "Ok. Loaded " + w + "\n\n"
    w += program.session.initial_prompt
    return w


def printStory(prog, argv, stderr=False):
    w = prog.session.getStory()

    if stderr:
        printerr(w)
        return ""
    else:
        return w
    
