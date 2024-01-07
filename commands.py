from session import Session
from kbcli_util import *
def newSession(program, argv):
    if argv == []:
        return "No path provided. Cannot Start."

    filepath = argv[0]    
    program.session = Session(dir=filepath, chat_user=program.chat_user)
    w = "Ok. Loaded " + filepath + "\n\n"
    w += program.session.initial_prompt
    return w


def printStory(prog, argv, stderr=False):
    w = prog.session.getStory()

    if stderr:
        printerr(w)
        return ""
    else:
        return w
    
def doContinue(prog, argv):
    setOption(prog, ["continue", "1"])

    # now comes some fuckery to get rid of trailing <|im_end|> etc.
    delim = prog.session.template_end
    if prog.session.story[-1].endswith(delim):
        prog.session.story[-1] = prog.session.story[-1][:-len(delim)]
    
    return ""

def setOption(prog, argv):
    if argv == []:
        return ""

    if len(argv) == 1:
        prog.options[argv[0]] = ""

    prog.options[argv[0]] = " ".join(argv[1:])
    return ""
    
def showOptions(prog, argv):
    w = ""
    for (k, v) in prog.options.items():
        w += k + ": " + v
    return w
