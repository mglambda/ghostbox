from session import Session
from kbcli_util import *
def newSession(program, argv):
    if argv == []:
        return "No path provided. Cannot Start."

    filepath = argv[0]    
    program.session = Session(dir=filepath, chat_user=program.getOption("chat_user"))
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
    name = argv[0]
    if argv == []:
        return ""

    if len(argv) == 1:
        prog.options[name] = ""

    w = " ".join(argv[1:])
    try:
        prog.options[name] = eval(w)
    except:
        return "Couldn't set " + name + " to " + w + ". Couldn't evaluate."
    return ""
    
def showOptions(prog, argv):
    w = ""
    for (k, v) in prog.options.items():
        w += k + ": " + str(v) + "\n"
    return w

def toggleChatMode(prog, argv):
    # when invoked as '/chatmode' toggles off, when invoked as '/chatmode Anna', enables chat mode and sets username to Anna
    if len(argv) > 0:
        name = " ".join(argv)
        prog.options["cli_prompt"] = ""
        prog.mode = "chat"
        prog.options["chat_user"] = name
        return "Chat mode on."

    #disable chat mode
    prog.options["chat_user"] = ""
    prog.mode = "default"
    prog.options["cli_prompt"] = prog.initial_cli_prompt
    return "Chat mode off."
             
        

    
