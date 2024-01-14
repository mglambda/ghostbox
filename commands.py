import os, datetime
from session import Session
from kbcli_util import *
from StoryFolder import *

def newSession(program, argv):
    if argv == []:
        if program.getOption("character_folder"):
            # synonymous with /restart
            argv.append(program.getOption("character_folder"))
        else:
            return "No path provided. Cannot Start."

    filepath = " ".join(argv)
    allpaths = [filepath] + [p + "/" + filepath for p in program.getOption("include")]
    for path in allpaths:
        path = os.path.normpath(path)
        failure = False
        try:
            program.session = Session(dir=path, chat_user=program.getOption("chat_user"), additional_keys=program.getOption("var_file"))
            break
        except FileNotFoundError as e:
            # session will throw if path is not valid, we keep going through the includes
            failure = e

    if failure:
        return "error: " + str(failure)

    w = ""
    # try to load config.json if present
    configpath = path + "/config.json"
    if os.path.isfile(configpath):
        w += loadConfig(program, [configpath]) + "\n"
    program.options["character_folder"] = path
    w += "Ok. Loaded " + path
    return w


def printStory(prog, argv, stderr=False):
    w = prog.session.showStory()

    if stderr:
        printerr(w)
        return ""
    else:
        return w
    
def doContinue(prog, argv):
    setOption(prog, ["continue", "1"])

    # now comes some fuckery to get rid of trailing <|im_end|> etc.
    delim = prog.session.template_end
    if prog.session.stories.getStory()[-1].endswith(delim):
        prog.session.stories.getStory()[-1] = prog.session.stories.getStory()[-1][:-len(delim)]
    
    return ""

def setOption(prog, argv):
    if argv == []:
        return showOptions(prog, [])
    name = argv[0]    
    if len(argv) == 1:
        prog.options[name] = ""
        return ""

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
        prog.options["cli_prompt"] = "\n" + mkChatPrompt(name)
        prog.mode = "chat"
        prog.options["chat_user"] = name
        return "Chat mode on."

    #disable chat mode
    prog.options["chat_user"] = ""
    prog.mode = "default"
    prog.options["cli_prompt"] = prog.initial_cli_prompt
    return "Chat mode off."

def toggleTTS(prog, argv):
    prog.options["tts"] = not(prog.options["tts"])
    w = ""
    if prog.options["tts"]:
        err = prog.initializeTTS()
        if err:
            return err
        if prog.getOption("streaming"):
            prog.options["streaming"] = False
            w += "Disabling streaming (this tends to work better with TTS. You can manually reenable streaming if you wish.)\n"
        w += "Try /hide for a more immersive experience.\n"
            
    return w + "TTS " + {True : "on.", False : "off."}[prog.options["tts"]]

def ttsDebug(prog, argv):
    if not(prog.tts):
        return ""

    prog.tts.stdout.flush()
    while True:
        w = prog.tts.stdout.readline()
        if w:
            printerr(w)
        else:
            break


    prog.tts.stderr.flush()
    while True:
        w = prog.tts.stderr.readline()
        if w:
            printerr(w)
        else:
            break
        
    return ""
    
def nextStory(prog, argv):
    r = prog.session.stories.nextStory()
    if r == 1:
        return "Cannot go to next story branch: No more branches. Create a new branch with /new or /retry."
    return "Now on branch " + str(prog.session.stories.index)

def previousStory(prog, argv):
    r = prog.session.stories.previousStory()
    if r == -1:
        return "Cannot go to previous story branch: No previous branch exists."
    return "Now on branch " + str(prog.session.stories.index)    

def retry(prog, argv):
    prog.session.stories.cloneStory()
    prog.session.stories.dropEntry()
    doContinue(prog, [])
    return "Now on branch " + str(prog.session.stories.index) 

def dropEntry(prog, argv):
    prog.session.stories.cloneStory()
    prog.session.stories.dropEntry()
    return "Now on branch " + str(prog.session.stories.index) + " with last entry dropped."

    
def newStory(prog, argv):
    prog.session.stories.newStory()
    return "Now on branch " + str(prog.session.stories.index) + " with a clean log."
    
def saveStoryFolder(prog, argv):
    if len(argv) > 0:
        name = " ".join(argv)
    else:
        name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    if not(name.endswith(".json")):
        name = name + ".json"
            
    filename = saveFile(name, prog.session.stories.toJSON())
    if not(filename):
        return "Could not save story folder. Maybe provide a different filename?"
    return "Saved story folder as " + filename

    
def loadStoryFolder(prog, argv):
    if len(argv) == 0:
        return "Please provide a legal json filename in the story-folder format."
    
    filename = " ".join(argv)
    try:
        w = open(filename, "r").read()
        s = StoryFolder(json_data=w)
    except FileNotFoundError as e:
        return "Could not load " + filename + ": file not found."
    except Exception as e:
        printerr(str(e))
        return "Could not load story folder: Maybe bogus JSON?"
    prog.session.stories = s
    return "Ok. Restored " + filename + "\nNow on branch " + str(prog.session.stories.index)
        
def gotoStory(prog, argv):
    if argv == []:
        return "Currently on branch " + str(prog.session.stories.index)

    w = " ".join(argv)
    try:
        n = int(w)
    except:
        return "Cannot go to that branch: Invalid Argument."
    
    err = prog.session.stories.shiftTo(n)
    if not(err):
        return ""
    return "Could not go to branch " + str(n) + ": " + err

def loadConfig(prog, argv):
    if argv == []:
        return "Please provide a filename for the config file to load."
    filename = " ".join(argv)
    try:
        w = open(filename, "r").read()
    except Exception as e:
        return str(e)
    err = prog.loadConfig(w)
    if err:
        return err
    return "Loaded config " + filename

def saveConfig(prog, argv):
    if argv == []:
        name = "config-" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    else:
        name = " ".join(argv)

    if not(name.endswith(".json")):
        name = name + ".json"

    filename = saveFile(name, json.dumps(prog.options, indent=4))
    if filename:
        return "Saved config to " + filename
    return "error: Could not save config."

    

def hide(prog, argv):
    # this is just convenient shorthand for when I want my screen reader to be less spammy
    prog.options["cli_prompt"] = ""
    prog.options["tts_subtitles"] = False
    prog.options["streaming"] = False
    prog.options["chat_show_ai_prompt"] = False
    return ""

def varfile(prog, argv):
    # $FIXME:
    return "Not implemented yet. Use the -V option for now."
