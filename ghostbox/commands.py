import os, datetime, glob, sys
from ghostbox.session import Session
from ghostbox.util import *
from ghostbox.StoryFolder import *

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
        w += loadConfig(program, [configpath], override=False) + "\n"
    program.options["character_folder"] = path
    w += "Ok. Loaded " + path

    # template_initial
    if "{$template_initial}" in program.session.keys:
        program.initial_print_flag = True
        
    # config may have set the mode, but we need to go through setMode
    program.setMode(program.getOption("mode"))

    # hide if option is set
    if program.getOption("hide"):
        hide(program, [])

    # init tts at this point, since config may have had tts vars
    if program.getOption("tts"):
        program.tts_flag = True
        
    return w


def printStory(prog, argv, stderr=False, apply_filter=True):
    w = prog.session.showStory(apply_filter=apply_filter)

    if stderr:
        printerr(w)
        return ""
    elif argv != []:
        filename = " ".join(argv)
        if os.path.isdir(filename):
            return "error: Cannot write to file " + filename + ": is a directory." 
        if os.path.isfile(filename):
            printerr("warning: File " + filename + " exists. Appending to file.")
            pre = open(filename, "r").read() + "\n--- new story ---"
        else:
            pre = ""
        f = open(filename, "w")
        f.write(pre + w)
        return ""
    else:
        return w
    
def doContinue(prog, argv):
    setOption(prog, ["continue", "1"])

    # now comes some fuckery to get rid of trailing <|im_end|> etc.
    delim = prog.session.template_end
    if delim == "":
        return ""
    
    if prog.session.stories.getStory()[-1].endswith(delim):
        prog.session.stories.getStory()[-1] = prog.session.stories.getStory()[-1][:-len(delim)]
    
    return ""

def setOption(prog, argv):
    if argv == []:
        return showOptions(prog, [])
    name = argv[0]    
    if len(argv) == 1:
        prog.setOption(name, "")
        return ""

    w = " ".join(argv[1:])
    try:
        prog.setOption(name, eval(w))
    except:
        return "Couldn't set " + name + " to " + w + ". Couldn't evaluate."
    return ""
    
def showOptions(prog, argv):
    w = ""
    for (k, v) in prog.options.items():
        w += k + ": " + str(v) + "\n"
    return w

def showVars(prog, argv):
    if argv == []:
        return "\n".join([f"{key}" for key in prog.session.keys.keys()])

    k = " ".join(argv)
    return prog.session.keys.get(k, f"Could not find var '{k}'")

def showChars(prog, argv):
    allchars = []
    for dir in prog.getOption("include"):
        if not(os.path.isdir(dir)):
            printerr("warning: Include path '" + dir + "' is not a directory.")
            continue
        for charpath in glob.glob(dir + "/*"):
            if os.path.isfile(charpath):
                continue
            allchars.append(os.path.split(charpath)[1])

    return "\n".join(sorted(allchars))


def showVoices(prog, argv):
    pollyvoices = "Lotte, Maxim, Ayanda, Salli, Ola, Arthur, Ida, Tomoko, Remi, Geraint, Miguel, Elin, Lisa, Giorgio, Marlene, Ines, Kajal, Zhiyu, Zeina, Suvi, Karl, Gwyneth, Joanna, Lucia, Cristiano, Astrid, Andres, Vicki, Mia, Vitoria, Bianca, Chantal, Raveena, Daniel, Amy, Liam, Ruth, Kevin, Brian, Russell, Aria, Matthew, Aditi, Zayd, Dora, Enrique, Hans, Danielle, Hiujin, Carmen, Sofie, Gregory, Ivy, Ewa, Maja, Gabrielle, Nicole, Filiz, Camila, Jacek, Thiago, Justin, Celine, Kazuha, Kendra, Arlet, Ricardo, Mads, Hannah, Mathieu, Lea, Sergio, Hala, Tatyana, Penelope, Naja, Olivia, Ruben, Laura, Takumi, Mizuki, Carla, Conchita, Jan, Kimberly, Liv, Adriano, Lupe, Joey, Pedro, Seoyeon, Emma, Niamh, Stephen".split(", ")
    w = ""
    if prog.getOption("tts_program") == "ghostbox-tts-polly":
        for voice in pollyvoices:
            w += voice + "\n"
    else:
        for file in glob.glob(prog.getOption("tts_voice_dir") + "/*"):
            if os.path.isfile(file):
                w += os.path.split(file)[1] + "\n"
                
    return w
        

def toggleMode(prog, argv):
    if argv == []:
        return "Currently in " + prog.getOption("mode") + " mode."

    mode = " ".join(argv)
    if prog.isValidMode(mode):
        prog.setMode(mode)
        return mode + " mode on"
    return "Not a valid mode. Possible values are 'default' or 'chat'"

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
    while False:
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

def loadConfig(prog, argv, override=True):
    # loads a config file, which may be a user config, or a config supplied by a character folder. if override is false, the config may not override command line arguments that have been manually supplied (in the long form)
    if argv == []:
        return "Please provide a filename for the config file to load."
    filename = " ".join(argv)
    try:
        w = open(filename, "r").read()
    except Exception as e:
        return str(e)
    err = prog.loadConfig(w, override=override)
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

def exitProgram(prog, argv):
    prog.running = False
    return ""
