import os, datetime, glob, sys
from ghostbox.session import Session
from ghostbox.util import *
from ghostbox.StoryFolder import *

def newSession(program, argv):
    """CHARACTER_FOLDER
    Start a new session with the character or template defined in CHARACTER_FOLDER. You can specify a full path, or just the folder name, in which case the program will look for it in all folders specified in the 'include' paths. See /lsoptions include."""
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
    """[FILENAME]
    Print the current story.
    If FILENAME is provided, save the story to that file.""" 
    w = prog.session.showStory(apply_filter=apply_filter)

    if stderr:
        printerr(w, prefix="")
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
        # if stderr=False, we actually want this to go to stdout
        print(w, end="")
        return ""
    
def doContinue(prog, argv):
    """
    Continue generating without new input. Use this whenever you want the AI to 'just keep talking'.
    Specifically, this will send the story up to this point verbatim to the LLM backend. You can check where you are with /log.
    If there are templated tokens that are normally inserted as part of the prompt format when a user message is received, they are not inserted.
    This command is also executed when you hit enter without any text."""

    prog.setOption("continue", "1")
    if prog.session.stories.empty():
        return ""
    
    # now comes some fuckery to get rid of trailing <|im_end|> etc.
    # FIXME: this doesn't work for other prompt templates than chat-ml
    delim = prog.session.template_end
    if delim == "":
        return ""
    
    if prog.session.stories.getStory()[-1]["text"].endswith(delim):
        prog.session.stories.getStory()[-1]["text"] = prog.session.stories.getStory()[-1]["text"][:-len(delim)]
    
    return ""

def setOption(prog, argv):
    """OPTION_NAME [OPTION_VALUE]
    Set options during program execution. To see a list of all possible values for OPTION_NAME, do /lsoptions. OPTION_VALUE must be a valid python expression, so if you want to set e.g. chat_user to Bob, do /set chat_user \"Bob\".
    When OPTION_VALUE is omitted, the value is set to \"\". This is equivalent to /unset OPTION_NAME."""

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
    """[OPTION_NAME]
    Displays the list of program options, along with their values. Provide OPTION_NAME to see just its value.
    The options displayed can all be set using /set OPTION_NAME.
    Almost all of them may also be provided as command line arguments with preceding dashes, e.g. include as --include=/some/path.
    Finally, they may be set in various config files, such as in character folders, or in a config file loaded with /load."""
    if argv == []:
        target = ""
    else:
        target = " ".join(argv)
        
    w = ""
    for (k, v) in prog.options.items():
        if target in k:
            w += k + ": " + str(v) + "\n"
    return w.strip()

def showVars(prog, argv):
    """[VARIABLE_NAME]
    Shows all program variables and their respective values. Show a single variable if VARIABLE_NAME is provided.
    Variables are automatically expanded within text that is provided by the user or generated by the LLM backend.
    E.g. if variable favorite_fruit is set to 'Tomatoe', any occurrence of {$favorite_fruit} in the text will be replaced by 'Tomatoe'.
    There are two ways to set variables:
      1. Loading a character folder (/start). All files in a character folder automatically become variables, with the respective filename becoming the name of the variable, and the file's content becoming the value of the variable. This is often used by character creators within their own templates.
      2. The -x or --varfile command line option. This argument provides additional files that may serve as variables, similar to (1). The argument may be repeated multiple times. This is most commonly used by users."""

    if argv == []:
        return "\n".join([f"{key}" for key in prog.session.keys.keys()])

    k = " ".join(argv)
    return prog.session.keys.get(k, f"Could not find var '{k}'")

def showChars(prog, argv):
    """
    Lists all available character folders. These can be used with the /start command or the -c command line argument.
    To see what places are searched for characters, see the value of the 'include' option."""
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
    """
    List all available voices for the TTS program. These can be used with /set tts_voice.
    To see the places searched for voices, see the value of tts_voice_dir.
    Depending on the value of tts_program, that location may be meaningless, as the voice won't be file based, like with amazon polly voices. In this case /lsvoices tries to give a helpful answer if it can."""
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
    """[MODE_NAME}
    Put the program into the specified mode, or show the current mode if MODE_NAME is omitted.
    Possible values are 'default', or 'chat'.
    Setting a certain mode has various effects on many aspects of program execution. Currently, most of this is undocumented :)"""    
    if argv == []:
        return "Currently in " + prog.getOption("mode") + " mode."

    mode = " ".join(argv)
    if prog.isValidMode(mode):
        prog.setMode(mode)
        return mode + " mode on"
    return "Not a valid mode. Possible values are 'default' or 'chat'"

def toggleTTS(prog, argv):
    """
    This turns the TTS (text-to-speech) module on or off. When TTS is on, text that is generated by the LLM backend will be spoken out loud.
    The TTS service that will be used depends on the value of tts_program. The tts_program can be any executable or shell script that reads lines from standard input. It may also support additional functionality.
    An example tts program for amazon polly voices is provided with 'ghostbox-tts-polly'. Note that this requires you to have credentials with amazon web services.
    On linux distributions with speech-dispatcher, you can set the value of tts_program to 'spd-say', or 'espeak-ng' if it's installed. This works, but it isn't very nice.
    The voice used depends on the value of tts_voice, which you can either /set or provide with the -V or --tts_voice command line option. It can also be set in character folder's config.json.
    ghostbox will attempt to provide the tts_program with the voice using a -V command line argument.
    To see the list of supported voices, try /lsvoices.
    Currently, turning TTS on disables token streaming, though you may reenable it with /set streaming True, if you wish."""
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
    """
    Get stdout and stderr from the underlying tts_program when TTS is enabled.
    This doesn't work right now and may hang the program."""
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
    """
    Go to next branch in story folder."""
    r = prog.session.stories.nextStory()
    if r == 1:
        return "Cannot go to next story branch: No more branches. Create a new branch with /new or /retry."
    return "Now on branch " + str(prog.session.stories.index)

def previousStory(prog, argv):
    """
    Go to previous branch in story folder."""
    r = prog.session.stories.previousStory()
    if r == -1:
        return "Cannot go to previous story branch: No previous branch exists."
    return "Now on branch " + str(prog.session.stories.index)    

def retry(prog, argv, predicate=lambda item: item["user_generated"] == True):
    """
    Retry generation of the LLM's response.
    This is in a family of commands consisting of /retry, /resend, and /rephrase. Specifically:
      /retry will drop the last generated response from the current story and generate it again. Use this in most cases where you want to regenerate.
      /resend will rewind the current story to the last user provided input (not /cont or newline), then regenerate a response from the backend. Use this if you hit enter a couple of times and let the AI talk, and now you want it to start again from your last question / instruction.
      /rephrase will rewind the current story to just before the user provided message, without regenerating. Use this if you want to discard the AI's response and you want to slightly rewrite what you said.
    Note that none of the above commands are destructive, i.e. they all create a new branch instead of modifying the current branch in the story folder."""

    prog.session.stories.cloneStory()
    prog.session.stories.dropEntriesUntil(predicate)
    doContinue(prog, [])
    printerr("Now on branch " + str(prog.session.stories.index) )
    return ""

def dropEntry(prog, argv):
    """
    Drops the last entry from the current story branch, regardless of wether it was user provided or generated by the LLM backend."""
    prog.session.stories.cloneStory()
    prog.session.stories.dropEntry()
    return "Now on branch " + str(prog.session.stories.index) + " with last entry dropped."

    
def newStory(prog, argv):
    """
    Create a new, empty branch in the story folder. You can always go back with /prev or /story."""
    prog.session.stories.newStory()
    return "Now on branch " + str(prog.session.stories.index) + " with a clean log."

def cloneStory(prog, argv):
    """
    Create a new branch in the story folder, copying the contents entirely from the current story."""
    prog.session.stories.cloneStory()
    return "Now on branch " + str(prog.session.stories.index) + " with a copy of the last branch."



def saveStoryFolder(prog, argv):
    """[STORY_FOLDER_NAME]
    Save the entire story folder in the file STORY_FOLDER_NAME. If no argument is provided, creates a file with the current timestampe as name.
    The file created is your accumulated chat history in its entirety, including the current story and all other ones (accessible with /prev and /next, etc). It is saved in the json format.
    A saved story folder may be loaded with /load."""
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
    """STORY_FOLDER_NAME
    Loads a previously saved story folder from file STORY_FOLDER_NAME. See the /save command on how to save story folders.
    A story folder is a json file containing the entire chat history."""
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
    """[BRANCH_NUMBER]
    Moves to a different branch in the story folder. If BRANCH_NUMBER is omitted, shows the current branch instead."""
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
    """CONFIG_FILE
    Loads a json config file at location CONFIG_FILE. A config file contains a dictionary of program options. 
    You can create an example config with /saveconfig example.conf.json.
    You can also load a config file at startup with the --config_file command line argument.
    If it exists, the ~/.ghostbox.conf.json will also be loaded at startup.
    The order of config file loading is as follows .ghostconf.conf.json > --config_file > conf.json (from character folder). Config files that are loaded later will override settings from earlier files."""

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
    """CONFIG_FILE
    Save the current program options and their values to the file at location CONFIG_FILE. This will either create or overwrite the CONFIG_FILE, deleting all its previous contents."""
    
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
    """
    Hide various program outputs and change some variables for a less verbose display.
    This does nothing that you cannot achieve by manually setting several options, it just bundles an eclectic mix of them in one command.
    I like to use this with TTS for a more immersive experience."""
    # this is just convenient shorthand for when I want my screen reader to be less spammy
    prog.options["cli_prompt"] = "\n"
    prog.options["audio_show_transcript"] = False
    prog.options["tts_subtitles"] = False
    prog.options["streaming"] = False
    prog.options["chat_show_ai_prompt"] = False
    return ""

def varfile(prog, argv):
    # $FIXME:
    return "Not implemented yet. Use the -V option for now."

def exitProgram(prog, argv):
    """
    Quit the program. Chat history (story folder) is discarded. All options are lost.
    See also /save, /saveoptions, /saveconfig"""
    prog.stopAudioTranscription()
    prog.running = False
    return ""

    
def transcribe(prog, argv):
    """Records using the microphone until you hit enter. The recording is then transcribed using openai's whisper, and inserted into the current line at the CLI.
THe precise model to be used for transcribing is determined using the 'whisper_model' option. Larger models transcribe more accurately, and may handle more languages, but also consume more resources. See https://huggingface.co/openai/whisper-large for a list of model names. The default is 'base.en'.
The model will be automatically downloaded the first time you transcribe with it. This may take a moment, but will only happen once for each model."""
    w = prog.whisper.transcribe()
    sys.stdout.write("\r" + w + "\n")
    prog.continueWith(w)
    return ""

def toggleAudio(prog, argv):
    """Enable/disable audio input. This means the program will automatically record audio and transcribe it using the openai whisper model. A query is send to the backend with transcribed speech whenever a longish pause is detected in the input audio.
See whisper_model for the model used for transcribing."""
    if prog.isAudioTranscribing():
        prog.stopAudioTranscription()
    else:
        prog.startAudioTranscription()
    return ""


    


cmds_additional_docs = {
    "/log" : """
    Prints the raw log of the current story branch.
    This includes prompt-format tokens and other stuff that is normally filtered out. For a prettier display, see /print.
    Also, /log prints to stderr, while /print will output to stdout.""",
    "/resend" : retry.__doc__,
    "/rephrase" : retry.__doc__,
    "/restart" : """
    Restarts the current character folder. Note that this will wipe the current story folder, i.e. your chat history, so you may want to /save.
    /restart is equivalent to /start CURRENT_CHAR_FOLDER."""    
    }
