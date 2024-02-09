import os, datetime, glob, sys, requests
from ghostbox.session import Session
from ghostbox.util import *
from ghostbox.StoryFolder import *

def newSession(program, argv, keep=False):
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
            s = Session(dir=path, chat_user=program.getOption("chat_user"), additional_keys=program.getOption("var_file"))
            break
        except FileNotFoundError as e:
            # session will throw if path is not valid, we keep going through the includes
            failure = e

    if failure:
        return "error: " + str(failure)

    # constructing new session worked
    if not(keep):
        program.session = s
    else:
        # something like /switch happened, we want to keep some stuff
        program.session.merge(s)
        
    w = ""
    # try to load config.json if present
    configpath = path + "/config.json"
    if os.path.isfile(configpath):
        w += loadConfig(program, [configpath], override=False) + "\n"
    program.options["character_folder"] = path
    # this might be very useful for people to debug their chars, so we are a bit verbose here by default
    w += "Found vars " + ", ".join([k for k in program.session.getVars().keys() if k != "chat_user" and k != "config.json"]) + "\n"
    w += "Ok. Loaded " + path


    # by convention, the initial message is stored in initial_msg
    if program.session.hasVar("initial_msg") and not(keep):
        program.initial_print_flag = True

    # config may have set the mode, but we need to go through setMode
    program.setMode(program.getOption("mode"))

    # hide if option is set
    if program.getOption("hide"):
        hide(program, [])

    return w

def printStory(prog, argv, stderr=False, apply_filter=True):
    """[FILENAME]
    Print the current story.
    If FILENAME is provided, save the story to that file."""
    # apply_filter basically means make it pretty
    if apply_filter:
        w = prog.formatStory()
    else:
        w = prog.showStory(append_hint=False)

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
        return "\n".join([f"{key}" for key in prog.session.fileVars.keys()])

    k = " ".join(argv)
    return prog.session.fileVars.get(k, f"Could not find var '{k}'")

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

def showTemplates(prog, argv):
    """
Lists all available templates for prompt formats.
To see the places searched, check the value of the template_include option."""
    allchars = []
    for dir in prog.getOption("template_include"):
        if not(os.path.isdir(dir)):
            printerr("warning: template Include path '" + dir + "' is not a directory.")
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

def retry(prog, argv, predicate=lambda item: item["role"] == "user"):
    """
    Retry generation of the LLM's response.
This will drop the last generated response from the current story and generate it again. Use this in most cases where you want to regenerate. If you extended the LLM's repsone (with /cont or hitting enter), the entire repsonse will be regenerated, not just the last part.
Note that /retry is not destructive. It always creates a new story branch before regenerating a repsonse.
    See also: /rephrase."""
    prog.session.stories.cloneStory()
    prog.session.stories.get().dropUntil(predicate)
    doContinue(prog, [])
    printerr("Now on branch " + str(prog.session.stories.index) )
    return ""

def rephrase(prog, argv):
    """
This will rewind the story to just before your last input, allowing you to rephrase your query.
    Note that /rephrase is not destructive. It will always create a new story branch before rewinding.
    See also: /retry"""
    prog.session.stories.cloneStory()
    story = prog.session.stories.get()
    story.dropUntil(lambda item: item["role"] == "user")
    story.dropUntil(lambda item: item["role"] == "assistant")
    printerr("Now on branch " + str(prog.session.stories.index) )    
    return ""



def dropEntry(prog, argv):
    """
    Drops the last entry from the current story branch, regardless of wether it was user provided or generated by the LLM backend."""
    prog.session.stories.cloneStory()
    prog.session.stories.get().drop()
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
        name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".json"
            
    filename = saveFile(name, prog.session.stories.toJSON())
    if not(filename):
        return "Could not save story folder. Maybe provide a different filename?"
    return "Saved story folder as " + filename



def changeTemplate(prog, argv):
    """[PROMPT_TEMPLATE]
Shows the current prompt template. If PROMPT_TEMPLATE is supplied, tries to load that template.
LLMs usually work best when supplied with input that is formatted similar to their training data. Prompt templates apply some changes to your inputs in the background to get them into a shape that the LLM expects.
    To find out the right template to use, consult the model card of the LLM you are using. When in doubt, 'chat-ml' is a very common prompt format.
To disable this, set the template to 'raw' and you won't have anything done to your inputs. This can be useful for experimentation.
Templates are searched for in the directories specified by the template_include option, which can be supplied at the command line.
To get a full list of available templates, try /lstemplates ."""

    if argv == []:
        return prog.getOption("prompt_format")
        
    choice = " ".join(argv)
    prog.loadTemplate(choice)
    return ""
    
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
    prog.stopImageWatch()
    prog.running = False
    return ""

    
def transcribe(prog, argv):
    """Records using the microphone until you hit enter. The recording is then transcribed using openai's whisper, and inserted into the current line at the CLI.
THe precise model to be used for transcribing is determined using the 'whisper_model' option. Larger models transcribe more accurately, and may handle more languages, but also consume more resources. See https://huggingface.co/openai/whisper-large for a list of model names. The default is 'base.en'.
The model will be automatically downloaded the first time you transcribe with it. This may take a moment, but will only happen once for each model."""
    w = prog.whisper.transcribeWithPrompt()
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

def image(prog, argv):
    """[--id=IMAGE_ID] IMAGE_PATH
Add images for multimodal models that can handle them. You can refer to images by their id in the form of `[img-ID]`. If --id is omitted, id= 1 is assumed. Examples:
```
    /image ~/Pictures.test.png
Please describe [img-1].
    ```

    Alternatively, with multiple images:
    ```
    /image --id=1 ~/Pictures.paris.png
    /image --id=2 ~/Pictures/berlin.png
Can you compare [img-1] and [img-2]?
    ```"""

    if argv == []:
        ws = []
        for (id, imagedata) in sorted(prog.images.items()):
            ws.append(mkImageEmbeddingString(id) + "\t" + imagedata["url"])
        return "\n".join(ws)

    if argv[0].startswith("--id="):
        id = maybeReadInt(argv[0].replace("--id=", ""))
        if id is None:
            return "error: Please specify a valid integer as id."
        url = " ".join(argv[1:])
    else:
        id = 1
        url = " ".join(argv)

    prog.loadImage(url, id)
    return ""
        
            
def debuglast(prog, argv):
    """
    Dumps a bunch of information about the last request send. Note that this won't do anything if you haven't sent a request to a working backend server that answered you."""
    r = prog.lastResult
    if not(r):
        return "Nothing."

    acc = []
    for (k, v) in r.items():
        acc.append(k + ": " + str(v))
    return "\n".join(acc)

def showTime(prog, argv):
    """
Show some performance stats for the last request."""
    r = prog.lastResult
    if not(r):
        return "No time statistics. Either no request has been sent yet, or the backend doesn't support timing."

    if prog.getOption("backend") != "llama.cpp":
        return "Timings are not implemented yet for this backend."
    
    w = ""
    # timings: {'predicted_ms': 4115.883, 'predicted_n': 300, 'predicted_per_second': 72.88836927580303, 'predicted_per_token_ms': 13.71961, 'prompt_ms': 25.703, 'prompt_n': 0, 'prompt_per_second': 0.0, 'prompt_per_token_ms': None}    
    dt = r["timings"]    
    #caching etc
    w += "generated: " + str(dt["predicted_n"]) 
    w += ", evaluated: " + str(r["tokens_evaluated"])
    w += ", cached: " + str(r["tokens_cached"]) + "\n"
    w += "context: " + str(r["tokens_evaluated"] + dt["predicted_n"]) + " / " +  str(prog.getOption("max_context_length")) + ", exceeded: " + str(r["truncated"])
    if prog._smartShifted:
        w += "(smart shifted)\n"
    else:
        w += "\n"
        


    
    factor = 1/1000
    unit = "s"
    prep = lambda u: str(round(u*factor, 2)) 
    w += prep(dt["prompt_ms"]) + unit + " spent evaluating prompt.\n"
    w += prep(dt["predicted_ms"]) + unit + " spent generating.\n"
    w += prep(dt["predicted_ms"] + dt["prompt_ms"]) + unit + " total processing time.\n"  
    w += str(round(dt["predicted_per_second"], 2)) + "T/s, " + prep(dt["predicted_per_token_ms"]) + unit + "/T"
    return w

def showStatus(prog, argv):
    """
Give an overall report about the program and some subprocesses."""
    if argv == []:
        topics = set("backend mode tts audio image_watch streaming".split(" "))
    else:
        topics = set(argv)

    w = ""
    if "backend" in topics:
        w += "backend: " + prog.getOption("backend") + " at " + prog.getOption("endpoint") + "\n"
        w += "backend status: " + str(prog.getBackend().health()) + "\n"
        w += "max_context_length: " + str(prog.getOption("max_context_length"))
        if prog._dirtyContextLlama:
            # context has been changed by llama server
            w += " (set by llama.cpp)\n"
        else:
            w += "\n"
        
        w += " models\n"
        models =         dirtyGetJSON(prog.getOption("endpoint") + "/v1/models").get("data", [])
        for m in models:
            w += " .. " + m["id"] + "\n\n"

    if "mode" in topics:
        w += "mode: " + prog.getOption("mode") + "\n"
        w += "\n"
        # FIXME: more mode stuff here

    if "tts" in topics:
        w += "tts: " + str(prog.getOption("tts")) + "\n"
        w += "tts_program: " + prog.getOption("tts_program") + "\n"
        w += "tts status: "
        # this is tricky
        tts = prog.getOption("tts")
        if not(tts):
            # tts is false so it hasn't been started or it has been stopped
            if prog.tts is None:
                w += "N/A\n"
            else:
                # this is a weird case
                w += "shutting down\n"
        else:
            # tts was true so someone started it, now we need to know about the process
            if prog.tts is None:
                w += "failed to start\n"
            else:
                r = prog.tts.poll()
                if r is None:
                    w += "running\n"
                else:
                    w += "exited with status " + str(r) + "\n"
        w += "\n"

    if "audio" in topics:
        w += "audio transcription: " + str(prog.getOption("audio")) + "\n"
        w += "whisper_model: " + prog.getOption("whisper_model") + "\n"
        w += "continuous record / transcribe status: "
        if prog.ct is None:
            w += "N/A\n"
        else:
            if prog.ct.running:
                w += "running"
                if prog.ct.isPaused():
                    w += " (paused)"
                w += "\n"
            else:
                w += "stopped\n"
        w += "\n"
                
    if "image_watch" in topics:
        w += "image_watch: " + str(prog.getOption("image_watch")) + "\n"
        w += "image_watch_dir: " + prog.getOption("image_watch_dir") + "\n"
        w += "status: "
        
        if prog.image_watch is None:
            w += "N/A\n"
        else:
            if prog.image_watch.running:
                w += "running\n"
            else:
                w += "halted"
        w += "\n"

        if "streaming" in topics:
            w += "streaming: " + str(prog.getOption("streaming")) + "\n\n"
        
    return w.strip()

def toggleImageWatch(prog, argv):
    """[DIR]
Enable / disable automatic watching for images in a specified folder.
    When new images are created / modified in image_watch_dir, a message (image_watch_msg)is automatically sent to the backend.
    If DIR is provided to this command, image_watch_dir will be set to DIR. Otherwise, the preexisting image_watch_dir is used, which defaults to the user's standard screenshot folder.
    This allows you to e.g. take screenshots and have the TTS automatically describe them without having to switch back to this program.
    Check status with /status image_watch."""
    dir = " ".join(argv)
    if dir != "":
        if not(os.path.isdir(dir)):
            return "error: Could not start image_watch: " + dir + " is not a directory."
        prog.setOption("image_watch", dir)
        
    # toggle
    prog.options["image_watch"] = not(prog.getOption("image_watch"))
    if prog.getOption("image_watch"):
        prog.startImageWatch()
    else:
        prog.stopImageWatch()
        return "image_watch off."
    return ""
        

def showRaw(prog, argv):
    """
    Displays the raw output for the last prompt that was sent to the backend."""
    printerr(prog._lastPrompt, prefix="")
    return ""


def switch(prog, argv):
    """ CHARACTER_FOLDER
Switch to another character folder, retaining the current story.
As opposed to /start or /restart, this will hot-switch the AI without wiping your story so far, or adding the initial message. This can e.g. allow the used of specialized AI's in certain situations.
Specifically, '/switch bob' will do the following:
 - Set your system prompt to bob/system_msg
 - Set all session variables defined in bob/, possibly overriding existing ones
 - Load bob/config.json, if present, not overriding command line arguments.
See also: /start, /restart, /lschars"""
    if argv == []:
        return "No character folder given. No switch occured."
    
    prog._lastChar = prog.session.dir
    w = newSession(prog, argv, keep=True)
    # FIXME: when people use this command they probably don't want to be spammed, so we discard w, but maybe we want a verbose mode?
    return w.split("\n")[-1]
                   
    
def tokenize(prog, argv):
    """MSG
    Send a tokenize request to the server. Will print raw tokens to standard output, one per line. This is mostly used to debug prompts."""
    ts = prog.tokenize("".join(argv))
    for t in ts:
        print(t)
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
