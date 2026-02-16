import os, datetime, glob, sys, requests, traceback, random, json
from typing import Any, List, Tuple, Callable, Optional, TYPE_CHECKING
from .session import Session
from .util import *
from .StoryFolder import *
from .definitions import *
from .api_internal import *
from .client import RemoteInfo
if TYPE_CHECKING:
    from .main import Plumbing


def newSession(program: 'Plumbing', argv: List[str], keep: bool = False) -> str:
    """CHARACTER_FOLDER
    Start a new session with the character or template defined in CHARACTER_FOLDER. You can specify a full path, or just the folder name, in which case the program will look for it in all folders specified in the 'include' paths. See /lsoptions include.
    """
    if argv == []:
        if program.getOption("character_folder"):
            # synonymous with /restart
            argv.append(program.getOption("character_folder"))
        else:
            return "No path provided. Cannot Start."

    filepath = " ".join(argv)
    return start_session(program, filepath, keep=keep)


def printStory(prog: 'Plumbing', argv: List[str], stderr: bool = False, apply_filter: bool = True) -> str:
    """[FILENAME]
    Print the current story.
    If FILENAME is provided, save the story to that file."""
    # apply_filter basically means make it pretty
    if apply_filter:
        w = prog.formatStory(with_color=prog.getOption("color"))
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


def doContinue(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Continue generating without new input. Use this whenever you want the AI to 'just keep talking'.
    Specifically, this will send the story up to this point verbatim to the LLM backend. You can check where you are with /log.
    If there are templated tokens that are normally inserted as part of the prompt format when a user message is received, they are not inserted. Existing end tokens, like <|im_end|> will be removed before sending the prompt back.
    This command is also executed when you hit enter without any text."""

    prog.setOption("continue", "1")
    # FIXME: often we just get EOS. this is sort of risky though.
    # prog.setOption("ignore_eos", True)
    if prog.session.stories.empty():
        return ""

    return ""


def setOption(prog: 'Plumbing', argv: List[str]) -> str:
    """OPTION_NAME [OPTION_VALUE]
    Set options during program execution. To see a list of all possible values for OPTION_NAME, do /lsoptions. OPTION_VALUE must be a valid python expression, so if you want to set e.g. chat_user to Bob, do /set chat_user \"Bob\".
    When OPTION_VALUE is omitted, the value is set to \"\". This is equivalent to /unset OPTION_NAME.
    """

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
        printerr(traceback.format_exc())
        return "Couldn't set " + name + " to " + w + ". Couldn't evaluate."
    return ""


def showOptions(prog: 'Plumbing', argv: List[str]) -> str:
    """ [OPTION_NAME]
    Displays the list of program options, along with their values. Provide OPTION_NAME to see just its value.
    The options displayed can all be set using /set OPTION_NAME.
    Almost all of them may also be provided as command line arguments with preceding dashes, e.g. include as --include=/some/path.
    Finally, they may be set in various config files, such as in character folders, or in a config file loaded with /load.
    """
   
    if argv == []:
        target = ""
    elif argv[0] == "--emacs":
        # undocumented, outputs all options in a neat form to put as keywords in ghostbox.el
        w = "`("
        for name in prog.options.keys():
            w += '"' + name + '"', 
        w = w[:-2] + ")"
        printerr(w)
        return ""
    else:
        target = " ".join(argv)

    w = ""
    for k, v in prog.options.items():
        if target in k:
            w += k + ": " + str(v) + "\n"
    return w.strip()


def showVars(prog: 'Plumbing', argv: List[str]) -> str:
    """[VARIABLE_NAME]
    Shows all session variables and their respective values. Show a single variable if VARIABLE_NAME is provided.
    Variables are automatically expanded within text that is provided by the user or generated by the LLM backend.
    E.g. if variable favorite_fruit is set to 'Tomatoe', any occurrence of {{favorite_fruit}} in the text will be replaced by 'Tomatoe'.
    There are three ways to set variables:
      1. Loading a character folder (/start). All files in a character folder automatically become variables, with the respective filename becoming the name of the variable, and the file's content becoming the value of the variable.
      2. The -x or --varfile command line option. This argument provides additional files that may serve as variables, similar to (1). The argument may be repeated multiple times.
      3. API only: Using the set_vars method on a Ghostbox object.

    You can also do {[FILENAME]} to ad-hoc splice the contents of FILENAME into a prompt. However, due to security reasons, this only works at the CLI.
    """

    if argv == []:
        return "\n".join(
            [f"{key}\t{value}" for key, value in prog.session.fileVars.items()]
        )

    k = " ".join(argv)
    return f"{prog.session.fileVars.get(k, f"Could not find var '{k}'")}"


def showChars(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Lists all available character folders. These can be used with the /start command or the -c command line argument, and are valid values for the character_folder parameter in the Ghostbox API.
    To see what places are searched for characters, see the value of the 'include' option.
    """
    return "\n".join(sorted(all_chars(prog)))


def showTemplates(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Lists all available templates for prompt formats.
    To see the places searched, check the value of the template_include option."""
    allchars: List[str] = []
    for dir in prog.getOption("template_include"):
        if not (os.path.isdir(dir)):
            printerr("warning: template Include path '" + dir + "' is not a directory.")
            continue
        for charpath in glob.glob(dir + "/*"):
            if os.path.isfile(charpath):
                continue
            allchars.append(os.path.split(charpath)[1])
    return "\n".join(sorted(allchars))


def showVoices(prog: 'Plumbing', argv: List[str]) -> str:
    """
    List all available voices for the TTS program. These can be used with /set tts_voice.
    To see the places searched for voices, see the value of tts_voice_dir.
    Depending on the value of tts_program, that location may be meaningless, as the voice won't be file based, like with amazon polly voices. In this case /lsvoices tries to give a helpful answer if it can.
    """
    return "\n".join(getVoices(prog))


def showModels(prog: 'Plumbing', argv: List[str]) -> str:
    """
    List all available models for the backend. For most local backends, this is not supported.
    """
    models = prog.getBackend().get_models()
    if models:
        w = "Name\tModel\tDescription"
        for model in models:
            w += f"\n{model.display_name}\t{model.name}\t{model.description}"
        return w

    return "Model listing not supported by backend."
    


def toggleMode(prog: 'Plumbing', argv: List[str]) -> str:
    """[MODE_NAME}
    Put the program into the specified mode, or show the current mode if MODE_NAME is omitted.
    Possible values are 'default', or 'chat'.
    Setting a certain mode has various effects on many aspects of program execution. Currently, most of this is undocumented :)
    """
    if argv == []:
        return "Currently in " + prog.getOption("mode") + " mode."

    mode = " ".join(argv)
    if prog.isValidMode(mode):
        prog.setMode(mode)
        return mode + " mode on"
    return "Not a valid mode. Possible values are 'default' or 'chat'"


def toggleTTS(prog: 'Plumbing', argv: List[str]) -> str:
    """
    This turns the TTS (text-to-speech) module on or off. When TTS is on, text that is generated by the LLM backend will be spoken out loud.
    The TTS service that will be used depends on the value of tts_program. The tts_program can be any executable or shell script that reads lines from standard input. It may also support additional functionality.
    An example tts program for amazon polly voices is provided with 'ghostbox-tts-polly'. Note that this requires you to have credentials with amazon web services.
    On linux distributions with speech-dispatcher, you can set the value of tts_program to 'spd-say', or 'espeak-ng' if it's installed. This works, but it isn't very nice.
    The voice used depends on the value of tts_voice, which you can either /set or provide with the -V or --tts_voice command line option. It can also be set in character folder's config.json.
    ghostbox will attempt to provide the tts_program with the voice using a -V command line argument.
    To see the list of supported voices, try /lsvoices.
    Enabling TTS will automatically set stream_flush to 'sentence', as this works best with most TTS engines. You can manually reset it to 'token' if you want, though.
    """
    return toggle_tts(prog)


def ttsDebug(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Get stdout and stderr from the underlying tts_program when TTS is enabled."""
    if not (prog.tts):
        return "TTS is None."

    w = "tts_program: " + prog.getOption("tts_program")
    w += "\ntts exit code: " + str(prog.tts.exit_code())
    w += "\ntts config: "
    if prog.tts_config is None:
        w += "unavailable"
    else:
        w += json.dumps(prog.tts_config, indent=4)
    printerr(w)

    w = "\n### STDOUT ###\n"
    w += "".join(prog.tts.get())
    w += "\n### STDERR ###\n"
    w += "".join(prog.tts.get_error())
    printerr(w, prefix="")
    return ""


def nextStory(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Go to next branch in story folder."""
    r = prog.session.stories.nextStory()
    if r == 1:
        return "Cannot go to next story branch: No more branches. Create a new branch with /new or /retry."
    return "Now on branch " + str(prog.session.stories.index)


def previousStory(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Go to previous branch in story folder."""
    r = prog.session.stories.previousStory()
    if r == -1:
        return "Cannot go to previous story branch: No previous branch exists."
    return "Now on branch " + str(prog.session.stories.index)


def retry(prog: 'Plumbing', argv: List[str], predicate: Callable[[Any], bool] = lambda item: item.role == "user") -> str:
    """
        Retry generation of the LLM's response.
    This will drop the last generated response from the current story and generate it again. Use this in most cases where you want to regenerate. If you extended the LLM's repsone (with /cont or hitting enter), the entire repsonse will be regenerated, not just the last part.
    Note that /retry is not destructive. It always creates a new story branch before regenerating a repsonse.
        See also: /rephrase."""
    prog.session.stories.cloneStory()
    prog.session.stories.get().dropUntil(predicate)
    doContinue(prog, [])
    printerr("Now on branch " + str(prog.session.stories.index))
    return ""


def rephrase(prog: 'Plumbing', argv: List[str]) -> str:
    """
    This will rewind the story to just before your last input, allowing you to rephrase your query.
        Note that /rephrase is not destructive. It will always create a new story branch before rewinding.
        See also: /retry"""
    prog.session.stories.cloneStory()
    story = prog.session.stories.get()
    story.dropUntil(lambda item: item.role == "user")
    story.dropUntil(lambda item: item.role == "assistant")
    printerr("Now on branch " + str(prog.session.stories.index))
    return ""


def dropEntry(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Drops the last entry from the current story branch, regardless of wether it was user provided or generated by the LLM backend.
    """
    prog.session.stories.cloneStory()
    prog.session.stories.get().drop()
    return (
        "Now on branch " + str(prog.session.stories.index) + " with last entry dropped."
    )


def newStory(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Create a new, empty branch in the story folder. You can always go back with /prev or /story.
    """
    prog.session.stories.newStory()
    return "Now on branch " + str(prog.session.stories.index) + " with a clean log."


def cloneStory(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Create a new branch in the story folder, copying the contents entirely from the current story.
    """
    prog.session.stories.cloneStory()
    return (
        "Now on branch "
        + str(prog.session.stories.index)
        + " with a copy of the last branch."
    )


def saveStoryFolder(prog: 'Plumbing', argv: List[str]) -> str:
    """[STORY_FOLDER_NAME]
    Save the entire story folder in the file STORY_FOLDER_NAME. If no argument is provided, creates a file with the current timestampe as name.
    The file created is your accumulated chat history in its entirety, including the current story and all other ones (accessible with /prev and /next, etc). It is saved in the json format.
    A saved story folder may be loaded with /load."""
    if len(argv) > 0:
        name = " ".join(argv)
    else:
        name = ""

    try:
        filename = save(prog, name, overwrite=False)
    except RuntimeError:
        filename = None
            
    if not (filename):
        return "Could not save story folder. Maybe provide a different filename?"
    return "Saved story folder as " + filename


def changeTemplate(prog: 'Plumbing', argv: List[str]) -> str:
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


def loadStoryFolder(prog: 'Plumbing', argv: List[str]) -> str:
    """STORY_FOLDER_NAME
    Loads a previously saved story folder from file STORY_FOLDER_NAME. See the /save command on how to save story folders.
    A story folder is a json file containing the entire chat history."""
    if len(argv) == 0:
        return "Please provide a legal json filename in the story-folder format."

    filename = " ".join(argv)
    try:
        load(prog, filename)
    except FileNotFoundError as e:
        return "Could not load " + filename + ": file not found."
    except Exception as e:
        printerr(str(e))
        return "Could not load story folder: Maybe bogus JSON?"

    return (
        "Ok. Restored "
        + filename
        + "\nNow on branch "
        + str(prog.session.stories.index)
    )


def gotoStory(prog: 'Plumbing', argv: List[str]) -> str:
    """[BRANCH_NUMBER]
    Moves to a different branch in the story folder. If BRANCH_NUMBER is omitted, shows the current branch instead.
    """
    if argv == []:
        return "Currently on branch " + str(prog.session.stories.index)

    w = " ".join(argv)
    try:
        n = int(w)
    except:
        return "Cannot go to that branch: Invalid Argument."

    err = prog.session.stories.shiftTo(n)
    if not (err):
        return ""
    return "Could not go to branch " + str(n) + ": " + err


def loadConfig(prog: 'Plumbing', argv: List[str], override: bool = True, protected_keys: List[str] = []) -> str:
    """CONFIG_FILE
    Loads a json config file at location CONFIG_FILE. A config file contains a dictionary of program options.
    You can create an example config with /saveconfig example.conf.json.
    You can also load a config file at startup with the --config_file command line argument.
    If it exists, the ~/.ghostbox.conf.json will also be loaded at startup.
    The order of config file loading is as follows .ghostconf.conf.json > --config_file > conf.json (from character folder). Config files that are loaded later will override settings from earlier files.
    """

    # loads a config file, which may be a user config, or a config supplied by a character folder. if override is false, the config may not override command line arguments that have been manually supplied (in the long form)
    if argv == []:
        return "Please provide a filename for the config file to load."
    filename = " ".join(argv)
    return load_config(prog, filename, override=override, protected_keys=protected_keys)


def saveConfig(prog: 'Plumbing', argv: List[str]) -> str:
    """CONFIG_FILE
    Save the current program options and their values to the file at location CONFIG_FILE. This will either create or overwrite the CONFIG_FILE, deleting all its previous contents.
    """

    if argv == []:
        name = "config-" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    else:
        name = " ".join(argv)

    if not (name.endswith(".json")):
        name = name + ".json"

    filename = saveFile(name, json.dumps(prog.options, indent=4))
    if filename:
        return "Saved config to " + filename
    return "error: Could not save config."


def hide(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Hide various program outputs and change some variables for a less verbose display.
    This does nothing that you cannot achieve by manually setting several options, it just bundles an eclectic mix of them in one command.
    I like to use this with TTS for a more immersive experience."""
    # this is just convenient shorthand for when I want my screen reader to be less spammy
    hide_some_output(prog)
    return ""


def varfile(prog: 'Plumbing', argv: List[str]) -> str:
    # $FIXME:
    return "Not implemented yet. Use the -V option for now."


def exitProgram(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Quit the program. Chat history (story folder) is discarded. All options are lost.
    See also /save, /saveoptions, /saveconfig"""
    prog.stopAudioTranscription()
    prog.stopImageWatch()
    # prevent last cli printing
    prog.setOption("cli_prompt", "")
    prog.running = False
    return ""


def transcribe(prog: 'Plumbing', argv: List[str]) -> str:
    """Records using the microphone until you hit enter. The recording is then transcribed using openai's whisper, and inserted into the current line at the CLI.
    THe precise model to be used for transcribing is determined using the 'whisper_model' option. Larger models transcribe more accurately, and may handle more languages, but also consume more resources. See https://huggingface.co/openai/whisper-large for a list of model names. The default is 'base.en'.
    The model will be automatically downloaded the first time you transcribe with it. This may take a moment, but will only happen once for each model.
    """
    w = prog.whisper.transcribeWithPrompt()
    sys.stdout.write("\r" + w + "\n")
    prog.continueWith(w)
    return ""


def toggleAudio(prog: 'Plumbing', argv: List[str]) -> str:
    """Enable/disable audio input. This means the program will automatically record audio and transcribe it using the openai whisper model. A query is send to the backend with transcribed speech whenever a longish pause is detected in the input audio.
    See whisper_model for the model used for transcribing."""
    if prog.isAudioTranscribing():
        prog.stopAudioTranscription()
    else:
        prog.startAudioTranscription()
    return ""


def image(prog: 'Plumbing', argv: List[str]) -> str:
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
        prog.images = {}
        return "Images reset."

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


def debuglast(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Dumps a bunch of information about the last result received. Note that this won't do anything if you haven't sent a request to a working backend server that answered you.
    """
    if not (r := prog.lastResult):
        return "Nothing."

    acc: List[str] = []
    for k, v in r.items():
        acc.append(k + ": " + str(v))
    return "\n".join(acc)


def lastRequest(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Dumps a bunch of information about the last request send. Note that this won't do anything if you haven't sent a request to a working backend server that answered you.
    """
    r = prog.getBackend().getLastRequest()
    if not (r):
        return "Nothing."

    return json.dumps(r, indent=4)


def showTime(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Show some performance stats for the last request."""
    r = prog.getBackend().timings()
    if not (r):
        return "No time statistics. Either no request has been sent yet, or the backend doesn't support timing."

    w = ""
    # timings: {'predicted_ms': 4115.883, 'predicted_n': 300, 'predicted_per_second': 72.88836927580303, 'predicted_per_token_ms': 13.71961, 'prompt_ms': 25.703, 'prompt_n': 0, 'prompt_per_second': 0.0, 'prompt_per_token_ms': None}
    # caching etc
    w += "generated: " + str(r.predicted_n)
    w += ", evaluated: " + str(r.prompt_n)
    w += (
        ", cached: " + (str(r.cached_n) if r.cached_n is not None else "unknown") + "\n"
    )
    w += (
        "context: "
        + str(r.total_n())
        + " / "
        + str(prog.getOption("max_context_length"))
        + ", exceeded: "
        + str(r.truncated)
    )
    if prog._smartShifted:
        w += "(smart shifted)\n"
    else:
        w += "\n"

    factor = 1 / 1000
    unit = "s"
    prep = lambda u: str(round(u * factor, 2))
    w += prep(r.prompt_ms) + unit + " spent evaluating prompt.\n"
    w += prep(r.predicted_ms) + unit + " spent generating.\n"
    w += prep(r.total_ms()) + unit + " total processing time.\n"
    w += (
        str(round(r.predicted_per_second, 2))
        + "T/s, "
        + prep(r.predicted_per_token_ms)
        + unit
        + "/T"
    )
    return w


def showStatus(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Give an overall report about the program and some subprocesses."""
    if argv == []:
        topics = set("backend mode tts audio image_watch streaming".split(" "))
    else:
        topics = set(argv)

    w = ""
    if "backend" in topics:
        w += (
            "backend: "
            + prog.getOption("backend")
            + " at "
            + prog.getOption("endpoint")
            + "\n"
        )
        w += "backend status: " + str(prog.getBackend().health()) + "\n"
        w += "max_context_length: " + str(prog.getOption("max_context_length"))
        if prog._dirtyContextLlama:
            # context has been set by server
            w += " (set by llama.cpp)\n"
        else:
            w += "\n"

        w += " models\n"
        models = dirtyGetJSON(prog.getOption("endpoint") + "/v1/models").get("data", [])
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
        if prog.tts is None:
            w += "uninitialized"
        else:
            if prog.tts.is_running():
                w += "running"
            else:
                w += "exit code " + str(prog.tts.exit_code())
        w += "\n\n"

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
            w += "streaming: " + str(prog.getOption("stream")) + "\n\n"

    return w.strip()


def toggleImageWatch(prog: 'Plumbing', argv: List[str]) -> str:
    """[DIR]
    Enable / disable automatic watching for images in a specified folder.
        When new images are created / modified in image_watch_dir, a message (image_watch_msg)is automatically sent to the backend.
        If DIR is provided to this command, image_watch_dir will be set to DIR. Otherwise, the preexisting image_watch_dir is used, which defaults to the user's standard screenshot folder.
        This allows you to e.g. take screenshots and have the TTS automatically describe them without having to switch back to this program.
        Check status with /status image_watch."""
    dir = " ".join(argv)
    if dir != "":
        if not (os.path.isdir(dir)):
            return "error: Could not start image_watch: " + dir + " is not a directory."
        prog.setOption("image_watch", dir)

    # toggle
    prog.options["image_watch"] = not (prog.getOption("image_watch"))
    if prog.getOption("image_watch"):
        prog.startImageWatch()
    else:
        prog.stopImageWatch()
        return "image_watch off."
    return ""


def showRaw(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Displays the raw output for the last prompt that was sent to the backend."""
    printerr(prog._lastPrompt, prefix="")
    return ""


def switch(prog: 'Plumbing', argv: List[str]) -> str:
    """CHARACTER_FOLDER
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


def tokenize(prog: 'Plumbing', argv: List[str]) -> str:
    """[-c] MSG
    Send a tokenize request to the server. Will print raw tokens to standard output, one per line. This is mostly used to debug prompts. With -c, it will print the number of tokens instead.
    """
    if argv != [] and argv[0] == "-c":
        count = True
        argv = argv[1:]
    else:
        count = False

    ts = prog.getBackend().tokenize(" ".join(argv))
    if count:
        return str(len(ts))

    for t in ts:
        print(t)
    return ""


def detokenize(prog: 'Plumbing', argv: List[str]) -> str:
    """[-n]
    Turn a list of tokens into strings.
    Tokens can be supplied like this /detokenize 23 20001 1
        If -n is supplied, reads one token per line until an empty line is found."""
    ts: List[int] = []
    if argv != [] and argv[0] == "-n":
        # one per line
        argv = []
        while True:
            w = input()
            if w == "":
                break
            argv.append(w)

    for arg in argv:
        try:
            t = int(arg)
        except:
            return "Please specify tokens as integers, seperated by spaces, or newlines in case you supplied -n."
        ts.append(t)

    w = prog.getBackend().detokenize(ts)
    print(w)
    return ""

def client_handshake(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Internal command used by ghostbox in --client mode."""
    required_options = RemoteInfo.model_fields.keys()
    payload = {option : prog.getOption(option) for option in required_options}
    # this will go to the client
    # it will get printed server side but oh well
    prog.print(RemoteInfo.show_json(json.dumps(payload)), tts=False)
    return ""

    

def testQuestion(prog: 'Plumbing', argv: List[str]) -> str:
    """
    Send a random test question to the backend.
    The question is pulled from a random set of example questions. These are sometimes funny, but also turn out to be quite useful to get a general sense of a model.
    """

    questions = """I have 8 eggs, 4 water bottles, and 1 laptop. Suggest to me a configuration of these objects with which to balance them on top of another.
How much does 1 kilogram of feathers weigh?
I have 3 apples. I give 2 to timmy. How many apples do I have?
Me and my friends have a rule: "If you borrow a sweater, then you have to return it." Last month I got Jane's sweater. I did not return it. What do I have to do now?
You’re in a desert walking along in the sand when all of the sudden you look down, and you see a tortoise, it’s crawling toward you. You reach down, you flip the tortoise over on its back. The tortoise lays on its back, its belly baking in the hot sun, beating its legs trying to turn itself over, but it can’t, not without your help. But you’re not helping. Why is that?
Describe in single words, only the good things that come into your mind about your mother.
In a magazine you come across a full-page color picture of a nude girl. Your husband likes the picture. The girl is lying facedown on a large and beautiful bearskin rug. Your husband hangs the picture up on the wall of his study. How do you feel?
A young boy shows you his butterfly collection, including the killing jar. How do you feel?
You’re reading a novel written in the old days before the war. The characters are visiting Fisherman’s Wharf in San Francisco. They become hungry and enter a seafood restaurant. One of them orders lobster, and the chef drops the lobster into the tub of boiling water while the characters watch. What do you think about this?
You are watching an old movie on TV, a movie from before the war. It shows a banquet in progress; the guests are enjoying raw oysters. The entrée consists of boiled dog, stuffed with rice. Are raw oysters more acceptable to you than a dish of boiled dog?""".split(
        "\n"
    )

    w = random.choice(questions)
    # no prefix
    print(w)
    prog.continueWith(w)
    return ""


cmds_additional_docs = {
    "/log": """
    Prints the raw log of the current story branch.
    This includes prompt-format tokens and other stuff that is normally filtered out. For a prettier display, see /print.
    Also, /log prints to stderr, while /print will output to stdout.""",
    "/resend": retry.__doc__,
    "/rephrase": retry.__doc__,
    "/restart": """
    Restarts the current character folder. Note that this will wipe the current story folder, i.e. your chat history, so you may want to /save.
    /restart is equivalent to /start CURRENT_CHAR_FOLDER.""",
}
