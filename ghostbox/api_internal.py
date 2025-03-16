import os, datetime, glob, sys, requests, traceback, random, json
from typing import *
from ghostbox.session import Session
from ghostbox.util import ultraglob

def start_session(plumbing, filepath: str, keep=False) -> str:
    
    allpaths = [filepath] + [p + "/" + filepath for p in plumbing.getOption("include")]    
    for path in allpaths:
        path = os.path.normpath(path)
        failure = False
        try:
            s = Session(dir=path, chat_user=plumbing.getOption("chat_user"), chat_ai=plumbing.getOption("chat_ai"), additional_keys=plumbing.getOption("var_file"), tools_forbidden=plumbing.getOption("tools_forbidden"))
            break
        except FileNotFoundError as e:
            # session will throw if path is not valid, we keep going through the includes
            failure = e


    if failure:
        return "error: " + str(failure)

    # constructing new session worked
    if not(keep):
        plumbing.session = s
    else:
        # something like /switch happened, we want to keep some stuff
        plumbing.session.merge(s)

    w = ""
    # try to load config.json if present
    configpath = path + "/config.json"
    if os.path.isfile(configpath):
        w += load_config(plumbing, configpath, override=False) + "\n"
    plumbing.options["character_folder"] = path

    # this might be very useful for people to debug their chars, so we are a bit verbose here by default
    w += "Found vars " + ", ".join([k for k in plumbing.session.getVars().keys() if k not in Session.special_files]) + "\n"


    # by convention, the initial message is stored in initial_msg
    if plumbing.session.hasVar("initial_msg") and not(keep):
        plumbing.initial_print_flag = True

    # enable tools if any are found
    if plumbing.session.tools:
        plumbing.setOption("use_tools", True)
        w += "Tool dictionary generated from tools.py, setting use_tools to True. Beware, the AI will now call functions.\n"
        if (callback := plumbing.getOption("tools_inject_dependency_function")):
            if plumbing.session.tools_module             is not None:
                callback(plumbing.session.tools_module            )

        if plumbing.getOption("tools_inject_ghostbox"):
            if plumbing.session.tools_module             is not None:
                plumbing.session.tools_module._ghostbox_plumbing = plumbing

            
        if plumbing.getOption("verbose"):
            w += "Dumping tool dictionary. Run with --no-verbose to disable this."
            w += json.dumps([tool.model_dump() for tool in plumbing.session.tools], indent=4) + "\n"
        else:
            w += "AI tools: " + ", ".join([t.function.name for t in plumbing.session.tools]) + "\n"


    # hide if option is set
    if plumbing.getOption("hide"):
        hide_some_output(plumbing)

    w += "Ok. Loaded " + path
    return w


def load_config(plumbing, filepath, override=True, protected_keys: List[str]=[]) -> str:
    try:
        w = open(filepath, "r").read()
    except Exception as e:
        return str(e)
    err = plumbing.loadConfig(w, override=override, protected_keys=protected_keys)
    if err:
        return err
    return "Loaded config " + filepath
    


def hide_some_output(plumbing):
    plumbing.options["cli_prompt"] = "\n"
    plumbing.options["audio_show_transcript"] = False
    plumbing.options["tts_subtitles"] = False
    #plumbing.options["stream"] = False
    plumbing.options["chat_show_ai_prompt"] = False
    plumbing.options["color"] = False
    

def toggle_tts(plumbing) -> str:
    prog = plumbing
    prog.options["tts"] = not(prog.options["tts"])
    w = ""
    if prog.options["tts"]:
        err = prog.initializeTTS()
        if err:
            return err
        w += "Try /hide for a more immersive experience.\n"
        prog.setOption("stream_flush", "sentence")
    else:
        # disabled tts
        prog.tts.close()
        prog.tts = None
    return w + "TTS " + {True : "on.", False : "off."}[prog.options["tts"]]
    


def all_chars(prog) -> List[str]:
    allchars = []
    for dir in prog.getOption("include"):
        if not(os.path.isdir(dir)):
            printerr("warning: Include path '" + dir + "' is not a directory.")
            continue
        for charpath in glob.glob(dir + "/*"):
            if os.path.isfile(charpath):
                continue
            allchars.append(os.path.split(charpath)[1])

    return allchars
