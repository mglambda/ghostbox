import os, datetime, sys, glob
import pyperclip
from typing import List,Dict


# Dependency Injection
# If tools_inject_ghostbox is true, the following identifier will point to a running ghostbox Plumbing instance
# _ghostbox_plumbing

file = os.path.expanduser("/home/butterscotch/butterscotch.org")
file_dir = os.path.expanduser("/home/butterscotch/files")
out_dir = os.path.expanduser("/home/butterscotch/out")


def directly_answer():
    """Calls an AI chatbot  to generate a response given the conversation history. Use this tool when no other tool is applicable."""
    return []

def suspend_self() -> Dict:
    """Suspend an AI chatbot's activity for an indeterminate amount of time. Use this when the user signals you to stop or be quiet for a while. Don't worry, you will be brought back.
    :return: A dict containing information about the suspension status."""
    # injection _ghostbox_plumbing
    
    # right now all this does is stop the AI from listening for a while
    # until the activation word is heard again, if user has an activation word set
    # alternatively user can still ask to wake up by text
    _ghostbox_plumbing.suspendTranscription()
    me = _ghostbox_plumbing.getOption("chat_ai")
    if (phrase := _ghostbox_plumbing.getOption("audio_activation_phrase")) != "":
        phrase_msg = ", or say '" + phrase + "' to get their attention."
    else:
        phrase_msg = "."
    _ghostbox_plumbing.console(me + " has gone to sleep. To resume listening, ask them by text to wake up" + phrase_msg)

    r = "Suspending operation."        
    if phrase:
        r += " Maybe tell the user that they can bring you back by saying '" + phrase + "'"
    return {"status": r}

def wake_up() -> str:
    """Unsuspend an AI chatbot.
    :return: A message confirming your unsuspension.""" 
    # injection _ghostbox_plumbing
    _ghostbox_plumbing.unsuspendTranscription()
    me = _ghostbox_plumbing.getOption("chat_ai")
    _ghostbox_plumbing.console(me + " has woken themselves up.")
    return "You have been unsuspended."
    
def take_note(label : str, text : str) -> dict:
    """Take down a note which will be written to a file on the hard disk." 
    :param label: A short label or heading for the note.
    :param text: The note to save"""
    global file
    try:
        if os.path.isfile(file):
            f = open(file, "a")
        else:
            f = open(file, "w")
        f.write("* " + label + "\ndate: " + datetime.datetime.now().isoformat() + "\n" + text + "\n")
        f.close()
    except:
        return { "status": "Couldn't save note.",
                 "error_message" : traceback.format_exc()}
    return {"status" : "Successfully saved note.",
            "note label" : label,
            "note text" : text}



def read_notes() -> dict:
    """Read the users notes."""
    global file
    if not(os.path.isfile(file)):
        return {"status" : "Failed to read notes.",
                 "error_msg" : "File not found."}
    ws = open(file, "r").read().split("\n*")
    d = {"status" : "Successfully read notes."}
    for i in range(len(ws)):
        if ws[i].strip() == "":
            continue
        vs = ws[i].split("\n")
        try:
            note_data = {"label" : vs[0],
                         "date" : vs[1].replace("date: ", "") if vs[1].startswith("date: ") else "",
                         "text" : vs[2:] if vs[1].startswith("date: ") else vs[1:]}
        except:
            print("warning: Syntax error in butterscotch notes, offending note: " + ws[i], file=sys.stderr)
            continue
        d["note " + str(i)] = note_data
    return d
    
def read_files() -> dict:
    """Read one or more files that the user wants to show you. Files will be retrieved from disk as well as from the user's clipboard."""
    if not(os.path.isdir(file_dir)):
        return {"status" : "Failed to read files.",
                "error_msg" : "Butterscotch directory does not exist."}

    clipboard = pyperclip.paste()
    clipboard_key = "*clipboard content*"
    file_list = glob.glob(file_dir + "/*")
    if file_list == []:
        return {"status" : "Failed to read files.",
                "error_msg" : "No files available. Directory is empty.",
                clipboard_key : clipboard}
    
    files = []
    for f in file_list:
        name = os.path.basename(f)
        try:
            w = open(f, "r").read()
        except:
            print("warning: Couldn't read file " + name + " while in butterscotch's directory.", file=sys.stderr)
            continue
        files.append({"filename" : name,
                     "content" : w})
    return {"status" : "Successfully read files.",
            "files" : files,
            clipboard_key : clipboard}

def save_file(filename : str, content : str) -> dict:
    """Write arbitrary data to a file on disk.
    :param filename: The name of the file that will be saved.
    :param content: The contents of the file as a string."""
    if not(os.path.isdir(out_dir)):
        return {"status" : "Failed to save file.",
                "error_msg" : "Output directory does not exist or is a file."}

    fullname = out_dir + "/" + filename
    if os.path.isfile(fullname):
        return {"status" : "Failed to save file.",
                "error_msg" : "File '" + filename + "' already exists."}

    try:
        f = open(fullname, "w")
        f.write(content)
        f.close()
    except:
        return {"status" : "Failed to save file.",
                "error_msg" : "Exception during opening/writing. Here's the backtrace: " + traceback.format_exc()}

    return {"status" : "Successfully saved file '" + filename + "'."}


def search_web(keywords: str) -> List[Dict[str,str]]:
    """Perform a web search using specified keywords. Use this tool when the user asks to google something.
    :param keywords: Terms to search for.
    :return: Search results as a dictionary."""
    from duckduckgo_search import DDGS
    import traceback
    # injection _ghostbox_plumbing
    _ghostbox_plumbing.console_me(" is searching the web for '" + keywords + "' ...")
    
    try:
        return DDGS().text(keywords, max_results=3, safesearch="off")
    except:
        return traceback.format_exc()

def visit_website(url: str) -> str:
    """Retrieves the contents of a website in markdown syntax.
    :param url: The http or https url to visit.
    :return: The website contents."""
    import markdownify
    import requests
    # injection _ghostbox_plumbing
    _ghostbox_plumbing.console_me(" is visiting " + url + " ...")
    
    r = requests.get(url)
    if r.status_code != 200:
        return "error: Couldn't visit webpage '" + url + "': status code " + str(r.status_code) + "'"
    m = markdownify.MarkdownConverter()
    markdown = m.convert(r.text)
    w = "".join(markdown.replace(" \n", "\n").split("\n\n"))
    return w


def shell_command(command: str, stdin: str=None, cwd=None) -> Dict:
    """Execute a shell command.
    :param command: A string which will be executed as if typed at a bash prompt.
    :param stdin: An optional string that will be fed to the invoked program's standard input.
    :param cwd: An optional string giving the current working directory to execute the command in.
    :return: A dictionary including stdout, stderr, and a status code."""
    import subprocess
    import traceback
    import getpass
    # injection _ghostbox_plumbing
    prog = _ghostbox_plumbing
    chat_ai = prog.getOption("chat_ai")
    if not(prog.getOption("tools_unprotected_shell_access")):
        if getpass.getuser() != chat_ai:
            msg = "Prevented " + chat_ai + " from running shell command\n  `" + command + "`\nsince they are not running as user `" + chat_ai + "` but `" + getpass.getuser() + "` instead.\nThis is a safety measure. Either create a " + chat_ai + " user in your system and \nstart ghostbox while logged in as them , or do `/set tools_unprotected_shell_access True`.\nUnprotected shell access may lead to data loss or worse. You have been warned."
            prog.console("warning: " + msg)
            return {"error": msg}

        # llms love to fill in optional vars with '' etc for some reason, but subprocess doesn't like that
    cwd = cwd if cwd else None
    stdin = stdin if stdin else None
    try:
        prog.console_me(" is executing `" + command + "` ...")
        r = subprocess.run(command,
                           text=True,
                           capture_output=True,
                           cwd=cwd,
                           shell=True)
    except:
        msg = traceback.format_exc()
        return {"error": msg}
    
    return {"return_code": r.returncode,
            "stdout" : r.stdout,
            "stderr": r.stderr}
