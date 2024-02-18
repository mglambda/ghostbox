import os, getpass, shutil, base64, requests, re, csv
from colorama import Fore, Back, Style
import appdirs
import sys
from functools import *

def getErrorPrefix():
    return " # "


def stringToColor(w):
    """Maps normal strings like 'red' to ANSI control codes using colorama package."""
    w = w.upper()
    for color in Fore.__dict__.keys():
        if w == color:
            return Fore.__dict__[color]
    return Fore.RESET

def stringToStyle(w):
    """Maps normal strings to ANSI control codes for style, like 'bright' etc. using colorama."""
    w = w.upper()
    for s in Style.__dict__.keys():
        if w == s:
            return Style.__dict__[s]
    return Style.RESET_ALL

def wrapColorStyle(w, color, style):
    return style + color + w + Fore.RESET + Style.RESET_ALL

def printerr(w, prefix=getErrorPrefix(), color=Fore.GREEN):
    if w == "":
        return

    if w.startswith("error:"):
        color = Fore.RED

    if w.startswith("warning:"):
        color = Fore.YELLOW
    
    # prepend all lines with prefix
    ws = w.split("\n")
    w = ("\n" + prefix).join(ws)
    print(color + prefix + w + Fore.RESET, file=sys.stderr)
    

def getArgument(argname, argv):
    ws = argv.split(argname)
    if len(ws) < 2:
        return None
    return ws[1].split(" ")[1]

def trimOn(stopword, w):
    return w.split(stopword)[0]

def trimChatUser(chatuser, w):
    if chatuser:
        return trimOn(mkChatPrompt(chatuser), trimOn(mkChatPrompt(chatuser).strip(), w))
    return w


def assertNotStartWith(assertion, w):
    # ensures w doesn't start with assertion
    l = len(assertion)
    if w.startswith(assertion):
        return w[l:]
    return w

def assertStartWith(assertion, w):
    # makes sure w starts with assertion. This is intended for making sure strings start with a chat prompt, i.e. Bob: bla bla bla, without duplicating it, as in Bob: Bob: bla bla
    if not(w.startswith(assertion)):
        return assertion + w
    return w


def mkChatPrompt(username, space=True):
    # turns USERNAME into USERNAME:, or, iuf we decide to change it, maybe <USERNAME> etc.
    if username == "":
        return ""
    if space:
        return username + ": "
    return username + ":"

def ensureColonSpace(usertxt, generatedtxt):
    """So here's the problem: Trailing spaces in the prompt mess with tokenization and force the backend to create emoticons, which isn't always what we want.
    However, for chat mode, trailing a colon (:) means the backend will immediately put a char behind it, which looks ugly. There doesn't seem to be a clean way to fix that, short of retraining the tokenizer. So we let it generate text behind the colon, and then add a space in this step manually. This is complicated by the way we split user and generated text.
    usertxt - User supplied text, which may end in something like 'Gary:'
generatedtxt - Text generated by backend. This immediately follows usertxt for any given pormpt / backend interaction. It may or may not start with a newline.
    Returns - The new usertxt as a string, possibly with a newline added to it."""
    if generatedtxt.startswith(" "):
        return usertxt
    
    if usertxt.endswith(":"):
        return usertxt + " "
    return usertxt

        
        
   

def ensureFirstLineSpaceAfterColon(w):
    # yes its a ridiculous name but at least it's descriptive
    if w == "":
        return w

    if len(w) <= 2:
        if w == "::":
            return ": :"
        elif w.endswith(":"):
            return w + " "
        


    
    ws = w.split("\n")
    v = ws[0]
    for i in range(0, len(v)-1):
        if v[i] == ":":
            if v[i+1] == " ":
                return w
            else:
                ws[0] = v[:i+1] + " " + v[i+1:]
                break
    return "\n".join(ws)
                
            

def filterPrompt(prompt, w):
    # filters out prompts like "Assistant: " at the start of a line, which can sometimes be generated by the LLM on their own
    return w.replace("\n" + prompt, "\n")



def filterLonelyPrompt(prompt, w):
    # this will filter out prompts like "Assistant: ", but only if it's the only thing on the line. This can happen after trimming. Also matches the prompt a little fuzzy, since sometimes only part of the prompt remains.
    ws = w.split("\n")
    return "\n".join(filter(lambda v: not(v in prompt), ws))

def discardFirstLine(w):
    return "\n".join(w.split("\n")[1:])

def filterLineBeginsWith(target, w):
    """Returns w with all lines that start with target removed. Matches target a little fuzzy."""
    acc = []
    targets = [target, " " + target, target + " "]
    for line in w.split("\n"):
        dirty = False
        for t in targets:
            if line.startswith(t):
                dirty = True
        if not(dirty):
            acc.append(line)
    return "\n".join(acc)



def saveFile(filename, w, depth=0):
# saves w in filename, but won't overwrite existing files, appending .new; returns the successful filename, if at all possible
    if depth > 10:
        return "" # give up
    
    if os.path.isfile(filename):
        parts = filename.split(".")
        if len(parts) > 1:
            newfilename = ".".join([parts[0], "new"] + parts[1:])
        else:
            newfilename = filename + ".new"
        return saveFile(newfilename, w, depth=depth+1)

    f = open(filename, "w")
    f.write(w)
    f.flush()
    return filename

        
        


def stripLeadingHyphens(w):
    #FIXME: this is hacky
    w = w.split("=")[0]
    
    if w.startswith("--"):
        return w[2:]

    if w.startswith("-"):
        return w[1:]

    return w


def userConfigFile(force=False):
    # return location of ~/.ghostbox.conf.json, or "" if not found
    userconf = "ghostbox.conf"
    path = appdirs.user_config_dir() + "/" + userconf
    if os.path.isfile(path) or force:
        return path
    return ""

def userCharDir():
    return appdirs.user_data_dir() + "/ghostbox/chars"

def ghostboxdir():
    return appdirs.user_data_dir() + "/ghostbox"

def userTemplateDir():
    return ghostboxdir() + "/templates"

def inputChoice(msg, choices):
    while True:
        w = input(msg)
        if w in choices:
            return w
        

def userSetup():
    if not(os.path.isfile(userConfigFile())):
        print("Creating config file " + userConfigFile(force=True))
        f = open(userConfigFile(force=True), "w")
        f.write('{\n"chat_user" : "' + getpass.getuser() +'"\n}\n')
        f.flush()
        
    if not(os.path.isdir(userCharDir())):
        print("Creating char dir " + userCharDir())
        os.makedirs(userCharDir())

    # try copying some example chars
    chars = "dolphin dolphin-kitten joshu minsk".split(" ")
    copyEntity("char", "chars", chars)
    templates = "chat-ml alpaca raw mistral user-assistant-newline".split(" ")
    copyEntity("template", "templates", templates)
    

def copyEntity(entitynoun, entitydir, entities):
    chars = entities
    choice = ""
    try:
        for char in chars:
            chardir = ghostboxdir() + "/" + entitydir + "/" + char
            if os.path.isdir(chardir) and choice != "a":
                choice = inputChoice(chardir + " exists. Overwrite? (y/n/a): ", "y n a".split(" "))
                if choice == "n":
                    continue
            print("Installing " + entitynoun + " " + char)
            shutil.copytree(entitydir + "/" + char, chardir, dirs_exist_ok=True)
    except:
        print("Warning: Couldn't copy example " + entitynoun + "s.")
          
def getJSONGrammar():
    return r"""root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"""




def loadImageData(image_path):
    with open(image_path, 'rb') as image_file:
        base64_bytes = base64.b64encode(image_file.read())
        return base64_bytes

def packageImageDataLlamacpp(data_base64, id):
    return {"data" : data_base64.decode("utf-8"), "id" : id}
    
def mkImageEmbeddingString(image_id):
    return "[img-" + str(image_id) + "]"


def maybeReadInt(w):
    try:
        n = int(w)
    except:
        return None
    return n
    
def isImageFile(file):
    # good enuff
    return file.endswith(".png")

    

def dirtyGetJSON(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return {}

def replaceFromDict(w, d, key_func=lambda k: k):
    return reduce(lambda v, pair: v.replace(key_func(pair[0]), pair[1]), d.items(), w)

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def stripANSI(w):
    return ansi_escape.sub('', w)
    


def getLayersFile():
    return appdirs.user_config_dir() + "/llm_layers"

def loadLayersFile():
    """Returns a list of dictionaries, one for each row in the layers file."""
    f = open(getLayersFile(), "r")
    return list(csv.DictReader(filter(lambda row: row[0] != "#", f), delimiter="\t"))
