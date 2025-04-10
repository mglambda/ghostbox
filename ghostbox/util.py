from typing import *
import os, getpass, shutil, base64, requests, re, csv, glob, time, traceback

# removed tortoise dependency because it will require torch, which import multiprocess, which won't work with renpy
# FIXME: not a big deal because tortoise and all tts are spawned with subprocess. However, we will have to find a better way to get the voices.
# import tortoise.utils.audio

from colorama import Fore, Back, Style
import appdirs
import sys
from functools import *



def getAITime() -> str:
    """Returns current time in a format that is unlikely to invalidate the cache when put into the system prompt of an AI."""
    return time.strftime("%A, %B %d, %Y")
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


# this can be used to modify printerr behaviour. It can be a function that accepts one argument -> the to be printed string
printerr_callback = None
printerr_disabled = False


def printerr(w, prefix=getErrorPrefix(), color=Fore.GREEN):
    global printerr_disabled
    if printerr_disabled:
        return

    if w == "":
        return

    if w.startswith("error:"):
        color = Fore.RED

    if w.startswith("warning:"):
        color = Fore.YELLOW

    # prepend all lines with prefix
    ws = w.split("\n")
    new_w = ("\n" + prefix).join(ws)
    formatted_w = color + prefix + new_w + Fore.RESET
    print(formatted_w, file=sys.stderr)
    if printerr_callback is not None:
        printerr_callback(color + w + Fore.RESET)


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
    if not (w.startswith(assertion)):
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
    for i in range(0, len(v) - 1):
        if v[i] == ":":
            if v[i + 1] == " ":
                return w
            else:
                ws[0] = v[: i + 1] + " " + v[i + 1 :]
                break
    return "\n".join(ws)


def filterPrompt(prompt, w):
    # filters out prompts like "Assistant: " at the start of a line, which can sometimes be generated by the LLM on their own
    return w.replace("\n" + prompt, "\n")


def filterLonelyPrompt(prompt, w):
    # this will filter out prompts like "Assistant: ", but only if it's the only thing on the line. This can happen after trimming. Also matches the prompt a little fuzzy, since sometimes only part of the prompt remains.
    ws = w.split("\n")
    return "\n".join(filter(lambda v: not (v in prompt), ws))


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
        if not (dirty):
            acc.append(line)
    return "\n".join(acc)


def saveFile(filename: str, w:str, overwrite: bool, depth=0):
    # saves w in filename, but won't overwrite existing files, appending .new; returns the successful filename, if at all possible
    if depth > 10:
        #return ""  # give up
        raise RuntimeError(f"error: Cannot save '{filename}': Exceeded depths of appending .new")

    if os.path.isfile(filename) and not(overwrite):
        parts = filename.split(".")
        if len(parts) > 1:
            newfilename = ".".join([parts[0], "new"] + parts[1:])
        else:
            newfilename = filename + ".new"
        return saveFile(newfilename, w, depth=depth + 1)

    f = open(filename, "w")
    f.write(w)
    f.flush()
    return filename


def stripLeadingHyphens(w):
    # FIXME: this is hacky
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
    if not (os.path.isfile(userConfigFile())):
        print("Creating config file " + userConfigFile(force=True))
        f = open(userConfigFile(force=True), "w")
        f.write('{\n"chat_user" : "' + getpass.getuser() + '"\n}\n')
        f.flush()

    if not (os.path.isdir(userCharDir())):
        print("Creating char dir " + userCharDir())
        os.makedirs(userCharDir())

    # try copying some example chars
    chars = "dolphin dolphin-kitten joshu minsk scribe command-r".split(" ")
    copyEntity("char", "chars", chars)
    templates = "chat-ml alpaca raw mistral user-assistant-newline vacuna command-r llama3 phi3-instruct".split(
        " "
    )
    copyEntity("template", "templates", templates)


def copyEntity(entitynoun, entitydir, entities):
    chars = entities
    choice = ""
    try:
        for char in chars:
            chardir = ghostboxdir() + "/" + entitydir + "/" + char
            if os.path.isdir(chardir) and choice != "a":
                choice = inputChoice(
                    chardir + " exists. Overwrite? (y/n/a): ", "y n a".split(" ")
                )
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
    with open(image_path, "rb") as image_file:
        base64_bytes = base64.b64encode(image_file.read())
        return base64_bytes


def packageImageDataLlamacpp(data_base64, id):
    return {"data": data_base64.decode("utf-8"), "id": id}


# def repackImages(images):
#    """Takes the plumbing.images object and makes sure its base64 encoded."""
#    return [{"data" : data_base64.decode("utf-8"), "id" : id}


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


def getImageExtension(url, default="png"):
    ws = url.split(".")
    if len(ws) < 2:
        return default
    return ws[-1]


def dirtyGetJSON(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return {}


def replaceFromDict(w, d, key_func=lambda k: k):
    def replace_and_check(v, pair):
        v_new = v.replace(key_func(pair[0]), pair[1])
        if type(v_new) != str:
            return str(v_new)
        return v_new

    return reduce(replace_and_check, d.items(), w)


ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def stripANSI(w):
    return ansi_escape.sub("", w)


def getLayersFile():
    return appdirs.user_config_dir() + "/llm_layers"


def loadLayersFile():
    """Returns a list of dictionaries, one for each row in the layers file."""
    f = open(getLayersFile(), "r")
    return list(csv.DictReader(filter(lambda row: row[0] != "#", f), delimiter="\t"))


def envFromDict(d):
    """Returns a kstandard shell environment with variables added from the provided dictionary d."""
    return os.environ | {k: str(v) for (k, v) in d.items()}


def explodeIncludeDir(include, extradir):
    """So that we can turn 'char/' into ['/include/path/char/', 'char/'] etc."""
    include = os.path.normpath(include)
    extradir = os.path.normpath(extradir)
    return [include, extradir, include + "/" + extradir]


def ultraglob(include_dirs, specific_dir):
    # everyone hates nested listcomprehensions
    acc = []
    for include_dir in include_dirs:
        acc += [
            glob.glob(dir + "/*")
            for dir in explodeIncludeDir(include_dir, specific_dir)
        ]
    return reduce(lambda xs, ys: xs + ys, acc, [])


def getVoices(prog):
    """Returns a list of strings with voice names for a given Program object."""
    pollyvoices = "Lotte, Maxim, Ayanda, Salli, Ola, Arthur, Ida, Tomoko, Remi, Geraint, Miguel, Elin, Lisa, Giorgio, Marlene, Ines, Kajal, Zhiyu, Zeina, Suvi, Karl, Gwyneth, Joanna, Lucia, Cristiano, Astrid, Andres, Vicki, Mia, Vitoria, Bianca, Chantal, Raveena, Daniel, Amy, Liam, Ruth, Kevin, Brian, Russell, Aria, Matthew, Aditi, Zayd, Dora, Enrique, Hans, Danielle, Hiujin, Carmen, Sofie, Gregory, Ivy, Ewa, Maja, Gabrielle, Nicole, Filiz, Camila, Jacek, Thiago, Justin, Celine, Kazuha, Kendra, Arlet, Ricardo, Mads, Hannah, Mathieu, Lea, Sergio, Hala, Tatyana, Penelope, Naja, Olivia, Ruben, Laura, Takumi, Mizuki, Carla, Conchita, Jan, Kimberly, Liv, Adriano, Lupe, Joey, Pedro, Seoyeon, Emma, Niamh, Stephen".split(
        ", "
    )

    kokoro_voices = """af_alloy
af_aoede
af_bella
af_heart
af_jessica
af_kore
af_nicole
af_nova
af_river
af_sarah
af_sky
am_adam
am_echo
am_eric
am_fenrir
am_liam
am_michael
am_onyx
am_puck
am_santa
bf_alice
bf_emma
bf_isabella
bf_lily
bm_daniel
bm_fable
bm_george
bm_lewis
ef_dora
em_alex
em_santa
ff_siwis
hf_alpha
hf_beta
hm_omega
hm_psi
if_sara
im_nicola
jf_alpha
jf_gongitsune
jf_nezumi
jf_tebukuro
jm_kumo
pf_dora
pm_alex
pm_santa
zf_xiaobei
zf_xiaoni
zf_xiaoxiao
zf_xiaoyi
zm_yunjian
zm_yunxi
zm_yunxia
zm_yunyang""".split(
        "\n"
    )


    # hardcoded for random voice selection. user gets the full list from ghostbox-tts after about 3 seconds
    orpheus_voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    vs = []
    if prog.getOption("tts_model") == "polly":
        for voice in pollyvoices:
            vs.append(voice)
    elif prog.getOption("tts_program") == "ghostbox-tts-tortoise":
        # vs = list(tortoise.utils.audio.get_voices(extra_voice_dirs=list(filter(bool, [prog.getOption("tts_voice_dir")]))))
        # FIXME: find another way to get the list of voices
        vs = []
    elif prog.getOption("tts_model") == "kokoro":
        english_voices = [
            voice
            for voice in kokoro_voices
            if voice.startswith("af_")
            or voice.startswith("am_")
            or voice.startswith("bm_")
            or voice.startswith("bf_")
        ]
        if prog.getOption("tts_language") == "en":
            return english_voices
        elif prog.getOption("tts_language") == "":
            return kokoro_voices
        else:
            return [voice for voice in kokoro_voice if voice not in english_voices]

    elif prog.getOption("tts_model") == "orpheus":
        return orpheus_voices
    else:
        # for file in ultraglob(prog.getOption("include"), prog.getOption("tts_voice_dir")):
        vs = [
            os.path.split(file)[1]
            for file in glob.glob(prog.tryGetAbsVoiceDir() + "/*")
            if os.path.isfile(file)
        ]
    return vs


def time_ms():
    return round(time.time() * 1000)


def compose2(f, g):
    return lambda x: f(g(x))


def get_default_microphone_sample_rate(pyaudio_object=None) -> Optional[int]:
    import pyaudio
    if pyaudio_object is None:
        p = pyaudio.PyAudio()
    else:
        p = pyaudio_object
        
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    for i in range(0, num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            # This is a microphone device
            default_sample_rate = device_info.get('defaultSampleRate')
            if pyaudio_object is None:
                p.terminate()
            return int(default_sample_rate)
    
    if pyaudio_object is None:
        p.terminate()
    return None

def get_default_output_sample_rate(pyaudio_object: Optional[Any] = None) -> Optional[int]:
    import pyaudio
    """Uses PyAudio to return the default sample rate for the default output device."""
    try:
        if (p := pyaudio_object) is None:
            p = pyaudio.PyAudio()

        info = p.get_default_output_device_info()
        return int(info['defaultSampleRate'])
    except Exception as e:
        printerr(f"Error getting default sample rate: {e}")
        return None


def get_default_output_device_info(pyaudio_object: Optional[Any] = None) -> Dict[str, Any]:
    import pyaudio
    try:
        if (p := pyaudio_object) is None:
            p = pyaudio.PyAudio()

        info = p.get_default_output_device_info()
        return info
    except Exception as e:
        printerr(f"Error getting default output device info: {e}")
        return None
    
def is_output_format_supported(rate: float, channels=None, format=None, pyaudio_object=None) -> bool:
    """Checks wether the given parameters work with the default output device."""
    import pyaudio
    try:
        if (p := pyaudio_object) is None:
            p = pyaudio.PyAudio()

        info = get_default_output_device_info(p)
        return p.is_format_supported(rate,
                                     output_channels=channels if channels is not None else info["maxOutputChannels"],
                                     output_format=format if format is not None else pyaudio.paInt16,
                                     output_device=info["index"])
                                     
    except Exception as e:
        printerr("warning: Exception while determining supported output formats for default sound device.\n" + traceback.format_exc())
        return False

def convert_int16_to_float(audio_bytes: bytes):
    """Converts paInt16 bytes to a float32 NumPy array."""
    # Convert bytes to a NumPy array of int16
    import numpy as np
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    # Normalize to float32 (range -1.0 to 1.0)
    audio_float = audio_array.astype(np.float32) / 32768.0  # or 2**15 if signed

    return audio_float
    
    
