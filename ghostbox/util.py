import os
import sys

def printerr(w, prefix=" # "):
    if w == "":
        return
    
    print(prefix + w, file=sys.stderr)
    
def trimIncompleteSentence(w):
    if w == "":
        return w
    
    stopchars = '! . ? :'.split(" ")
    for i in range(len(w)-1, -1, -1):
        if w[i] in stopchars:
            break

    if i == 0:
        return w
    return w[:i+1]




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


def mkChatPrompt(username):
    # turns USERNAME into USERNAME:, or, iuf we decide to change it, maybe <USERNAME> etc.
    if username == "":
        return ""
    return username + ": "

def filterPrompt(prompt, w):
    # filters out prompts like "Assistant: " at the start of a line, which can sometimes be generated by the LLM on their own
    return w.replace("\n" + prompt, "\n")



def filterLonelyPrompt(prompt, w):
    # this will filter out prompts like "Assistant: ", but only if it's the only thing on the line. This can happen after trimming. Also matches the prompt a little fuzzy, since sometimes only part of the prompt remains.
    ws = w.split("\n")
    return "\n".join(filter(lambda v: not(v in prompt), ws))
    

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
