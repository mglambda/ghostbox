import os


def take_note(text):
    """Take down a note which will be written to a file on the hard disk."""
    file = os.path.expanduser("~/ai_scribe_note.txt")
    if os.path.isfile(file):
        f = open(file, "a")
    else:
        f = open(file, "w")
    f.write(text + "\n")
    f.close()
    return ""
