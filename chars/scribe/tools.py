import os


def take_note(text : str):
    """Take down a note which will be written to a file on the hard disk.
    :param text: The note to save"""
    file = os.path.expanduser("~/ai_scribe_note.txt")
    if os.path.isfile(file):
        f = open(file, "a")
    else:
        f = open(file, "w")
    f.write(text + "\n")
    f.close()

