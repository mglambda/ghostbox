import os

file = os.path.expanduser("~/ai_scribe_note.txt")

def take_note(text : str):
    """Take down a note which will be written to a file on the hard disk.
    :param text: The note to save"""
    global file
    if os.path.isfile(file):
        f = open(file, "a")
    else:
        f = open(file, "w")
    f.write(text + "\n")
    f.close()

def read_notes():
    """Read the users notes."""
    global file
    if not(os.path.isfile(file)):
        return "*** error: Could not read notes: File not found. ***"
    return open(file, "r").read()

    
