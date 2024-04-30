

def take_note(text):
    """Take down a note which will be written to a file on the hard disk."""
    f = open("/hoem/marius/ainote.txt", "a")
    f.write(note)
    f.close()
    
