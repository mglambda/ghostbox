import os


file = os.path.expanduser("~/ai_scribe_note.txt")

def directly_answer():
    """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history"""
    return []
    
def take_note(text : str) -> dict:
    """Take down a note which will be written to a file on the hard disk.
    :param text: The note to save"""
    global file
    try:
        if os.path.isfile(file):
            f = open(file, "a")
        else:
            f = open(file, "w")
            f.write(text + "\n")
        f.close()
    except:
        return { "status": "Couldn't save note.",
                 "error_message" : traceback.format_exc()}
    return {"status" : "Successfully saved note."}

def read_notes() -> dict:
    """Read the users notes."""
    global file
    if not(os.path.isfile(file)):
        return {"status" : "Failed to read notes.",
                 "error_msg" : "File not found."}
    ws = open(file, "r").read().split("\n")
    d = {"status" : "Successfully read notes."}
    for i in range(len(ws)):
        d["note " + str(i)] = ws[i]
    return d
    

