import os, datetime, sys


file = os.path.expanduser("~/butterscotch.org")

def directly_answer():
    """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history"""
    return []
    
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
    

