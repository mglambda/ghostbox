from session import Session

def newSession(program, w):
    program.session = Session(dir=w, chat_user=program.chat_user)
    w = "Ok. Loaded " + w + "\n\n"
    w += program.session.initial_prompt
    return w
