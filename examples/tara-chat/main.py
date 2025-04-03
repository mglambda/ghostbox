#!/usr/bin/env python
import os, time, traceback
import ghostbox

local_dir = os.path.dirname(os.path.abspath(__file__))
tara_dir = os.path.join(local_dir, "tara")
history_file = os.path.join(local_dir, "chat_history.json")
memory_file = os.path.join(tara_dir, "memory")
def system_msg(w: str) -> str:
    """Wraps a string in brackets and adds system tag.
    This is to mark parts of the user's message as coming from the system."""
    return f"[System Message: {w}]\n"


def modify_transcription(text: str) -> str:
    """Add some context information to a users message after it was transcribed."""
    time_str = time.strftime("%c")
    w = f"[{time_str}]\n{text}"
    return w



def save_and_exit(box: ghostbox.Ghostbox) -> None:
    box.save(history_file)
    # this next interaction won't be part of the saved history
    with open(memory_file, "r") as f:
        memory = f.read().strip()

    msg = system_msg(f"The user has exited. You may     now edit a part of your own system prompt. Below is the current system prompt:\n```\n{memory}\n```\nPlease output the new system prompt, or the same system prompt as above if you wish for it to remain unchanged. Be exact and don't add any conversational flourishes. Your message will not be shown to the user, but added to the system prompt verbatim.")
    new_memory = box.text(msg)
    with open(memory_file, "w") as f:
        f.write(new_memory)

def main():
    box = ghostbox.from_generic(character_folder=tara_dir,
                                tts=True,
                                # tara has more options set in her config.json
                                audio=True
                                #stderr=False,
                                #stdout=False
                                )
    if os.path.isfile(history_file):
        box.load(history_file)
        
    
    box.audio_on_transcription(modify_transcription)

    running = True
    print("Speak into the microphone to interact with tara. Control+d or type 'q' to quit. Other inputs will be sent to tara.")
    while running:
        try:
            w = input()
        except EOFError:
            save_and_exit(box)
            running = False
            
        if w == "q":
            save_and_exit(box)
            running = False
        else:
            box.text_stream(f"[following message was typed by user]\n{w}",
                            chunk_callback=lambda w: w,
                            generation_callback= lambda w: w)
            


        
if __name__ == "__main__":
    main()
