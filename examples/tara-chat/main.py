#!/usr/bin/env python
import os, time, traceback, random, threading, json, sys
import shutil
import ghostbox
from ghostbox import Ghostbox

from pydantic import BaseModel, Field

local_dir = os.path.dirname(os.path.abspath(__file__))
tara_dir = os.path.join(local_dir, "tara")
history_file = os.path.join(local_dir, "chat_history.json")
memory_file = os.path.join(tara_dir, "memory")
debug_mode = True

def debug(w):
    if debug_mode:
        print(w, file=sys.stderr)


def ensure_exists(filename: str) -> None:
    if os.path.isdir(filename):
        raise RuntimeError(f"fatal error: {filename} is a directory.")

    if not (os.path.isfile(filename)):
        with open(filename, "w") as f:
            f.write("")


class Config(BaseModel):
    # time the AI will tolerate non-interaction from the user before receiving a system note to drive the conversation onward
    awkward_silence_seconds: float = 40.0
    awkward_silence_std_deviation: float = 3.5
    # with each failed attempt at restarting conversation, how much more time should the ai wait before initiating again
    awkward_silence_increment: float = 15.0
    # how many times should the AI try to push before going dormant
    max_conversation_start_attempts: int = 3


class State(BaseModel, arbitrary_types_allowed=True):
    last_interaction_time: float = Field(default_factory=lambda: time.time())
    running: bool = True
    # how often the AI has tried to restart conversation consecutively
    conversation_start_attempts: int = 0
    # suspended means tara is still running but will not start interactions on her own
    suspended: bool = False

    # the interaction timer cancels if this is set
    wait_for_activity_flag: threading.Event = Field(
        default_factory=lambda: threading.Event()
    )

    # similarly, a speaking user will block the nudging
    user_is_speaking: threading.Event = Field(default_factory=lambda: threading.Event())

    def suspend(self, box):
        system_msg(
            f"Due to lack of interaction by the user, you are suspending your operation for a while, until the user comes back. Please generate a short message indicating this, in a way that fits naturally into the conversation.",
            box,
        )

    def reset_timer(self) -> None:
        t = time.time()
        delta = t - self.last_interaction_time
        debug(f"Timer reset after {delta}")
        self.last_interaction_time = time.time()

    def user_activity(self):
        self.conversation_start_attempts = 0
        self.wait_for_activity_flag.clear()


class Program(BaseModel, arbitrary_types_allowed=True):
    box: Ghostbox
    config: Config = Field(default_factory=lambda: Config())
    state: State = Field(default_factory=lambda: State())


def system_msg(w: str, box: Ghostbox) -> str:
    """Sends a system msg to the backend, which is formatted differently from user messages.
    This function will block and then return the backend  generation."""
    # we do it using text_stream instead of using box.text, because we still want the tokens to be streamed to the tts.
    done = threading.Event()
    done.clear()
    tmp = ""

    def finish(w):
        nonlocal done
        nonlocal tmp
        tmp = w
        done.set()

    time_str = time.strftime("%c")
    msg = f"[{time_str}]\n[System Message: {w}]\n"
    box.text_stream(msg, chunk_callback=lambda w: w, generation_callback=finish)
    done.wait()
    return tmp


def modify_transcription(text: str, prog: Program) -> str:
    """Add some context information to a users message after it was transcribed."""
    # since we got a transcription the user interacted
    state = prog.state
    state.reset_timer()
    # and is done speaking
    state.user_is_speaking.clear()
    

    time_str = time.strftime("%c")
    w = f"[{time_str}]\n{text}"
    return w


def save_and_exit(prog: Program) -> None:
    print("Saving chat history...")
    prog.box.save(history_file)
    # this next interaction won't be part of the saved history
    with open(memory_file, "r") as f:
        memory = f.read().strip()

    new_memory = system_msg(
        f"The user has exited. You may     now edit a part of your own system prompt which we will call your 'memory'. Below is your current 'memory':\n```\n{memory}\n```\nPlease output the new 'memory', or the same 'memory' as above if you wish for it to remain unchanged. Be exact and don't add any conversational flourishes. Your message will not be shown to the user, but added to the system prompt verbatim.",
        prog.box,
    )

    with open(memory_file, "w") as f:
        # the AI sometimes adds backticks, sometimes not
        # so we just remove them in any case
        new_memory = new_memory.replace("```", "").strip()
        if new_memory == "":
            answer = input("AI wants to wipe memory. Really delete it (sometimes this is by mistake)? y/n")
            if answer != "y":
                return
        f.write(new_memory)


def spawn_timer(prog: Program) -> None:
    """Spawns a timer that counts time since last interaction, pushing the AI if it exceeds a limit of awkward silence time."""
    box, state, config = prog.box, prog.state, prog.config
    state.reset_timer()
    state.user_activity()

    # if, for whatever reason, we already have a timer going, bail out
    if state.wait_for_activity_flag.is_set():
        return

    def count():
        # the limit is fixed per wait loop
        limit = random.normalvariate(
            mu=config.awkward_silence_seconds
            + (state.conversation_start_attempts * config.awkward_silence_increment),
            sigma=config.awkward_silence_std_deviation,
        )
        state.wait_for_activity_flag.set()
        while state.wait_for_activity_flag.is_set():
            if state.suspended:
                break

            if state.user_is_speaking.is_set():
                break
            
            # this functions as time.sleep(1)
            # it also prevents the system_msg from interrupting tara
            if box.tts_is_speaking():
                debug("Resetting due to AI speaking.")
                state.reset_timer()
            box.tts_wait(minimum=1.0)


            delta = time.time() - state.last_interaction_time
            if delta > limit:
                debug(f"Reached limit {delta}")
                delta = round(delta, 2)
                state.conversation_start_attempts += 1
                if (
                    state.conversation_start_attempts
                    >= config.max_conversation_start_attempts
                ):
                    system_msg(
                        "You are suspending yourself due to user inactivity. You may respond with a message indicating this to {{chat_user}}. Don't worry, you will be back.",
                        box,
                    )
                elif state.conversation_start_attempts == (
                    config.max_conversation_start_attempts - 1
                ):
                    system_msg(
                        f"User has not interacted in {delta} seconds. Give a small response indicating that you are still conversing, or gently try to keep the conversation going. This is the last system message before you will be suspended.",
                        box,
                    )
                else:
                    system_msg(
                        f"The user has not interacted in {delta} seconds. Give a short, context appropriate response that indicates you are still engaged in the conversation.",
                        box,
                    )

    # start the timer
    t = threading.Thread(target=count, daemon=True)
    t.start()


def audio_activation(prog: Program):
    prog.state.user_is_speaking.set()
    
    
def main():
    ensure_exists(memory_file)

    box = ghostbox.from_generic(
        character_folder=tara_dir,
        tts=True,
        # tara has more options set in her config.json
        audio=True,
        #verbose=True,
        # stderr=False,
        # stdout=False
    )
    prog = Program(box=box)

    state = prog.state

    if os.path.isfile(history_file):
        box.load(history_file)
        # sometimes things go wrong you know
        shutil.copy(history_file, history_file + ".backup")

    box.audio_on_transcription(lambda w: modify_transcription(w, prog))
    box.audio_on_activation(lambda: audio_activation(prog))
    box.audio_on_generation(lambda w: state.user_activity())
    box.on_interaction(state.reset_timer)
    box.on_interaction_finished(lambda: spawn_timer(prog))

    print(
        "Speak into the microphone to interact with tara. Control+d or type 'q' to quit. Other inputs will be sent to tara."
    )

    state.reset_timer()
    while state.running:
        try:
            w = input()
        except EOFError:
            save_and_exit(prog)
            state.running = False
            continue

        if w == "q":
            save_and_exit(prog)
            state.running = False
        else:
            state.user_activity()
            system_msg(f"The following message was typed by user." + f"\n{w}", box)


if __name__ == "__main__":
    main()
