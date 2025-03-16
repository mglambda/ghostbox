#!/usr/bin/env python
# A simple example of streaming generated text to stdout and TTS
# This is the most dead-simple version, where you let ghostbox do the most heavy lifting.
# For a more in-depth example, see streaming_quiet.py

import ghostbox, threading

box = ghostbox.from_generic(
    character_folder="ghost",
    tts=True,  # we want speech output
    tts_model="kokoro",
    tts_voice="am_santa",
    stderr=False,  # it's a CLI program, so we don't want clutter
)

# we will use this flag to signal whem streaming is done
done = threading.Event()


def done_generating(w: str) -> None:
    """Callback for when we are done streaming."""
    global done
    # we could do something with w here
    # it contains the entire generation
    # but it was already printed/spoken, so we're done
    done.set()


# start the actual streaming
box.text_stream(
    "Can you explain the basics of TCP/IP to me? Give both an explain-like-I'm-five version and one where you think I have technical expertise.",
    chunk_callback=lambda w: None,  # do nothing here. Ghostbox will print/speak the chunks
    generation_callback=done_generating,
)  # this will be called when streaming is done
# Now, text is streamed to stdout and the tts backend simultaneously.
# You can prevent this by setting quiet = True (see also examples/streaming_quiet.py)
# You may notice that text appears sentence-by-sentence, and not token-by-token.
# streaming individual tokens to the tts engine is a bad idea, so ghostbox does some chunking in the background.
# this behaviour is determined by the stream_flush option. It is set to 'sentence' automatically when the tts is active.
# It also determines what will be passed to the chunk_callback. Setting stream_flush to 'token' will give you the most power.

# now we need to keep the program from exiting right away
done.wait()
