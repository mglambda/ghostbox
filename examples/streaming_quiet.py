#!/usr/bin/env python
# This is a DIY, roll your own example of streaming output.
# In this example, we don't let ghostbox do the output and tts streaming,
# but do it ourselves instead.
# This is supposed to illustrate how to use ghostbox to output to resources of your own
# since you probably want to do more interesting things than printing to stdout
# and probably want more fine grained control over the tts

import ghostbox, threading, tempfile

box = ghostbox.from_generic(
    character_folder="ghost",
     stderr=False, # no clutter
    tts=True,  # we still want a tts engine active
    tts_model="kokoro",
    tts_voice="af_sky",
    quiet=True,  # **important** this keeps ghostbox from printing and speaking on its own
    stream_flush="token",  # this ensures that chunk_callback gets called on each token, not each sentence
)

# let's imagine we have some resource where we want to save generations
resource_handle = tempfile.NamedTemporaryFile(mode="w", delete=False)

# this flag signals when we're done streaming
done = threading.Event()


def save_generation(generation: str) -> None:
    """Called when streaming is done with the entire generation as argument."""
    # here we save/insert into database/send over network our finished generation
    resource_handle.write(generation)
    resource_handle.flush()
    # and we're done streaming
    done.set()

    # now for handling individual tokens


buffer = ""


def process_token(token: str) -> None:
    """This will be called on every token as they are being streamed."""
    # let's do minimal chunking and send to tts
    global buffer
    buffer += token
    for punct in ".!:?":
        if punct in buffer:
            # I did say minimal
            box.tts_say(buffer, interrupt=False)
            buffer = ""
            break
    # we could also print to stdout here
    # with e.g. print(token, end="", flush=True)
    # but for this example we will print the entire generation at once at the end


# start the actual prompt and streaming generation
box.text_stream(
    "I used to play starcraft with my friends over IPX, but I never understood what that protocol actually is. Can you explain what IPX is in comparison to TCP? Also, please talk like a Protoss while you explain.",
    chunk_callback=process_token,  # called on every token
    generation_callback=save_generation,
)  # called on everything at the end

# need to keep the program from exiting
done.wait()

# now, since we saved the generation
with open(resource_handle.name, "r") as f:
    print(
        "## What is IPX, a Protoss perspective\n   or: How I learned to stop worrying and love the overmind\n\n"
    )
    print(f.read())
