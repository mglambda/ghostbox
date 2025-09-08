# Ghostbox

Do you love comprehensive LLM frameworks? Me neither.

I like things like this:

```python
import ghostbox
box = ghostbox.from_llamacpp(character_folder="unix_philosopher")

with box.options(temperature=0.6, samplers=["min_p", "temperature"]):
    answer = box.text("How do you make developing with LLMs easy and painless?")
    
box.tts_say(answer)
```

You would hear a voice saying "ghostbox". Probably.

## What it is
<a href="https://raw.github.com/mglambda/ghostbox/master/screenshots/terminal.png"><img alt="A screenshot displaying ghostbox used in a terminal." width="300" align="right" src="https://raw.github.com/mglambda/ghostbox/master/screenshots/terminal.png"></a>
Ghostbox is a python library and toolkit for querying LLMs (large language models), both locally hosted and remote. It let's you use AI independent of any particular provider and backend. It wants to make developing applications with tightly integrated AI as painless as possible, without tieing you down to some kind of framework.

Ghostbox ships with the `ghostbox` CLI program, a fully featured terminal client that let's you interact and chat with LLMs from the comfort of your own shell, as well as monitor and debug AIs that are running in your program.

It also includes `ghostbox-tts`, which allows text-to-speech synthesis with various SOTA speech models. This is coupled with the library, but serves as a standalone TTS client in its own right.

I wrote this because I wanted to build stuff with LLMs, while still understanding what's going on under the hood. And also because I wanted an actually good, blind accessible terminal client. Ghostbox is those things.

## Who this is for

If you

 - are a developer who wants to add AI assistance to your application
 - are a developer who wants to add structured AI generated content to your program
 - are anyone who wants a non-trivial terminal client for LLMs
 - are blind and looking for fully accessible software to engage with AI 

ghostbox might be for you.

## Features
<a href="https://raw.github.com/mglambda/ghostbox/master/screenshots/webui.png"><img alt="A screenshot displaying ghostbox used in the Chrome web browser through the Web UI" width="300" align="right" src="https://raw.github.com/mglambda/ghostbox/master/screenshots/webui.png"></a>
 - Generate text, json, or pydantic class instances with LLMs
 - Support for OpenAI, Llama.cpp, Lllama-box, and anyone who supports the OAI API.
 - Interruptible, streaming TTS with orpheus, Kokoro, Zonos, Amazon Polli, and others. Local or over the network.
 - Continuous, voice activated transcription with OpenAI's whisper model
 - Include images for multimodal models (OAI and Llama-box only)
 - Create, configure, and switch between AI characters 
 - Tool calling for dummys: Write python functions in tools.py, ghostbox does the rest.
 - Track, save, and reload chat history (with branching, retries, and all the usual features in the CLI)
 - Integrated HTTP webserver using websockets, with a basic web UI that let's you monitor your applications AI while it is running
 - Prompt format templates if you want them (you probably don't, everyone uses Jinja now)
 - Remote networking support. Run ghostbox on your nvidia rig and have it talk through a raspberry pie (WIP)
 - Self documenting: Try `ghostbox -cghostbox-helper` to have a friendly chat with an expert on the project
 - Much more. This is a work in progress.


## Examples

### Connecting to backends

```python
import ghostbox

# ghostbox can work with various backends.
# the generic adapter will work with anything that supports the OAI API
# it is the recommended way to make a ghostbox instance
box = ghostbox.from_generic(character_folder="ghost", endpoint="localhost:8080")

# this one is specific to OpenAI
# you don't have to specify the endpoint
cloud_box = ghostbox.from_openai(character_folder="ghost", api_key="...")

# using the llamacpp backend unlocks certain features specific to llama.cpp, e.g. setting your own samplers
# or getting better timing statistics
llama_box = ghostbox.from_llamacpp(
    character_folder="ghost", samplers=["min_p", "dry", "xtc", "temperature"]
)
```

### Using the terminal client

Here's a tiny example of the CLI.

```bash
marius@interlock ghostbox λ ghostbox -cghost
 # Prompt format template set to 'auto': Formatting is handled server side.
 # Loaded config /home/marius/.config/ghostbox.conf
 # Loaded config /home/marius/prog/ai/ghostbox/ghostbox/data/chars/ghost/config.json
 # Found vars chat_user, system_msg, current_tokens
 # Ok. Loaded /home/marius/prog/ai/ghostbox/ghostbox/data/chars/ghost
 0 👻 Write a haiku about cats.
Whiskers twitching soft,
Purring in the moonlight's glow,
Cats rule the night.
 43 👻 /time
 # generated: 22, evaluated: 17, cached: 42
 # context: 39 / 32768, exceeded: False
 # 0.48s spent evaluating prompt.
 # 3.75s spent generating.
 # 4.23s total processing time.
 # 5.87T/s, 0.17s/T

 43 👻 /set temperature 1.8

 43 👻 /retry
 # Now on branch 1
Whiskers twitch softly
Silent hunters in the night
Purring hearts' lullaby
 42 👻 
```

You can do much more with the CLI. Try `/help`, or consult the [full list of commands](https://github.com/mglambda/ghostbox/blob/master/COMMANDS.md).

### Structured Output

Getting structured output for use in applications is fun and easy using [pydantic](https://docs.pydantic.dev/latest/). If you are familiar with the OpenAI python library, you might already know this. Thanks to llama.cpp and its grammar constraints, we can also do structured output locally.

```python
from pydantic import BaseModel
from typing import *
import ghostbox, json

box = ghostbox.from_generic(character_folder="ghost-writer")

# this is the type for the object that we will let the LLM create
# how we name things here really matters
class BlogPost(BaseModel):
    title: str
    content: str
    tags: List[str]


post = box.new(
    BlogPost,  # this tells ghostbox and the backend what the structure should be
    "Write an extremely argumentative post about how an overabundance of busking is ruining berlin.",
)  # the prompt will provide context for filling in the python object
print(json.dumps(post.model_dump(), indent=4))
```

And here's the output (shortened a bit):

```
{
    "title": "Berkelbach on Buskers: When the Music Stops the Music Stops! ",
    "content": "As die hard city dwellers we are always asked for tolerance to what is perceived by outsiders to be  'fleetng and trivial inconveniences'. But there comes a time where we must stand up for a livability standard that was Berlin at one point,  not just Berlin that is . When you come into Alexander Platz on the 15 bus, the first thing that should come to mind isn\u2019t the thought 'What would make my walk home less appealing today?', but, maybe it was!  And we should really ask, to whose detriment and why?   If you think there isn\u2019t a limit on busking or noise ...",
    "tags": [
        "berlin",
        "city-life",
        "music",
        "public-space"
    ]
}
```

As a german I guarantee this is 100% authentic.

### Multimodality (images)

Make sure you use a backend and model that actually supports multimodal inputs. I use Llama-box.

```python
import ghostbox

# assume we have these
image1, image2 = "bard.jpeg", "gnome.jpeg"

box = ghostbox.from_generic(character_folder="art_critic",
                            stderr=False) # don't spam

# we need both images in context before asking the art critic
with box.images([image1, image2]):
    criticism = box.text("Which of these images is more aesthetically pleasing, and why?")

# note that once we exited the context, the images got dropped
# though they remain in the chat history.
# This is usually what you want to not resend the same image over and over, racking up tokens
print(criticism)
```

Here's `art_critic/system_msg`:

```
You are an art critic. You are an expert on art, painting, visual design, photography, and sculpture.
Your opinions on art are nuanced and sophisticated. Your sense of aesthetics is impeccable. Your artistic intuition rivaled only by your hard earned erudition.
When asked your opinion, you react extremely snobbish and arrogantly. You love to criticise bad art in creative ways, and you do so often, and with great intensity.
```

Here's the output (using qwen2-vl): 
```
I am indeed a discerning critic, and these artworks certainly do not meet my standards of artistry and design.

The first image depicts a scene with a character dressed in vibrant medieval attire. However, the composition is chaotic and disjointed, with a lack of depth and coherence in the scene. The characters and objects seem to be placed randomly, lacking any real narrative or sense of space. The color palette, while eye-catching, seems to lack harmony and balance. It seems to lack the subtlety and nuance that I would expect from a well-crafted piece of art.

The second image, featuring a dragon flying over a castle, is equally lacking in sophistication. The use of color is overly bright and garish, lacking subtlety and nuance. The composition is also disjointed and chaotic, with no clear focus or sense of narrative. The dragon, while visually striking, lacks the elegance and grace I would expect from a creature of fantasy.

Overall, these works fail to impress me. They lack the complexity and nuance of truly great art, instead resorting to surface-level appeal and spectacle. They seem to prioritize entertainment and distraction over any deeper exploration or expression of human experience. I would recommend that both artists take the time to develop a deeper understanding of composition, color, and narrative before attempting to create more meaningful work in the future.
```

Ouch. See `examples/image-compare/` for the images, which are indeed somewhat tasteless spawns of stable diffusion. Though most people do kind of like the bard one.

### Tools

To use tools a.k.a. function calling, simply write a python file with your desired functions, call it `tools.py` and place it in the character folder of the character that is supposed to use the tools. For example, with a character folder called `scribe`:

`scribe/tools.py`

```python
# this is /chars/scribe/tools.py
import os, datetime, sys

file = os.path.expanduser("~/scribe_notes.org")

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
            print("warning: Syntax error in Scribe notes, offending note: " + ws[i], file=sys.stderr)
            continue
        d["note " + str(i)] = note_data
    return d
```

The above file defines three tools for the AI: `read_notes`, `take_note`, and `directly_answer`.

The note taking tools allow the AI to interact with the filesystem, using the global FILE defined at the top of tools.py. When itneracting with a user, the scribe AI will freely choose which of the tools to apply.

The `directly_answer` tool is a small trick born out of the idiosyncrasies of tool calling: Which tool should the AI call, when it doesn't really need to call a tool? Imagine a user simply says "Hello Scribe, what's up?". That's not really worthy of taking a note, and it's not appropriate to just start reading out notes either. So in such cases, the AI can call the `directly_answer` tool, which will do nothing, and then return control to the AI.

Here is `scribe/system_msg`

```
You are {{chat_ai}}. You help a user to take notes and write down ideas. You have the capability to do so, but you may choose to not write anything down if it's not appropriate or necessary.
```

And `scribe/config.json`

```
{
	"chat_ai" : "Scribe",
	"cli_prompt" : "\n 🪶",
	"cli_prompt_color" : "blue",
	"temperature" : 0.1
}
```

If `tools.py` is found in the character folder, the `use_tools` option is automatically set to True, and ghostbox parses the file, building tool descriptions for the AI from the top level python functions. The tool descriptions offered to the AI will include information from type hints and docstrings, and this can have a big impact on the AI's ability to make good use of the tools, so it's really worth it to pick up your socks when writing tools.py. Start ghostbox with `--verbose` to see the tools that are built.

Tool calling is a very exciting and active area of development and you can expect to see more functionality here from ghostbox in the future. For a more in-depth example of a tool calling AI assistant, try out `chars/butterscotch`. Just beware. Butterscotch has a kind heart but also full shell access. You have been warned.

### Mini Adventure Game

This is `mini-adventure.py`. 

```python
import ghostbox, time, random

# the generic adapter will work with anything that supports the OAI API
box = ghostbox.from_generic(
    character_folder="game_master",  # see below
    stderr=False,  # since this is a CLI program, we don't want clutter
    tts=True,  # start the tts engine
    quiet=False,  # this means generations will be spoken automatically
    tts_model="kokoro",  # kokoro is nice because it's small and good
    tts_voice="bm_daniel",  # daniel is real GM material
)

if name := input("What is your cool adventurer name?\nName: "):
    print(f"Welcome, {name}! A game master will be with you shortly...")
else:
    name = "Drizzt Do'Urden"
    print("Better sharpen your scimitars...")

# this will make {{chat_user}} expand to the adventurer name
box.set_vars({"chat_user": name})

print(
    box.text(
        "Come up with an adventure scenario and give an introduction to the player."
    )
)

# we start conservative, but the adventure will get wilder as we go on
box.temperature = 0.3
escalation_factor = 0.05
while True:
    user_msg = input("Your response (q to quit): ")
    box.tts_stop()  # users usually like it when the tts shuts up after they hit enter

    if user_msg == "q":
        print(
            box.text(
                "{{chat_user}} will quit the game now. Please conclude the adventure and write a proper goodbye."
            )
        )
        break

    with box.options(
        max_length=100
        + 10
        * random.randint(
            -3, 3
        ),  # keep it from talking for too long, but give some variety
    ):
        print(box.text(user_msg))

    box.temperature = min(box.temperature + escalation_factor, 1.3)

# prevent halting of program until the epilogue narration has ended
box.tts_wait()
```

And this would be in `game_master/system_msg`:

```
You are a game master in a role playing game.
You tell a story collaboratively with a user, who is playing as {{chat_user}}.
```

Try it yourself for the output. With most modern models, you'll get a semi-decent, stereotypical adventure story.

Note that this only works because ghostbox is keeping track of the chat history. Ghostbox will also due context shifting when you exceed the `max_context_length`, which is something that can easily happen in RP scenarios.

### Streaming

All of the above examples used blocking calls for simplicity. Ghostbox has asynchronous and streaming variants of (almost) all of its payload methods. Here's a super simple example:

```python
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
# Now, text is being streamed to stdout and the tts backend simultaneously.
# You can prevent this by setting quiet = True (see also examples/streaming_quiet.py)
# You may notice that text appears sentence-by-sentence, and not token-by-token.
# streaming individual tokens to the tts engine is a bad idea, so ghostbox does some chunking in the background.
# this behaviour is determined by the stream_flush option. It is set to 'sentence' automatically when the tts is active.
# It also determines what will be passed to the chunk_callback. Setting stream_flush to 'token' will give you the most power.

# now we need to keep the program from exiting right away, while streaming is happening
done.wait()

# speaking is usually slower than generation, so we wait for the tts as well
box.tts_wait()
```

### Requirements
#### Backend 
Ghostbox requires an LLM backend. Currently supported backends are
 - [Llama.cpp](https://github.com/ggerganov/llama.cpp)  
 - [KoboldCPP](https://github.com/LostRuins/koboldcpp)
 - [Llama-box](https://github.com/gpustack/llama-box) (use this for multimodality atm)
 - OpenAI (untested)
 - If you have another backend that supports the OpenAI API, feel free to test ghostbox and give me feedback.
 
If you want to run LLM locally, clone your chosen repository, build the project and run the backend server. Make sure you start ghostbox with the correct `--endpoint` parameter for your chosen backend, at least if you run on a non-default endpoint.

If you use a cloud provider, be sure to set your API key with the `--api_key` command line option or do this if you use ghostbox as a library

```python
import ghostbox
box = ghostbox.from_openai(api_key="hF8sk;08xi'mnottellingyoumykeyareyoucrazy")
bot_payload = box.text("Can you create a typical Sam Altman 🔥 tweet?")
```

#### Feedwater

Ghostbox requires the [feedwater](https://github.com/mglambda/feedwater) python package, which is used to spawn the TTS process. I wrote this myself, and until I get it on PyPI, you will have to do:

```bash
python -m venv env
./env/bin/activate
git clone https://github.com/mglambda/feedwater
cd feedwater
pip install .
```

### Ghostbox Python Package
The repository can be installed as a python package.

```bash
git clone https://github.com/mglambda/ghostbox
cd ghostbox
python -m venv env # skip this if you already did it for feedwater above
. env/bin/activate
pip install .
```

I try to keep the pyproject.toml up-to-date, but the installation might fail due to one or two missing python packages. If you simply `pip install <package>` for every missing package while in the environment created above, ghostbox will eventually install.


This should make both the `ghostbox` and `ghostbox-tts` commands available. Alternatively, they can be found in the `scripts/` directory.

After a successful installation, while a backend is running, do

```bash
ghostbox -cghost
```

to begin text chat with the helpful 👻 assistant, or try

```bash
ghostbox -cjoshu --tts --audio --hide
```

for an immersive chat with a zen master.


## Additional Resources
### Orpheus TTS

[Orpheus](https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained) is a new state-of-the-art TTS model with a natural, conversational style, recently released by [Canopy Labs](https://canopylabs.ai/). It talks like a human, can laugh and sniffle and is permissively licensed (apache as of April 2025, though it should be llama3 licensed IMO). It's honestly really cool. To use it with ghostbox, set `tts_model` to `orpheus`. By default ghostbox will then acquire a [4bit quantization](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) of the model. This will knock about 2.5 gigs off your vram, with an additional 300 or so MB taken up by the snac decoder. It's worth it though.

Since orpheus is technically an LLM and based on llama3, it needs to be served by an inference engine. `ghostbox-tts` can work with any OpenAI compatible server, but has currently only been tested with llama.cpp. By default, it will try to start the `llama-server` program, so make sure that it is in your path. `ghostbox-tts` will also download other quants, and can be pointed to any gguf file for an orpheus model. See `ghostbox-tts --help` for more.

#### Orpheus Quickstart

1. Make sure llama-server is in your path. You can e.g. do the following, assuming you compiled llama.cpp in `~/llama.cpp/build`

```bash
cd /usr/bin
sudo ln -s ~/llama.cpp/build/bin/llama-server
```

2. Start ghostbox with orpheus

```bash
ghostbox -cghost --tts --tts_model orpheus
```

It will then begin to download the quantized model from huggingface. This only has to be done once. After it's done, you can do `/ttsdebug` to see the output of `ghostbox-tts`. It should look something like this

```bash
 # Initializing orpheus TTS backend.
 # Using device: cuda
 # Spawning LLM server...
 # Considering orpheus model: 

Fetching 3 files:   0%|          | 0/3 [00:00<?, ?it/s]
Fetching 3 files: 100%|██████████| 3/3 [00:00<00:00, 1886.78it/s]
 # Executing `llama-server --port 8181 -c 1024 -ngl 999 -fa -ctk q4_0 -ctv q4_0 --mlock`.
 # Using 'raw' as prompt format template.
 # Loaded config /home/marius/.config/ghostbox.conf
 # Dumping TTS config options. Set them with '/<OPTION> <VALUE>'. /ls to list again.
 #     temperature	0.6
 #     top_p	0.9
 #     repeat_penalty	1.1
 #     max_length	1024
 #     samplers	['penalties', 'min_p', 'temperature']
 #     available_voices	['tara', 'leah', 'jess', 'leo', 'dan', 'mia', 'zac', 'zoe']
 #     special_tags	['<laugh>', '<chuckle>', '<sigh>', '<cough>', '<sniffle>', '<groan>', '<yawn>', '<gasp>']
 #     voice	jess
 #     volume	1.0
 #     start_token_id	128259
 #     end_token_ids	[128009, 128260, 128261, 128257]
 #     custom_token_prefix	<custom_token_
 #     snac_model	hubertsiuzdak/snac_24khz
 #     sample_rate	24000
 #     voices	False
 #     quiet	False
 #     language	en
 #     pause_duration	1
 #     clone	
 #     clone_dir	/home/marius/prog/ai/ghostbox/voices
 #     seed	420
 #     sound_dir	sounds
 #     model	orpheus
 #     output_method	default
 #     websock_host	localhost
 #     websock_port	5052
 #     zonos_model	hybrid
 #     orpheus_quantization	Q4_K_M
 #     orpheus_model	
 #     llm_server	
 #     llm_server_executable	llama-server
 #     filepath	
 # Good to go. Reading messages from standard input. Hint: Type stuff and it will be spoken.
```

And everything should work from there.

#### Cloning with Orpheus

Coming soon!

#### Additional notes on Orpheus

Orpheus is super new and things are a bit volatile. Here's some things to consider:
 - `tara` is the best voice. I found her a bit quiet though, so `ghostbox-tts` boosts her by 25% by default. She unfortunately also has a light reverb, and I hereby pledge to donate 100% of my egg cartons to Canopy Labs, with which they can plaster the walls of their recording studio for a cheap, DIY soundbooth.
 - Realtime streaming with the 4bit quant works. Here's how:
    - According to the orpheus devs, you need~ 80 token/second on the orpheus LLM to do streaming. I have found I needed more like ~100t/s to avoid buffer underruns.
     - The factory settings for orpheus give me around 60t/s with the 4bit quant (that's on an RTX 3090).
     - The `ghostbox-tts` default settings replace the top_p sampler with min_p, which roughly doubles the t/s for me.
     - You can also disable the `penalties` sampler as well, getting a 3x speed boost and dropping the repeat_penalty. However, this will give glitchy results.
     - This way, I get about 160 to 180 t/s with good quality. Being able to set the samplers is one of the main reasons to use llama.cpp
     - I don't know why this works. I haven't observed such strong effects of samplers on generation speeds before. This may be due to the relatively large token count, or I made a mistake somewhere. Idk, DM if you can explain.
 - Orpheus has special conversational tags like `<laugh>` or `<cough>` trained into the model. By default, ghostbox will append these to the system prompt along with instructions if orpheus is being used. You can disable this with the `--no-tts_modify_system_msg` command line option.
 - The ~2.5G vram proclaimed above are only achievable with a tiny context of 1024 tokens and a quantized KV cache (thanks, llama.cpp). This is enough, though. `ghostbox` and `ghostbox-tts` work together to do chunking on the streamed text to feed on average two complete sentences at a time to the orpheus LLM. This tends to stay within the token limit, while also giving the TTS enough semantic context for nuanced generation. Incidentally, this is also roughly the length of inputs the model was trained on, so it's all coming up milhouse.
 
### Kokoro

[Kokoro](https://github.com/hexgrad/kokoro) is a very lightweight (<1GB vram) but high-quality TTS model. It is installed by default alongside the ghostbox package, and can be used with ghostbox-tts.

By default, the kokoro GPU package is installed. If you want to use the CPU only package, then after installing ghostbox, while in the virtual environment, do

```bash
pip install kokoro_onnx[cpu]
```

If you want to use ghostbox-tts alongside GPU acceleration, To ensure kokoro makes use of the GPU and cuda, do

```bash
export ONNX_PROVIDER=CUDAExecutionProvider
```

This is only needed for the standalone TTS, ghostbox itself sets the environment variable automatically.

To see a list of supported voices, consult the kokoro documentation, or do

```bash
ghostbox --tts --tts_model kokoro
/lsvoices
```

The voices listed will depend on the value of `tts_language`. You can use any of this as the ghostbox tts voice, e.g.

```bash
ghostbox -cghost --tts --tts_voice af_sky
```

#### Coming soon
 - Voice mixing with kokoro
 

### Zonos

[Zonos](https://github.com/Zyphra/Zonos) is a large sota TTS model by Zyphra. It's really good, but will knock 2 or 3 gigs off of your vram, so just be aware of that.

Zonos doesn't come with ghostbox by default, because, as of this writing (March 2025), the official packaging seems broken. To install it yourself, just (again, in the ghostbox virtual environment) do

```bash
git clone http://github.com/Zyphra/Zonos
pip install .[compile]
```

The Zonos model architecture comes in two variants: A pure transformer implementation and a transformer-mamba hybrid. The hybrid is generally better (according to my own testing), but requires flash attention. If, for whatever reason, you don't want it, just leave out the `[compile]` optional dependency above.

To use zonos with ghostbox, simply do

```bash
ghostbox -cghost --tts --tts_model zonos 
```

and if you want to use the pure transformer version, change it to

```bash
ghostbox -cghost --tts --tts_model zonos --tts_zonos_model transformer
```

#### Voice Cloning with Zonos

The Zonos TTS model is able to create voices from small audio samples. These only need to be 5 or 10 seconds long. If you use longer samples, the quality may improve, but the embeddings will become prohibitively large. You may have to experiment a bit. To create a sample from a given file called `example.ext`, do

```bash
ffmpeg -i example.ext -t 10 -ac 1 -ar 44100 sample.wav
```

This will create a sample.wav at 44.1kh sampling rate, which seems to be what Zonos wants natively. Ghostbox looks for voices in the current directory and in `tts_voice_dir`, so make sure the sample.wav is in either of those. You can then do

```bash
ghostbox -cghost --tts --tts_model zonos --tts_zonos_model hybrid --tts_voice sample.wav
```

to tie it all together, and have a helpful assistant with a cloned voice of your choice.

### Amazon Polly
*Note: currently defunct as I'm reworking the TTS backends.*

The ghostbox-tts program allows the use of amazon web services (aws) polly voices. To use them, you must create an aws account, which will give you API keys. You can then do (example for arch-linux)

```bash
pacman -S aws
aws configure
```

and you will be asked for your keys. You can then do

```bash
ghostbox -cghost --tts --tts_model polly
```
and ghostbox should talk using AWS. Doing `/lsvoices` will show you the available voices. The polly voices aren't very expressive, but have the advantage of being cloud hosted and so won't hog up your gpu.

### Local User Data

Run
```bash
./scripts/ghostbox-install
```

To create data folders with some example characters in `~/.local/share/ghostbox` and a default config file `~.config/ghostbox.conf`. This isn't necessary, just convenient.



## Character Folders

Ghostbox relates to AIs as characters. That is, the unifying principle of an LLMs operation is conceived of in terms of personality, task, and intent. This is regardless of wether you want to use your LLM to monitor heavy-duty network traffic, or to be a friendly customer support clerk, and it is irrespective of anyone's opinions on consciousness or AGI or whatever. It is a conceptual crutch, and it works well as such.

In this sense, any generation you make with ghostbox will be in the context of an AI character. You can define an AI character through a character folder, which is an actual directory on your hard drive.

A character folder may contain arbitrary files, but the following ones will be treated as special:
 - `system_msg`: The system prompt sent to the backend with every generation. This defines the personality and intent of the character.
 - `config.json`: A file containing one json dictionary, with key/value pairs being ghostbox options and their respective values. This can be used to set options particular to the character, such as tts_voice, text color, temperature and so on.
 - `tools.py`: A python file containing arbitrary python code, with all top-level functions that do not start with an underscore being taken as tools for this character. If this file is found, `use_tools` is automatically enabled for the character.
 - `initial_msg`: Deprecated. An initial message that is sent to the user on startup and prepended to the chat history with `assistant` role. This used to be a great way to give the LLM some initial color and style, but I'm deprecating it because many newer models break without the first message being from the user.

Any other file that is found in the character folder will be loaded as a file variable, with the name of the variable being the filename, and its content being the file content.

### Using character folders
Ghostbox expects character folders in the following places
 - With the `-c` or `--character_folder` command line argument
 - As the `character_folder` argument to any of the various factory functions, like `ghostbox.from_generic` or `ghostbox.from_llamacpp`
 - As an argument to the `/start` command in the terminal program
 - As an argument to the `start_session` api method, which can let you switch a character while keeping the history constant (see also `/switch`). 
 - various other places
 
Ghostbox will the nlook for character folders in the following places:
 - The current working directory
 - Whatever valid directory paths are set in the `include` option, in order. By default, these are
     - A platform specific ghostbox Application directory location, e.g. `~/.local/share/ghostbox/chars/`
     - A char dir in the ghostbox python package. This contains a handful of built-in characters (like `ghost`). Its location depends on your package manager, but will be something like `env/lib/python/site-packages/ghostbox/data/chars/`.
     - A directory `chars/` in the current working directory.

You can append to the include paths by using the `--include` command line argument.

### File Variables
As mentioned above, all files in a character folder become file variables.

File variables are expanded in text that is sent to the LLM backend. For example, if you have a `system_msg` like this

```
You are Sherlock. You help a user to solve difficult crime mysteries.
The crime you and the user want to solve today is the following:
{{crime.txt}}
```

and you have a file `crime.txt` in the character folder that looks like this

```
Things that are missing:
  - my coffee
  - where is it
```

What the LLM will get as system prompt is this
```
You are Sherlock. You help a user to solve difficult crime mysteries.
The crime you and the user want to solve today is the following:
Things that are missing:
  - my coffee
  - where is it
```

You can list variables like this at the CLI with `/lsvars`, or set them in the API with `set_vars`. Variable expansion is recursive, but stops at a depth of three.

### Dynamic File Variables

Normal file variables are loaded once upon character folder initialization and then expanded/substituted with the same content throughout the execution of the program, unless they are manually reset.

A dynamic file variable is loaded ad-hoc and its contents reloaded from its file everytime it is expanded. Dynamic file variables use square brackets within curly braces, like this
```
{[some_file.py]}
```

You can use this for great convenience at the CLI, e.g.

```bash
 420 👻> Below is a readme file for a small LLM toolkit and library called 'ghostbox'. Can you give me som feedback on it?\
 \
 {[README.md]}
```

In this case, the `{[README.md]}` expression would be expanded into the content of this very file (woah). Incidentally, the backslashes above are used to enter a newline at the CLI.

Note that although they are useful, for security reasons, dynamic file variables are disabled for every other input method except the terminal program. Think putting `{[/etc/passwd]}` deep in some python code and other such skullduggery.

## Tool Calling Guide

Coming soon!

## Options

Ghostbox uses options extensively. See the full list of options and their documentation [here](https://github.com/mglambda/ghostbox/blob/master/OPTIONS.md). An option is something like "temperature" or "backend".

Options can be used in the following places: 
 - command line arguments to the `ghostbox` program, e.g. `--temperature 1.3` or `--tts_voice af_sky`
 - `config.json` files in character folders, which contain one json dictionary that has options as key/value pairs
 - User config files, which also contain json
 - As a property on a ghostbox instance, e.g. `box.temperature = 1.0`
 - Parameters to API functions, including
     - The `from_*` factory functions, e.g. `box = from_llamacpp(temperature=1.3, tts_voice="af_sky")`.
     - The `options` context manager, e.g. `with box.options(temperature=1.3, tts_voice="af_sky"):`
     - The `Ghostbox` constructor
     - as `**options` parameter in many method signatures
 - The `/set` command in the CLI, e.g. `/set temperature 1.3` or `/set tts_voice "af_sky"`. List these with `/lsoptions`

Across these different uses for options, the naming is always consistent.

### Most useful options

Coming soon!

### Services

Setting some options to `True` has side effects, and may start services that run on seperate threads. Those options and their respective services are:

| Option | Effect when set to True |
| --- | ---- |
| tts | Starts a seperate tts process. Which program to start depends on the value of `tts_program`. The default is `ghostbox-tts`. If this is running, it will automatically speak generations, except when `quiet` is true. |
| audio | Begins automatic audio transcription. In the CLI, you can pause the transcription with CTRL+C. This is highly configurable with the various `audio_*` options, including silence threshold and activation phrase. |
| image_watch | Starts a service that watches a particular directory for new image files. When a new file appears, it gets send to the LLM for description. By default, this watches your platforms screenshot directory. |
| http | Starts a simple HTTP web server, serving the web UI at `http_host` with port `http_port`. That's `http://localhost:5050` by default. |
| websock | Starts a websock server that sends out LLM generations and listens for input. After the initial HTTP handshake, this behaves almost exactly like stdout/stdin. This is used by the web UI. |

## Further Documentation

 -  [Full list of options](https://github.com/mglambda/ghostbox/blob/master/OPTIONS.md).
 - [Full list of CLI commands](https://github.com/mglambda/ghostbox/blob/master/COMMANDS.md)
 See the `doc/` directory for extensive API documentation.
 - The `examples/` folder has many in-depth examples, as does the `tests/`. folder.
 -  If anything is still unclear, just ask the `ghostbox-helper` AI about it. I'm only half joking.

## Credits and Contributing

Thanks to the excellent [r/localllama](https://www.reddit.com/r/locallama/), and to all those who have contributed there, either through code or expertise. Thanks to the people at [llama.cpp](https://github.com/ggml-org/llama.cpp) for making local LLMs a reality, and thanks to [KoboldCPP](https://github.com/LostRuins/koboldcpp) for also making them accessible. Speaking of kobolds, you should also check out their [Kobold AI horde](https://horde.koboldai.net/).

Thanks to

https://github.com/isaiahbjork/orpheus-tts-local

for figuring out decoding of orpheus tts tokens.


The number one way to contribut to ghostbox is to test it out and give me feedback. Either by opening github issues or just telling me what kind of features you actually use or want to see in the future. I wrote this in order to build cool stuff with LLMs, and then published in partly in hopes of inspiring others to do the same, so seeing people build stuff with it is always great.