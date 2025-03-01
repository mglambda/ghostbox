# ghostbox

Ghostbox is a command line interface to local LLM (large language model) server applications, such as koboldcpp or llama.cpp. It's primary purpose is to give a simple, stripped-down way to engage with AI chat bots.
Ghostbox was made to be blind-accessible, and is fully compatible with any screen reader that can read a shell.

## Features

 - Command line interface that plays well with standard input and output
 - Supports various backends, including llama.cpp, llama.box, as well as anything using the OpenAI API
 - Character templates to define AI characters in a simple, file based manner
 - Includes prompt templates for many popular formats (like chat-ml)
 - Chat functionality, including retrying or rewriting prompts, and a chat history that can be saved and reloaded
 - Live audio transcription using OpenAI's Whisper model
 - Multimodal support for images (llava, qwen2-vl), including automatic descriptions for screenshots or arbitrary image files.
 - Painless JSON output (just do --json)
 - Grammars to arbitrarily restrict token output 
 - TTS (text-to-speech) with various backends, including Zonos, Kokoro and Amazon Polli, using ghostbox-tts.
 - Token streaming
 - Model hyperparameter control (temperature, top p, etc.), including with config-files
 - small web ui with --http on localhost:5050
 - Simple design. By default, ghostbox let's you just send raw text to your LLM, and returns the generated output. This is great if you want to learn about proper prompting, which was part of my motivation for making this.

## Documentation
### Character Templates

Coming soon!

### Additional Documentation

Try `ghostbox --help` or type `/help` at the CLI for additional information.

## Installation
### Requirements

Ghostbox requires an LLM backend. Currently supported backends are
 - Llama.cpp (currently prefered, find it at https://github.com/ggerganov/llama.cpp )  
 - Koboldcpp ( https://github.com/LostRuins/koboldcpp )

Clone either repository, build the project and run the backend server. Make sure you start ghostbox with the --endpoint parameter to the endpoint provided by the backend. This is http://localhost:8080 for Llama.cpp (default), or http://localhost:5001 for koboldcpp, at least by default.


### Amazon Polly

The default TTS script provided allows the use of amazon web services (aws) polly voices. To use them, you must create an aws account, which will give you API keys. You can then do (example for arch-linux)

```
pacman -S aws
aws configure
```

and you will be asked for your keys. To test if it works, you can either run ghostbox with the `--tts` option, activate TTS by typing `/tts` at the ghostbox CLI, or try the `scripts/ghostbox-tts-polly` script in the repository.
### Multimodal Image Description

Acquire the latest Llava model from huggingface. You will need a gguf file and a mmproj file with model projections for the underlying LLM (e.g. `mmproj-model-f16.gguf`). You can then start Llama.cpp like this

```
cd ~/llama.cpp/build/bin 
./server --ctx-size 2048 --parallel 1 --mlock --no-mmap -m llava-v1.6-mistral-7b.Q5_K_M.gguf --mmproj mmproj-model-f16.gguf
```

assuming you built the project in `llama.cpp/build`. You may then do either 
```
ghostbox -cdolphin
/image ~/Pictures.paris.png
Here is a picture of paris. [img-1] Can you please describe it?
```

or, for automatically describing the screen when you make a screenshot

```
ghostbox -cdolphin --image_watch
```

assuming screenshots are stored in `~/Pictures/Screenshots/`, which is the default for the gnome-screenshot utility.
Note that multimodal support is currently working with Llama.cpp only.

### Python Package
The repository can be installed as a python package.

```
git clone https://github.com/mglambda/ghostbox
cd ghostbox
python -m venv env
. env/bin/activate
pip install .
```

I try to keep the setup.py up-to-date, but the installation might fail due to one or two missing python packages. If you simply `pip install <package>` for every missing package while in the environment created above, ghostbox will eventually install.

This should make the `ghostbox` command available. Alternatively, it can be found in `scripts/ghostbox`.

### Data

Run
```
./scripts/ghostbox-install
```

To create data folders with some example characters in `~/.local/share/ghostbox` and a default config file `~.config/ghostbox.conf`.

After a successful installation, while a backend is running, try
```
ghostbox -cdolphin
```

to begin text chat with the helpful dolphin assistant, or try

```
ghostbox -cjoshu --tts --audio --hide
```

for an immersive chat with a zen master.
