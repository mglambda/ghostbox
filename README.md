# ghostbox

Ghostbox is a command line interface to local LLM (large language model) server applications, such as koboldcpp or llama.cpp. It's primary purpose is to give a simple, stripped-down way to engage with AI chat bots.
Ghostbox was made to be blind-accessible, and is fully compatible with any screen reader that can read a shell.

## Features

 - Command line interface that plays well with standard input and output
 - Character folders to define AI characters in a simple, file based manner
  - Template system to quickly and dynamically create and alter AI characters
 - Includes prompt templates for many popular formats (like chat-ml)
 - Chat functionality, including retrying or rewriting prompts, and a chat history that can be saved and reloaded
 - Live audio transcription using OpenAI's Whisper model
 - TTS (text-to-speech) capabilities (experimental) 
 - Token streaming
 - Model hyperparameter control (temperature, top p, etc.), including with config-files
 - Simple design. By default, ghostbox let's you just send raw text to your LLM, and returns the generated output. This is great if you want to learn about proper prompting, which was part of my motivation for making this.

## Documentation
### Character Templates

Coming soon!

### Additional Documentation

Try `ghostbox --help` or type `/help` at the CLI for additional information.

## Installation
### Requirements

Ghostbox requires an LLM backend. I like koboldcpp, and that is the only one I have tested so far.

```
git clone https://github.com/LostRuins/koboldcpp
```

Follow the koboldcpp installation instructions, then run it with a model of your choice. Make sure it provides an http endpoint (it does by default). If you change the endpoint from localhost:5000, you must tell ghostbox with the `--endpoint` command line argument. 

### Amazon Polly

The default TTS script provided allows the use of amazon web services (aws) polly voices. To use them, you must create an aws account, which will give you API keys. You can then do (example for arch-linux)

```
pacman -S aws
aws configure
```

and you will be asked for your keys. To test if it works, you can either run ghostbox with the `--tts` option, activate TTS by typing `/tts` at the ghostbox CLI, or try the `scripts/ghostbox-tts-polly` script in the repository.

### Python Package
The repository can be installed as a python package.

````
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

After a successful installation, while koboldcpp is running, try
```
ghostbox -cdolphin
```

to begin text chat with the helpful dolphin assistant, or try

```
ghostbox -cjoshu --tts --audio --hide
```

for an immersive chat with a zen master.
