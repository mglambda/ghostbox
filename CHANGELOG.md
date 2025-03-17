# ghostbox changelog

In reverse chronological order.


v0.21.5         Fix not being able to wait for tts to finish speaking.
v0.21.4         Fix user configs overriding API calls. Also overhaul README.
v0.21.3         Fix images and multimodality.
v0.21.2         Fix kokoro voices not being listed.
v0.21.1         Add documentation to the project.
v0.21.0         License change to lgpl v3.0
v0.20.2         Fix some cosmetic Web UI issues.
v0.20.1         Fix bug with kokoro crashing on pure punctuation inputs.
v0.20.0         Now supporting generalized tool calling using tools.py (OpenAI API compatible).
v0.19.7         Fix infuriatingly erratic CLI prompt behaviour.
v0.19.6         Add order of samplers as command lien argument.
v0.19.5         Fix timing statistics. More support for sampling parameters in different backends.
v0.19.4         Various improvements to WebUI.
v0.19.3         Improve TTS performance.
v0.19.2         Add dynamic file vars, i.e. expanding {[FILENAME]} to the contents of FILENAME.
v0.19.1         Fix not being able to set http and websock as options. Also fix initialization order.
v0.19.0         Web UI on localhost:5050 with --http command line argument.
v0.18.2         Fix not being able to stop streaming generation.
v0.18.1         Fix ghostbox-tts not interrupting speech properly.
v0.18.0         Kokoro TTS support!
v0.17.0         Zonos TTS support. Also generally improved ghostbox-tts, which is now a useful program in its own right.
v0.16.0         OpenAI API support and true multimodality.
v0.15.5         Add tools_inject_dependency_function to give dependencies to all tool-using AIs at startup.
v0.15.4         Fix broken chat formatting. Fix being unable to switch characters with API.
v0.15.3         Fix bug with streamed text not being properly formatted in generation_callback. Fix errors in phi3-instruct template.
v0.15.2         Add phi3-instruct template.
v0.15.1         Fix quiet being default in API, overriding config options.
v0.15.0         Add dependency injection to tools. Also contains many small API improvements and fixes.
v0.14.3         Fix hide not being implemented in internal API.
v0.14.2         Fix numerous bugs with API and TTS. Add tests for TTS functions.
v0.14.1         Fix {{chat_ai}} not being replaced in character templates. Refactor session a bit. Fix session.getVar and session.getVars always return expanded text. Add recursion depths (3) to session.expandVars.
v0.14.0         Add activation phrase to only trigger interactions with the AI when certain words are spoken (audio mode). Refactor main loop into reusable interaction method. Fix CLI prompt string not being displayed correctly with streaming. Fix program not being repsonsive while interacting (is now non-blocking). Add some foundational API code.
v0.13.0         Add use of tools using simple tool.py python files. Only tested with command-r currently.
v0.12.4         Fix bug in vacuna template.
v0.12.3         Add llama3 template. At least provisionally.
v0.12.2         Add vacuna template.
v0.12.1         Fix ghostbox.el broken keywords list
v0.12.0         Now with ghostbox.el for basic emacs integration as a comint process!
v0.11.1         Fix error in handling http requests of llama server. Add flag to disable smart context. Add a funny /test question command.
v0.11.0         Add tortoise-tts support.
v0.10.3         Fix guessing.
v0.10.2         Add guessing of prompt formats with new layers file.
v0.10.0         Various stability and proper prompt formatting improvements. Now autosaves the story on an unrecoverable crash, and you can /detokenize to test your prompt format.
v0.9.9          Fix bug in alpaca and user-aassistant-newline templates. Add templates to install script.
v0.9.7          Add helpful startup script generator in scripts/ghostbox-generate-startup-scripts. Point it at your model folder and get a bunch of convenient scripts to start llama server, including a spreadsheet file to manage models and the layers and context you want for them.
v0.9.6          Fix broken raw template.
v0.9.5          Fix smart shifting being too eager to shift. Also add session variable expansion for all user inputs (can do /tokenize -c {{system_msg}} to get token count in system prompt).
v0.9.4          Fix username not being in stop words for backend.
v0.9.3          Fix minsk subtitles :)
v0.9.2          Fix image_watch crashing when watched folder is empty. Add image_watch_hint to solicit AI.
v0.9.1          Fix backend status and timing data not updating when streaming.
v0.9.0          Streaming works and works well with TTS. Also fix messy backend code.
v0.8.8          Fix previous fixes lol.
v0.8.7          Fix tts not respecting command line arguments. Again. Add /switch command.
v0.8.6          Fix /save appneding .json to everything. why would you do that it no longer does that
v0.8.5          Fix command line arguments not taking precedence over config files.
v0.8.4          Fix some command line options being ignored when also specified in a char's config.json.
v0.8.3          Fix broken /retry. Remove useless /resend. Add non-existing /rephrase.
v0.8.2          Fix broken continue.
v0.8.1          Fix install script and remove unnecessary chat-ml char.
v0.8.0          Prompt format templates!
v0.7.0          Smart context handling. At least our version of it. Also add some utility functions like /tokenize and /raw.
v0.6.5          Fix determining of max_context_length with llama.cpp. Add some timing data. Fix timings breaking koboldcpp.
v0.6.4          Fix tts not starting when only supplied as command line option. Fix not being able to start image_watch except through command line option. Add /image_watch command.
v0.6.3          Fix cli prompt not being changed by char/config.json. Fix tts restarting every time a char is loaded.
v0.6.2          Add /status and /time. Fix bug with continuous transcriber not allowing program to exit if recording was paused.
v0.6.1          Remove noisy testing code.
v0.6.0          Multimodal models. Image description works but has been tested only with llava-v1.6 and llama.cpp. Includes automatic description for screenshots (in gnome).
v0.5.2          Fix trailing space problem. Add some preparations for future chat-thought mode.
v0.5.1          Fix CTRL+c interrupt signal with audio transcriptions. Fix ghostbox-transcribe script. Rename WhisperTranscriber.transcribe to Whispertranscriber.transcribeWithPrompt.
v0.5.0          Now with audio recording as input method using openai's whisper model to transcribe audio.
v0.4.0          Add grammar support. This uses GBNF grammars. Examples can be found with llama.cpp. I'm Add a --json flag for convenience.
v0.3.1          Fix config file being in wrong place and being platform dependent. Fix correct location of char dir. Add an install script for deployment of default config file and example chars.
v0.3.0          Add multiline inputs using backslash at end of line.
v0.2.6          Fix not being able to adjust volume on TTS. Fix TTS not restarting properly. This works now, but only on linux.
v0.2.5          Fix ghostbox-tts-polly script garbling output when multiple lines of input are received quickly. Now using pygame, which should be just as platform independent.
v0.2.4          Fix generation of '....' in some scenarios, which messes up some TTS. New option forbid_strings has a list of strings that get replaced with empty string.
v0.2.3          Add proper loading of user config file in home directory (.ghostbox.conf.json). Fix order of config file loading. Fix some formatting with /hide.
v0.2.2          Prettier
v0.2.1          Proper documentation.
v0.2.0          This version breaks compatibility with saved stories.
