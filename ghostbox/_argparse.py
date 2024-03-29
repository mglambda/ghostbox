import argparse, os
from ghostbox.util import *


def makeArgParser(default_params):
    # default_params are only the hyperparameters (temp, etc.), not command line parameters
    parser = argparse.ArgumentParser(description="ghostbox - koboldcpp Command Line Interface", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-I", '--include', action="append", default=[userCharDir(), "chars/"], help="Include paths that will be searched for character folders named with the /start command or the --character_folder command line argument.")
    parser.add_argument('--template_include', action="append", default=[userTemplateDir(), "templates/"], help="Include paths that will be searched for prompt templates. You can specify a template to use with the -T option.")
    parser.add_argument("-T", '--prompt_format', type=str, default="guess", help="Prompt format template to use. The default is 'guess', which means ghostbox will try to determine the format through various heuristics. Often, this will result in 'chat-ml'.")
    #parser.add_argument('--stop', action="append", default=[], help="Forbidden strings that will stop the LLM backend generation.")
    parser.add_argument("-c", '--character_folder', type=str, default="", help="character folder to load at startup. The folder may contain templates, as well as arbitrary text files that may be injected in the templates. See the examples for more. Path is attempted to be resolved relative to the include paths, if any are provided.")    
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080", help="Address of koboldcpp http endpoint.")
    parser.add_argument("--backend", type=str, default="llama.cpp", help="Backend to use. Either 'llama.cpp', or 'koboldcpp'")
    parser.add_argument("--max_length", type=int, default=300, help="Number of tokens to request from koboldcpp for generation.")
    parser.add_argument("--max_context_length", type=int, default=4092, help="Maximum number of tokens to keep in context.")
    parser.add_argument("-u", "--chat_user", type=str, default="user", help="Username you wish to be called when chatting in 'chat' mode. It will also replace occurrences of {chat_user} anywhere in the character files.")
    parser.add_argument("-m", "--mode", type=str, default="default", help="Mode of operation. Changes various things behind-the-scenes. Values are currently 'default', or 'chat'.")
    parser.add_argument("-g", "--grammar_file", type=str, default="", help="Grammar file used to restrict generation output. Grammar format is GBNF.") 
    parser.add_argument("--chat_ai", type=str, default="", help="Name  the AI will have when chatting. Has various effects on the prompt when chat mode is enabled. This is usually set automatically in the config.json file of a chracter folder.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True, help="Enable streaming mode.")
    parser.add_argument("--multiline", action=argparse.BooleanOptionalAction, default=False, help="Makes multiline mode the dfault, meaning that newlines no longer trigger a message being sent to the backend. instead, you must enter the value of --multiline_delimiter to trigger a send.")
    parser.add_argument("--multiline_delimiter", type=str, default="\\", help="String that signifies the end of user input. This is only relevant for when --multiline is enabled. By default this is a backslash, inverting the normal behaviour of backslashes allowing to enter a newline ad-hoc while in multiline mode. This option is intended to be used by scripts to change the delimiter to something less common.")
    parser.add_argument("--color", action=argparse.BooleanOptionalAction, default=True, help="Enable colored output.")
    parser.add_argument("--text_ai_color", type=str, default="none", help="Color for the generated text, as long as --color is enabled. Most ANSI terminal colors are supported.")
    parser.add_argument("--text_ai_style", type=str, default="bright", help="Style for the generated text, as long as --color is enabled. Most ANSI terminal styles are supported.")    
    parser.add_argument("--warn_trailing_space", action=argparse.BooleanOptionalAction, default=True, help="Warn if the prompt that is sent to the backend ends on a space. This can cause e.g. excessive emoticon use by the model.")
    parser.add_argument("--warn_hint", action=argparse.BooleanOptionalAction, default=True, help="Warn if you have a hint set.")
    parser.add_argument("--json", action=argparse.BooleanOptionalAction, default=False, help="Force generation output to be in JSON format. This is equivalent to using -g with a json.gbnf grammar file, but this option is provided for convenience.") 
    parser.add_argument("--stream_flush", type=str, default="token", help="When to flush the streaming buffer. When set to 'token', will print each token immediately. When set to 'sentence', it will wait for a complete sentence before printing. This can be useful for TTS software. Default is 'token'.")
    parser.add_argument("--cli_prompt", type=str, default="\n 🧠 ", help="String to show at the bottom as command prompt. Can be empty.")
    parser.add_argument("--cli_prompt_color", type=str, default="none", help="Color of the prompt. Uses names of standard ANSI terminal colors. Requires --color to be enabled.")
    parser.add_argument("--hint", type=str, default="", help="Hint for the AI. This string will be appended to the prompt behind the scenes. It's the first thing the AI sees. Try setting it to 'Of course,' to get a more compliant AI.")
    parser.add_argument("--tts", action=argparse.BooleanOptionalAction, default=False, help="Enable text to speech on generated text.")
    parser.add_argument("--tts_program", type=str, default="ghostbox-tts-tortoise", help="Path to a TTS (Text-to-speech) program to verbalize generated text. The TTS program should read lines from standard input. Many examples are provided in scripts/ghostbox-tts-* .")
    parser.add_argument("--tts_voice_dir", type=str, default=os.getcwd() + "/voices/", help="Directory to check first for voice file.")
    parser.add_argument("--tts_tortoise_quality", type=str, default="fast", help="Quality preset. tortoise-tts only. Can be 'ultra_fast', 'fast' (default), 'standard', or 'high_quality'")
    parser.add_argument("--tts_volume", type=float, default=1.0, help="Volume for TTS voice program. Is passed to tts_program as environment variable.")
    parser.add_argument("--tts_rate", type=int, default=50, help="Speaking rate for TTS voice program. Is passed to tts_program as environment variable. Note that speaking rate is not supported by all TTS engines.")
    parser.add_argument("--tts_additional_arguments", type=str, default="", help="Additional command line arguments that will be passed to the tts_program.")
    parser.add_argument("--image_watch", action=argparse.BooleanOptionalAction, default=False, help="Enable watching of a directory for new images. If a new image appears in the folder, the image will be loaded with id 0 and can be inserted as [img-0]. This works with multimodal models only (like llava).")
    parser.add_argument("--image_watch_dir", type=str, default=os.path.expanduser("~/Pictures/Screenshots/"), help="Directory that will be watched for new image files when --image_watch is enabled.")
    parser.add_argument("--image_watch_msg", type=str, default="[img-0]Can you describe this image?", help="If image_watch is enabled, this message will be automatically send to the backend whenever a new image is detected. Set this to '' to disable automatic messages, while still keeping the automatic update the image with id 0.")
    parser.add_argument("--image_watch_hint", type=str, default="", help="If image_watch is enabled, this string will be sent to the backend as start of the AI response whenever a new image is detected and automatically described. This allows you to guide or solicit the AI by setting it to e.g. 'Of course, this image show' or similar. Default is ''")
    parser.add_argument("--whisper_model", type=str, default="base.en", help="Name of the model to use for transcriptions using the openai whisper model. Default is 'base.en'. For a list of model names, see https://huggingface.co/openai/whisper-large")
    parser.add_argument("-y", "--tts_voice", type=str, default="random", help="Voice file to use for TTS. Default is 'random', which is a special value that picks a random available voice for your chosen tts_program. The value of tts_voice will be changed at startup if random is chosen, so when you find a voice you like you can find out its name with /lsoptions and checking tts_voice.")
    parser.add_argument("--tts_subtitles", action=argparse.BooleanOptionalAction, default=True, help="Enable printing of generated text while TTS is enabled.")     
    parser.add_argument("--config_file", type=str, default="", help="Path to a config fail in JSON format, containing a dictionary with OPTION : VALUE pairs to be loaded on startup. Same as /loadconfig or /loadoptions. To produce an example config-file, try /saveconfig example.json.")
    parser.add_argument("--chat_show_ai_prompt", action=argparse.BooleanOptionalAction, default=True, help="Controls wether to show AI prompt in chat mode. Specifically, assuming chat_ai = 'Bob', setting chat_show_ai_prompt to True will show 'Bob: ' in front of the AI's responses. Note that this is always sent to the back-end (in chat mode), this parameter merely controls wether it is shown.")
    parser.add_argument("--smart_context", action=argparse.BooleanOptionalAction, default=True, help="Enables ghostbox version of smart context, which means dropping text at user message boundaries when the backend's context is exceeded. If you disable this, it usually means the backend will truncate the raw message. Enabling smart context means better responses and longer processing time due to cache invalidation, disabling it means worse responses with faster processing time.")
    parser.add_argument("--hide", action=argparse.BooleanOptionalAction, default=False, help="Hides some unnecessary output, providing a more immersive experience. Same as starting with /hide.")
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=False, help="Enable automatic transcription of audio input using openai whisper model. Obviously, you need a mic for this.")
    parser.add_argument("--audio_silence_threshold", type=int, default=2000, help="An integer value denoting the threshold for when automatic audio transcription starts recording. (default 2000)")
    parser.add_argument("--audio_show_transcript", action=argparse.BooleanOptionalAction, default=True, help="Show transcript of recorded user speech when kaudio transcribing is enabled. When disabled, you can still see the full transcript with /log or /print.")
    parser.add_argument("--expand_user_input", action=argparse.BooleanOptionalAction, default=True, help="Expand variables in user input. E.g. {$var} will be replaced with content of var. Variables are initialized from character folders (i.e. file 'memory' will be {$memory}), or can be set manually with the /varfile command or --varfile option.")
    parser.add_argument("-x", '--var_file', action="append", default=[], help="Files that will be added to the list of variables that can be expanded. E.g. -Vmemory means {$memory} will be expanded to the contents of file memory, provided expand_user_input is set. Can be used to override values set in character folders.")


    for (param, value) in default_params.items():
        parser.add_argument("--" + param, type=type(value), default=value, help="Passed on to koboldcpp. Change during runtime with /set " + param + ".")
        
    return parser

