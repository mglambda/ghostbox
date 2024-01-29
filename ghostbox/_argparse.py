import argparse, os
from ghostbox.util import *

def makeArgParser(default_params):
    # default_params are only the hyperparameters (temp, etc.), not command line parameters
    parser = argparse.ArgumentParser(description="ghostbox - koboldcpp Command Line Interface", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-I", '--include', action="append", default=[userCharDir(), "chars/"], help="Include paths that will be searched for character folders named with the /start command or the --character_folder command line argument.")
    parser.add_argument('--forbid_strings', action="append", default=[], help="Strings that will be replaced with the empty string in LLM generated text. This is a repeatable option.")
    parser.add_argument("-c", '--character_folder', type=str, default="", help="character folder to load at startup. The folder may contain templates, as well as arbitrary text files that may be injected in the templates. See the examples for more. Path is attempted to be resolved relative to the include paths, if any are provided.")    
    parser.add_argument("--endpoint", type=str, default="http://localhost:5001", help="Address of koboldcpp http endpoint.")
    parser.add_argument("--max_length", type=int, default=300, help="Number of tokens to request from koboldcpp for generation.")
    parser.add_argument("--max_context_length", type=int, default=2048, help="Maximum tokens to keep in context.")
    parser.add_argument("-u", "--chat_user", type=str, default="user", help="Username you wish to be called when chatting in 'chat' mode. It will also replace occurrences of {chat_user} anywhere in the character files.")
    parser.add_argument("-m", "--mode", type=str, default="default", help="Mode of operation. Changes various things behind-the-scenes. Values are currently 'default', or 'chat'.")
    parser.add_argument("-g", "--grammar_file", type=str, default="", help="Grammar file used to restrict generation output. Grammar format is GBNF.") 
    parser.add_argument("--chat_ai", type=str, default="", help="Name  the AI will have when chatting. Has various effects on the prompt when chat mode is enabled. This is usually set automatically in the config.json file of a chracter folder.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Enable streaming mode.")
    parser.add_argument("--json", action=argparse.BooleanOptionalAction, default=False, help="Force generation output to be in JSON format. This is equivalent to using -g with a json.gbnf grammar file, but this option is provided for convenience.") 
    parser.add_argument("--streaming_flush", action=argparse.BooleanOptionalAction, default=True, help="When True, flush print buffer immediately in streaming mode (print token-by-token). When set to false, waits for newline until generated text is printed.")
    parser.add_argument("--cli_prompt", type=str, default="\n 🧠 ", help="String to show at the bottom as command prompt. Can be empty.")
    parser.add_argument("--tts", action=argparse.BooleanOptionalAction, default=False, help="Enable text to speech on generated text.")
    parser.add_argument("--tts_program", type=str, default="ghostbox-tts-polly", help="Path to a TTS (Text-to-speech) program to verbalize generated text. The TTS program should read lines from standard input.")
    parser.add_argument("--tts_voice_dir", type=str, default=os.getcwd() + "/voices/", help="Directory to check first for voice file.")
    parser.add_argument("--tts_volume", type=float, default=1.0, help="Volume for TTS voice program. Is passed to tts_program with --volume.")
    parser.add_argument("--whisper_model", type=str, default="base.en", help="Name of the model to use for transcriptions using the openai whisper model. Default is 'base.en'. For a list of model names, see https://huggingface.co/openai/whisper-large")
    parser.add_argument("-V", "--tts_voice", type=str, default="Joey", help="Voice file to use for TTS.")
    parser.add_argument("--tts_subtitles", action=argparse.BooleanOptionalAction, default=True, help="Enable printing of generated text while TTS is enabled.")     
    parser.add_argument("--config_file", type=str, default="", help="Path to a config fail in JSON format, containing a dictionary with OPTION : VALUE pairs to be loaded on startup. Same as /loadconfig or /loadoptions. To produce an example config-file, try /saveconfig example.json.")
    parser.add_argument("--chat_show_ai_prompt", action=argparse.BooleanOptionalAction, default=True, help="Controls wether to show AI prompt in chat mode. Specifically, assuming chat_ai = 'Bob', setting chat_show_ai_prompt to True will show 'Bob: ' in front of the AI's responses. Note that this is always sent to the back-end (in chat mode), this parameter merely controls wether it is shown. This parameter is automatically set to false when enabling TTS.")
    parser.add_argument("--hide", action=argparse.BooleanOptionalAction, default=False, help="Hides some unnecessary output, providing a more immersive experience. Same as starting with /hide.")
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=False, help="Enable automatic transcription of audio input using openai whisper model. Obviously, you need a mic for this.")
    parser.add_argument("--audio_show_transcript", action=argparse.BooleanOptionalAction, default=True, help="Show transcript of recorded user speech when kaudio transcribing is enabled. When disabled, you can still see the full transcript with /log or /print.")
    parser.add_argument("--expand_user_input", action=argparse.BooleanOptionalAction, default=True, help="Expand variables in user input. E.g. {$var} will be replaced with content of var. Variables are initialized from character folders (i.e. file 'memory' will be {$memory}), or can be set manually with the /varfile command or --varfile option.")
    parser.add_argument("-x", '--var_file', action="append", default=[], help="Files that will be added to the list of variables that can be expanded. E.g. -Vmemory means {$memory} will be expanded to the contents of file memory, provided expand_user_input is set. Can be used to override values set in character folders.")


    for (param, value) in default_params.items():
        parser.add_argument("--" + param, type=type(value), default=value, help="Passed on to koboldcpp. Change during runtime with /set " + param + ".")
        
    return parser

