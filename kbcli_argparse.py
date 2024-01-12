import argparse

def makeArgParser(default_params):
    # default_params are only the hyperparameters (temp, etc.), not command line parameters
    parser = argparse.ArgumentParser(description="kbcli - koboldcpp Command Line Interface", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-I", '--include', action="append", default=["chars/"], help="Include paths that will be searched for character folders named with the /start command or the --character_folder command line argument.")
    parser.add_argument("-c", '--character_folder', type=str, default="", help="character folder to load at startup. The folder may contain templates, as well as arbitrary text files that may be injected in the templates. See the examples for more. Path is attempted to be resolved relative to the include paths, if any are provided.")    
    parser.add_argument("--endpoint", type=str, default="http://localhost:5001", help="Address of koboldcpp http endpoint.")
    parser.add_argument("--max_length", type=int, default=300, help="Number of tokens to request from koboldcpp for generation.")
    parser.add_argument("--chat_user", type=str, default="", help="Username you wish to be called when chatting. Setting this automatically enables chat mode. It will also replace occurrences of {chat_user} anywhere in the character files.")
    parser.add_argument("--chat_ai", type=str, default="", help="Name  the AI will have when chatting. Has various effects on the prompt when chat mode is enabled. This is usually set automatically in the config.json file of a chracter folder.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Enable streaming mode.")
    parser.add_argument("--streaming_flush", action=argparse.BooleanOptionalAction, default=False, help="When True, flush print buffer immediately in streaming mode (print token-by-token). When set to false, waits for newline until generated text is printed.")
    parser.add_argument("--cli_prompt", type=str, default="\n 🧠 ", help="String to show at the bottom as command prompt. Can be empty.")
    parser.add_argument("--tts", action=argparse.BooleanOptionalAction, default=False, help="Enable text to speech on generated text.")
    parser.add_argument("--tts_program", type=str, default="tts.sh", help="Path to a TTS (Text-to-speech) program to verbalize generated text. The TTS program should read lines from standard input.")
    parser.add_argument("--tts_subtitles", action=argparse.BooleanOptionalAction, default=True, help="Enable printing of generated text while TTS is enabled.") 
    parser.add_argument("--config_file", type=str, default="", help="Path to a config fail in JSON format, containing a dictionary with OPTION : VALUE pairs to be loaded on startup. Same as /loadconfig or /loadoptions. To produce an example config-file, try /saveconfig example.json.")
    parser.add_argument("--chat_show_ai_prompt", type=str, default=True, help="Controls wether to show AI prompt in chat mode. Specifically, assuming chat_ai = 'Bob', setting chat_show_ai_prompt to True will show 'Bob: ' in front of the AI's responses. Note that this is always sent to the back-end (in chat mode), this parameter merely controls wether it is shown. This parameter is automatically set to false when enabling TTS.")
    parser.add_argument("--hide", action=argparse.BooleanOptionalAction, default=False, help="Hides some unnecessary output, providing a more immersive experience. Same as starting with /hide.")
    
    for (param, value) in default_params.items():
        parser.add_argument("--" + param, type=type(value), default=value, help="Passed on to koboldcpp. Change during runtime with /set " + param + ".")
        
    return parser
