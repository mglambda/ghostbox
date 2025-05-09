import argparse, os
from ghostbox.util import *
from ghostbox import backends
from ghostbox.definitions import *
from ghostbox.__init__ import get_ghostbox_data


class TaggedArgumentParser:
    """Creates an argument parser along with a set of tags for each argument.
    Arguments to the constructor are passed on to argparse.ArgumentParser.__init__ .
    You can then use add_arguments just like with argparse, except that there is an additional keyword argument 'tags', which is a dictionary that will be associated with that command line argument.
    """

    def __init__(self, **kwargs):
        self.parser = argparse.ArgumentParser(**kwargs)
        self.tags = {}

    def add_argument(self, *args, **kwargs):
        if "tag" in kwargs:
            # this is a bit tricky, argparse does a lot to find the arg name, but this might do
            # find the longest arg, strip leading hyphens, replace remaining hyphens with _
            arg = (
                sorted(args, key=lambda w: len(w), reverse=True)[0]
                .strip("-")
                .replace("-", "_")
            )
            self.tags[arg] = kwargs["tag"]
            self.tags[arg].name = arg
            # and if there is no help then let it blow up
            self.tags[arg].help = kwargs["help"]
            if "default" in kwargs:
                self.tags[arg].default_value = (
                    kwargs["default"] if kwargs["default"] is not None else "None"
                )
            del kwargs["tag"]

        self.parser.add_argument(*args, **kwargs)

    def get_parser(self):
        return self.parser

    def get_tags(self):
        return self.tags


def makeTaggedParser(default_params) -> TaggedArgumentParser:
    parser = TaggedArgumentParser(
        description="LLM Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # we'll be typing these a lot so buckle up
    mktag = ArgumentTag
    AT = ArgumentType
    AG = ArgumentGroup
    parser.add_argument(
        "-I",
        "--include",
        action="append",
        default=[userCharDir(), get_ghostbox_data("chars/"), "chars/"],
        help="Include paths that will be searched for character folders named with the /start command or the --character_folder command line argument.",
        tag=mktag(type=AT.Porcelain, group=AG.Characters, motd=True),
    )
    parser.add_argument(
        "--template_include",
        action="append",
        default=[userTemplateDir(), get_ghostbox_data("templates/"), "templates/"],
        help="Include paths that will be searched for prompt templates. You can specify a template to use with the -T option.",
        tag=mktag(type=AT.Plumbing, group=AG.Templates),
    )
    parser.add_argument(
        "--history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, do not append messages to chat history. This is used mostly in the API to send system messages that won't clutter up the user's chat history.",
        tag=mktag(type=AT.Plumbing, group=AG.General),
    )
    parser.add_argument(
        "--history_force_alternating_roles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force chat history to keep alternating roles between 'assistant' and 'user'. This can be important as many backends require this server-side, throwing a server error if it is not observed. Enabling this will enforce the alternation through a variety of strategies, some of which may alter your messages.",
        tag=mktag(type=AT.Plumbing, group=AG.General),
    )        
    parser.add_argument(
        "-T",
        "--prompt_format",
        type=str,
        default="auto",
        help="Prompt format template to use. The default is 'auto', which means ghostbox will .let the backend handle templating, which is usually the right choice. You can still use other settings, like 'raw', to experiment. This is ignored if you use the generic or openai backend. Note: Prompt format templates used to be more important in the early days of LLMs, as confusion was rampant and mistakes were not uncommon even in official releases. Nowadays, it is quite safe to use the official templates. You may still want to use this option for experimentation, however.",
        tag=mktag(type=AT.Plumbing, group=AG.Templates),
    )
    parser.add_argument(
        "-s",
        "--stop",
        action="append",
        default=[],
        help="Forbidden strings that will stop the LLM backend generation.",
        tag=mktag(type=AT.Porcelain, group=AG.Generation, motd=True),
    )
    parser.add_argument(
        "-c",
        "--character_folder",
        type=str,
        default="",
        help="character folder to load at startup. The folder may contain a `system_msg` file, a `config.json`, and a `tools.py`, as well as various other files used as file variables. See the examples and documentation for more.",
        tag=mktag(
            type=AT.Porcelain, group=AG.Characters, very_important=True, motd=True
        ),
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="If provided, process the prompt and exit.",
        tag=mktag(type=AT.Porcelain, group=AG.Generation),
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8080",
        help="Address of backend http endpoint. This is a URL that is dependent on the backend you use, though the default of localhost:8080 works for most, including Llama.cpp and Kobold.cpp. If you want to connect to an online provider that is not part of the explicitly supported backends, this is where you would supply their API address.",
        tag=mktag(type=AT.Porcelain, group=AG.Backend),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=LLMBackend.generic.name,
        help="Backend to use. The default is `generic`, which conforms to the OpenAI REST API, and is supported by most LLM providers. Choosing a more specific backend may provide additional functionality. Other possible values are "
        + ", ".join([e.name for e in LLMBackend])
        + ".",
        tag=mktag(type=AT.Porcelain, group=AG.Backend, very_important=True),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for OpenAI. Without the `--backend openai` option, this has no effect.",
        tag=mktag(type=AT.Plumbing, group=AG.OpenAI),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=300,
        help="Number of tokens to request from backend for generation. Generation is stopped when this number is exceeded. Negative values mean generation is unlimited and will terminate when the backend generates a stop token.",
        tag=mktag(type=AT.Porcelain, group=AG.Generation, very_important=True),
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=32768,
        help="Maximum number of tokens to keep in context.",
        tag=mktag(type=AT.Porcelain, group=AG.Generation, very_important=True),
    )
    parser.add_argument(
        "-u",
        "--chat_user",
        type=str,
        default="user",
        help="Username you wish to be called when chatting in 'chat' mode. It will also replace occurrences of {chat_user} anywhere in the character files. If you don't provide one here, your username will be determined by your OS session login.",
        tag=mktag(type=AT.Porcelain, group=AG.General, very_important=True),
    )
    parser.add_argument(
        "-M",
        "--mode",
        type=str,
        default="default",
        help="Mode of operation. Changes various things behind-the-scenes. Values are currently 'default', or 'chat'.",
        tag=mktag(type=AT.Plumbing, group=AG.Templates),
    )
    parser.add_argument(
        "--force_params",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Force sending of sample parameters, even when they are seemingly not supported by the backend (use to debug or with generic",
        tag=mktag(type=AT.Plumbing, group=AG.Backend),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="LLM to use for requests. This only works if the backend supports choosing models.",
        tag=mktag(type=AT.Porcelain, group=AG.Backend, very_important=True),
    )
    parser.add_argument(
        "-g",
        "--grammar_file",
        type=str,
        default="",
        help="Grammar file used to restrict generation output. Grammar format is GBNF.",
        tag=mktag(type=AT.Plumbing, group=AG.Generation, motd=True),
    )
    parser.add_argument(
        "--chat_ai",
        type=str,
        default="",
        help="Name  the AI will have when chatting. Has various effects on the prompt when chat mode is enabled. This is usually set automatically in the config.json file of a character folder.",
        tag=mktag(type=AT.Plumbing, group=AG.Characters),
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable streaming mode. This will print generations by the LLM piecemeal, instead of waiting for a full generation to complete. Results may be printed per-token, per-sentence, or otherwise, according to --stream_flush.",
        tag=mktag(type=AT.Porcelain, group=AG.Generation, very_important=True),
    )
    parser.add_argument(
        "--http",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a small webserver with minimal UI. By default, you'll find it at localhost:5050.",
        tag=mktag(type=AT.Porcelain, group=AG.Interface, service=True),
    )
    parser.add_argument(
        "--websock",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sending and receiving commands on a websock server running on --websock_host and --websock_port. This is enabled automatically with --http.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface, service=True, motd=True),
    )
    parser.add_argument(
        "--websock_host",
        type=str,
        default="localhost",
        help="The hostname that the websocket server binds to.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--websock_port",
        type=int,
        default=5150,
        help="The port that the websock server will listen on. By default, this is the http port +100.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--http_host",
        type=str,
        default="localhost",
        help="Hostname to bind to if --http is enabled.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--http_port",
        type=int,
        default=5050,
        help="Port for the web server to listen on if --http is provided. By default, the --audio_websock_port will be --http_port+1, and --tts_websock_port will be --http_port+2, e.g. 5051 and 5052.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--http_override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, the values of --audio_websock, --tts_websock, --audio_websock_host, --audio_websock_port, --tts_websock_host, --tts_websock_port will be overriden if --http is provided. Use --no-http_override to disable this, so you can set your own host/port values for the websock services or disable them entirely.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--multiline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Makes multiline mode the dfault, meaning that newlines no longer trigger a message being sent to the backend. instead, you must enter the value of --multiline_delimiter to trigger a send.",
        tag=mktag(type=AT.Porcelain, group=AG.Interface),
    )
    parser.add_argument(
        "--multiline_delimiter",
        type=str,
        default="\\",
        help="String that signifies the end of user input. This is only relevant for when --multiline is enabled. By default this is a backslash, inverting the normal behaviour of backslashes allowing to enter a newline ad-hoc while in multiline mode. This option is intended to be used by scripts to change the delimiter to something less common.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable colored output.",
        tag=mktag(type=AT.Porcelain, group=AG.Interface, motd=True),
    )
    parser.add_argument(
        "--text_ai_color",
        type=str,
        default="none",
        help="Color for the generated text, as long as --color is enabled. Most ANSI terminal colors are supported.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface, motd=True),
    )
    parser.add_argument(
        "--text_ai_style",
        type=str,
        default="bright",
        help="Style for the generated text, as long as --color is enabled. Most ANSI terminal styles are supported.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--dynamic_file_vars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dynamic file vars are strings of the form {[FILE1]}. If FILE1 is found, the entire expression is replaced with the contents of FILE1. This is dynamic in the sense that the contents of FILE1 are loaded each time the replacement is encountered, which is different from the normal file vars with {{FILENAME}}, which are loaded once during character initialization. Replacement happens in user inputs only. In particular, dynamic file vars are ignored in system messages or saved chats. If you want the LLM to get file contents, use tools. disabling this means no replacement happens. This can be a security vulnerability, so it is disabled by default on the API.",
        tag=mktag(type=AT.Plumbing, group=AG.Templates, motd=True),
    )
    parser.add_argument(
        "--warn_trailing_space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warn if the prompt that is sent to the backend ends on a space. This can cause e.g. excessive emoticon use by the model.",
        tag=mktag(type=AT.Plumbing, group=AG.Generation),
    )
    parser.add_argument(
        "--warn_unsupported_sampling_parameter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warn if you have set an option that is usually considered a sampling parameter, but happens to be not supported by the chose nbackend.",
        tag=mktag(type=AT.Plumbing, group=AG.SamplingParameters),
    )
    parser.add_argument(
        "--warn_audio_activation_phrase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warn if audio is being transcribed, but no activation phrase is found. Normally this only will warn once. Set to -1 if you want to be warned every time.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "--warn_hint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warn if you have a hint set.",
        tag=mktag(type=AT.Plumbing, group=AG.Generation),
    )
    parser.add_argument(
        "--json_grammar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force generation output to be in JSON format. This is equivalent to using -g with a json.gbnf grammar file, but this option is provided for convenience.",
        tag=mktag(type=AT.Porcelain, group=AG.Generation, motd=True),
    )
    parser.add_argument(
        "--stream_flush",
        type=str,
        default="token",
        help="When to flush the streaming buffer. When set to 'token', will print each token immediately. When set to 'sentence', it will wait for a complete sentence before printing. This can be useful for TTS software. When set to 'flex', will act like 'sentence', but prebuffer a minimum amount of characters before flushing, according to stream_flush_flex_value. Default is 'token'.",
        tag=mktag(type=AT.Plumbing, group=AG.Generation, motd=True),
    )
    parser.add_argument(
        "--stream_flush_flex_value",
        type=int,
        default=50,
        help="How many characters (not tokens) at least to buffer before flushing the queue, when stream_flush is set to 'flex'.",
        tag=mktag(type=AT.Plumbing, group=AG.Generation, motd=False),
    )    
    parser.add_argument(
        "--cli_prompt",
        type=str,
        default=" {{current_tokens}} > ",
        help="String to show at the bottom as command prompt. Can be empty.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface, motd=True),
    )
    parser.add_argument(
        "--cli_prompt_color",
        type=str,
        default="none",
        help="Color of the prompt. Uses names of standard ANSI terminal colors. Requires --color to be enabled.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--hint",
        type=str,
        default="",
        help="Hint for the AI. This string will be appended to the prompt behind the scenes. It's the first thing the AI sees. Try setting it to 'Of course,' to get a more compliant AI. Also refered to as 'prefill'.",
        tag=mktag(type=AT.Porcelain, group=AG.Generation),
    )
    parser.add_argument(
        "--hint_sticky",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If disabled, hint will be shown to the AI as part of prompt, but will be omitted from the story.",
        tag=mktag(type=AT.Plumbing, group=AG.Generation),
    )
    parser.add_argument(
        "--tts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable text to speech on generated text.",
        tag=mktag(type=AT.Porcelain, group=AG.TTS, very_important=True, service=True),
    )
    parser.add_argument(
        "--tts_model",
        type=str,
        default=TTSModel.kokoro.name,
        help="The TTS model to use. This is ignored unless you use ghostbox-tts as your tts_program. Options are:  "
        + ", ".join([m.name for m in TTSModel]),
        tag=mktag(type=AT.Porcelain, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_zonos_model",
        type=str,
        default=ZonosTTSModel.hybrid.name,
        help="The Zonos TTS model offers two architecural variants: A pure transformer implementation or a transformer-mamba hybrid variant. Hybrid usually gives the best results, but requires flash attention. This option has no effect on non-zonos TTS engines. Options are: "
        + ", ".join([m.name for m in ZonosTTSModel]),
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_orpheus_model",
        type=str,
        default="",
        help="Filepath to gguf file or huggingface repo name of the orpheus model you wish to use with the tts. Leave empty for some reasonable defaults. If you don't use the default, and set this to a repo, the underlying ghostbox-tts might silently download the snapshot, so check with /ttsdebug. This option is ignored unless you actually set tts_model to orpheus.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_llm_server",
        type=str,
        default="",
        help="Address of another LLM server that serves a TTS model on an OpenAI compatible backend. This is only relevant if you use a tts model that needs an LLM as backend. This is currently the case with orpheus. You can also let ghostbox-tts spawn a server on its own if you have llama-server in your path, by setting tts_orpheus_model to an orpheus gguf file or huggingface repository.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_language",
        type=str,
        default="en",
        help="Set the TTS voice language. Right now, this is only relevant for kokoro.",
        tag=mktag(type=AT.Porcelain, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_output_method",
        type=str,
        choices=[om.name for om in TTSOutputMethod],
        default=TTSOutputMethod.default.name,
        help="How to play the generated speech. Using the --http argument automatically sets this to websock.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_websock",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable websock as the output method for TTS. This is equivalent to `--tts_output_method websock`.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_websock_host",
        type=str,
        default="localhost",
        help="The address to bind to for the underlying TTS program when using websock as output method. ghostbox-tts only. This option is normally overriden by --http.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_websock_port",
        type=int,
        default=5052,
        help="The port to listen on for the underlying TTS program when using websock as output method. ghostbox-tts only. This option is normally overriden by --http.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_interrupt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop an ongoing TTS whenever a new generation is spoken. When set to false, will queue messages instead.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_program",
        type=str,
        default="ghostbox-tts",
        help="Path to a TTS (Text-to-speech) program to verbalize generated text. The TTS program should read lines from standard input. Many examples are provided in scripts/ghostbox-tts-* . The ghostbox-tts script offers a native solution using various supported models.",
        tag=mktag(type=AT.Porcelain, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_clone_dir",
        type=str,
        default="",
        help="Directory to check first for wave files used in voice cloning. Note that voice cloning isn't supported by all tts models.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_tortoise_quality",
        type=str,
        default="fast",
        help="Quality preset. tortoise-tts only. Can be 'ultra_fast', 'fast' (default), 'standard', or 'high_quality'",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_volume",
        type=float,
        default=1.0,
        help="Volume for TTS voice program. Is passed to tts_program as environment variable.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_modify_system_msg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, instructions specific to the underlying TTS model will be appended to the system prompt. This may e.g. point out usage of tags like <laugh> and <cough> to the LLM. Support varies depending on model.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )    
    parser.add_argument(
        "--tts_rate",
        type=int,
        default=50,
        help="Speaking rate for TTS voice program. Is passed to tts_program as environment variable. Note that speaking rate is not supported by all TTS engines.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_additional_arguments",
        type=str,
        default="",
        help="Additional command line arguments that will be passed to the tts_program.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--image_watch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable watching of a directory for new images. If a new image appears in the folder, the image will be loaded with id 0 and sent to the backend. works with multimodal models only (like llava).",
        tag=mktag(type=AT.Porcelain, group=AG.Images, service=True, motd=True),
    )
    parser.add_argument(
        "--image_watch_dir",
        type=str,
        default=os.path.expanduser("~/Pictures/Screenshots/"),
        help="Directory that will be watched for new image files when --image_watch is enabled.",
        tag=mktag(type=AT.Plumbing, group=AG.Images),
    )
    parser.add_argument(
        "--image_watch_msg",
        type=str,
        default="Can you describe this image?",
        help="If image_watch is enabled, this message will be automatically send to the backend whenever a new image is detected. Set this to '' to disable automatic messages, while still keeping the automatic update the image with id 0.",
        tag=mktag(type=AT.Plumbing, group=AG.Images, motd=True),
    )
    parser.add_argument(
        "--image_watch_hint",
        type=str,
        default="",
        help="If image_watch is enabled, this string will be sent to the backend as start of the AI response whenever a new image is detected and automatically described. This allows you to guide or solicit the AI by setting it to e.g. 'Of course, this image show' or similar. Default is ''",
        tag=mktag(type=AT.Plumbing, group=AG.Images, motd=True),
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base.en",
        help="Name of the model to use for transcriptions using the openai whisper model. Default is 'base.en'. For a list of model names, see https://huggingface.co/openai/whisper-large",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "-y",
        "--tts_voice",
        type=str,
        default="random",
        help="Voice to use for TTS. Default is 'random', which is a special value that picks a random available voice for your chosen tts_program. The value of tts_voice will be changed at startup if random is chosen, so when you find a voice you like you can find out its name with /lsoptions and checking tts_voice. To get a list of voices, start ghostbox with your desired tts model and do /lsvoices.",
        tag=mktag(type=AT.Porcelain, group=AG.TTS),
    )
    parser.add_argument(
        "--tts_subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable printing of generated text while TTS is enabled.",
        tag=mktag(type=AT.Plumbing, group=AG.TTS),
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="Path to a config fail in JSON format, containing a dictionary with OPTION : VALUE pairs to be loaded on startup. Same as /loadconfig or /loadoptions. To produce an example config-file, try /saveconfig example.json.",
        tag=mktag(type=AT.Porcelain, group=AG.General),
    )
    parser.add_argument(
        "--chat_show_ai_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Controls wether to show AI prompt in chat mode. Specifically, assuming chat_ai = 'Bob', setting chat_show_ai_prompt to True will show 'Bob: ' in front of the AI's responses. Note that this is always sent to the back-end (in chat mode), this parameter merely controls wether it is shown.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--smart_context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enables ghostbox version of smart context, which means dropping text at user message boundaries when the backend's context is exceeded. If you disable this, it usually means the backend will truncate the raw message. Enabling smart context means better responses and longer processing time due to cache invalidation, disabling it means worse responses with faster processing time. Note from marius: Beware I haven't looked at this in a while, since newer models all have very large contexts.",
        tag=mktag(type=AT.Plumbing, group=AG.Generation, motd=True),
    )
    parser.add_argument(
        "--hide",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hides some unnecessary output, providing a more immersive experience. Same as typing /hide.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface, motd=True),
    )
    parser.add_argument(
        "--audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable automatic transcription of audio input using openai whisper model. Obviously, you need a mic for this.",
        tag=mktag(type=AT.Porcelain, group=AG.Audio, service=True, very_important=True),
    )
    parser.add_argument(
        "--audio_silence_threshold",
        type=int,
        default=2000,
        help="An integer value denoting the threshold for when automatic audio transcription starts recording. (default 2000)",
        tag=mktag(type=AT.Plumbing, group=AG.Audio, motd=True),
    )
    parser.add_argument(
        "--audio_activation_phrase",
        type=str,
        default="",
        help="When set, the phrase must be detected in the beginning of recorded audio, or the recording will be ignored. Phrase matching is fuzzy with punctuation removed.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio, motd=True),
    )
    parser.add_argument(
        "--audio_activation_period_ms",
        type=int,
        default=0,
        help="Period in milliseconds where no further activation phrase is necessary to trigger a response. The period starts after any interaction with the AI, spoken or otherwise.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "--audio_interrupt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stops generation and TTS  when you start speaking. Does not require activation phrase.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "--audio_activation_phrase_keep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false and an activation phrase is set, the triggering phrase will be removed in messages that are sent to the backend.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "--audio_show_transcript",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show transcript of recorded user speech when kaudio transcribing is enabled. When disabled, you can still see the full transcript with /log or /print.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "--audio_websock",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable to listen for audio on an HTTP websocket at the given `--websock_url`, instead of recording audio from a microphone. This can be used to stream audio through a website. This is enabled by default with the --http option unless you also supply --no-http_overrid.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "--audio_websock_host",
        type=str,
        default="localhost",
        help="The address to bind to when `--audio_websock` is enabled. You can stream audio to this endpoint using the websocket protocol for audio transcription. Normally overriden by --http.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "--audio_websock_port",
        type=int,
        default=5051,
        help="The port to listen on when `--audio_websock` is enabled. You can stream audio to this endpoint using the websocket protocol for audio transcription. Normally overriden by --http.",
        tag=mktag(type=AT.Plumbing, group=AG.Audio),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show additional output for various things.",
        tag=mktag(type=AT.Plumbing, group=AG.General),
    )
    parser.add_argument(
        "--log_time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print timing and performance statistics to stderr with every generation. Auto enabled for the API.",
        tag=mktag(type=AT.Plumbing, group=AG.General),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prevents printing and TTS vocalization of generations. Often used with the API when you want to handle generation results yourself and don't want printing to console.",
        tag=mktag(type=AT.Porcelain, group=AG.Interface),
    )
    parser.add_argument(
        "--stderr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wether printing to stderr is enabled. You may want to disable this when building terminal applications using the API.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--stdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wether printing to stdout is enabled. You may want to disable this when building terminal applications using the API.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--expand_user_input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand variables in user input. E.g. {$var} will be replaced with content of var. Variables are initialized from character folders (i.e. file 'memory' will be {$memory}), or can be set manually with the /varfile command or --varfile option. See also --dynamic_file_vars.",
        tag=mktag(type=AT.Plumbing, group=AG.Interface),
    )
    parser.add_argument(
        "--tools_unprotected_shell_access",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow an AI to run shell commands, even if not logged in to their own account. The safe way of doing this is to create an account on your system with the same name as the AI, and then run this program under their account. If you don't want to do that, and you are ok with an AI deleting your files through accident or malice, set this flag to true.",
        tag=mktag(type=AT.Plumbing, group=AG.Tools),
    )
    # parser.add_argument("--tools_reflection", action=argparse.BooleanOptionalAction, default=True, help="Continue generation after tools have been applied. This allows the AI to reflect on the results, e.g. summarize a file it retrieved. Only applies when use_tools is true.",
    # tag=mktag(type=AT.Plumbing, group=AG.Tools))
    parser.add_argument(
        "-d",
        "--tools_forbidden",
        action="append",
        default=["Any", "List", "Dict", "launch_nukes"],
        help="Blacklist certain tools. Specify multiple times to forbid several tools. The default blacklist contains some common module imports that can pollute a tools.py namespace. You can override this in a character folders config.json if necessary.",
        tag=mktag(type=AT.Plumbing, group=AG.Tools, very_important=True, motd=True),
    )
    parser.add_argument(
        "--tools_hint",
        type=str,
        default="",
        help="Text that will be appended to the system prompt when use_tools is true.",
        tag=mktag(type=AT.Plumbing, group=AG.Tools),
    )
    parser.add_argument(
        "--tools_inject_dependency_function",
        type=str,
        default="",
        help="API only. Set a callback function to be called whenever an tool-using Ai is initialized. The callback will receive one argument: The tools.py module. You can use this to inject dependency or modify the module after it is loaded.",
        tag=mktag(type=AT.Plumbing, group=AG.Tools, motd=True),
    )
    parser.add_argument(
        "--tools_inject_ghostbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject a reference to ghostbox itself into an AI's tool module. This will make the '_ghostbox_plumbing' identifier available in the tools module and point it to the running ghostbox Plumbing instance. Disabling this will break many of the standard AI tools that ship with ghostbox.",
        tag=mktag(type=AT.Plumbing, group=AG.Tools),
    )
    parser.add_argument(
        "--use_tools",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable use of tools, i.e. model may call python functions. This will do nothing if tools.py isn't present in the char directory. If tools.py is found, this will be automatically enabled.",
        tag=mktag(type=AT.Porcelain, group=AG.Tools, very_important=True, motd=True),
    )
    parser.add_argument(
        "-x",
        "--var_file",
        action="append",
        default=[],
        help="Files that will be added to the list of variables that can be expanded. E.g. -Vmemory means {$memory} will be expanded to the contents of file memory, provided expand_user_input is set. Can be used to override values set in character folders. Instead of using this, you can also just type {[FILENAME]} to have it be automatically expanded with the contents of FILENAME, provided --dynamic_file_vars is enabled.",
        tag=mktag(type=AT.Porcelain, group=AG.Interface, motd=True),
    )

    # don't show all possible parameters on command line, but do show some
    params = backends.supported_parameters | backends.sometimes_parameters
    for name, param in params.items():
        # we need to avoid some redefinitions
        if name in "stop max_length".split(" "):
            continue
        # others require special treatment
        if name == "samplers":
            parser.add_argument(
                "-S",
                "--" + param.name,
                nargs="*",
                default=param.default_value,
                help=param.description,
                tag=mktag(
                    type=AT.Plumbing,
                    group=AG.SamplingParameters,
                    very_important=True,
                    motd=True,
                    is_option=True,
                ),
            )
            continue

        parser.add_argument(
            "--" + param.name,
            type=type(param.default_value),
            default=param.default_value,
            help=param.description,
            tag=mktag(type=AT.Plumbing, group=AG.SamplingParameters, is_option=True),
        )
    return parser


def makeDefaultOptions():
    """Returns a pair of default options and tags."""
    tp = makeTaggedParser(backends.default_params)
    parser = tp.get_parser()
    return parser.parse_args(args=""), tp.get_tags()
