# Full list of ghostbox options

Below is a full list of ghostbox options, with descriptions and defaults.

You can generate this list yoruself with

```bash
ghostbox
/help options --markdown
```

or simply list all options and their current values with `/lsoptions`


## `api_key`
API key for OpenAI. Without the `--backend openai` option, this has no effect.

```
It is a plumbing option in the openai group. You can set it with `/set
api_key VALUE` or provide it as a command line parameter with
`--api_key`
Its default value is ""
```

## `audio`
Enable automatic transcription of audio input using openai whisper model. Obviously, you need a mic for this.

```
It is a porcelain option in the audio group. You can set it with `/set
audio VALUE` or provide it as a command line parameter with `--audio`
Its default value is False
Setting it to True will start the corresponding service.
```

## `audio_activation_period_ms`
Period in milliseconds where no further activation phrase is necessary to trigger a response. The period starts after any interaction with the AI, spoken or otherwise.

```
It is a plumbing option in the audio group. You can set it with `/set
audio_activation_period_ms VALUE` or provide it as a command line
parameter with `--audio_activation_period_ms`
Its default value is 0
```

## `audio_activation_phrase`
When set, the phrase must be detected in the beginning of recorded audio, or the recording will be ignored. Phrase matching is fuzzy with punctuation removed.

```
It is a plumbing option in the audio group. You can set it with `/set
audio_activation_phrase VALUE` or provide it as a command line
parameter with `--audio_activation_phrase`
Its default value is ""
```

## `audio_activation_phrase_keep`
If false and an activation phrase is set, the triggering phrase will be removed in messages that are sent to the backend.

```
It is a plumbing option in the audio group. You can set it with `/set
audio_activation_phrase_keep VALUE` or provide it as a command line
parameter with `--audio_activation_phrase_keep`
Its default value is True
```

## `audio_interrupt`
Stops generation and TTS  when you start speaking. Does not require activation phrase.

```
It is a plumbing option in the audio group. You can set it with `/set
audio_interrupt VALUE` or provide it as a command line parameter with
`--audio_interrupt`
Its default value is True
```

## `audio_show_transcript`
Show transcript of recorded user speech when kaudio transcribing is enabled. When disabled, you can still see the full transcript with /log or /print.

```
It is a plumbing option in the audio group. You can set it with `/set
audio_show_transcript VALUE` or provide it as a command line parameter
with `--audio_show_transcript`
Its default value is True
```

## `audio_silence_threshold`
An integer value denoting the threshold for when automatic audio transcription starts recording. (default 2000)

```
It is a plumbing option in the audio group. You can set it with `/set
audio_silence_threshold VALUE` or provide it as a command line
parameter with `--audio_silence_threshold`
Its default value is 2000
```

## `audio_websock`
Enable to listen for audio on an HTTP websocket at the given `--websock_url`, instead of recording audio from a microphone. This can be used to stream audio through a website. This is enabled by default with the --http option unless you also supply --no-http_overrid.

```
It is a plumbing option in the audio group. You can set it with `/set
audio_websock VALUE` or provide it as a command line parameter with
`--audio_websock`
Its default value is False
```

## `audio_websock_host`
The address to bind to when `--audio_websock` is enabled. You can stream audio to this endpoint using the websocket protocol for audio transcription. Normally overriden by --http.

```
It is a plumbing option in the audio group. You can set it with `/set
audio_websock_host VALUE` or provide it as a command line parameter
with `--audio_websock_host`
Its default value is localhost
```

## `audio_websock_port`
The port to listen on when `--audio_websock` is enabled. You can stream audio to this endpoint using the websocket protocol for audio transcription. Normally overriden by --http.

```
It is a plumbing option in the audio group. You can set it with `/set
audio_websock_port VALUE` or provide it as a command line parameter
with `--audio_websock_port`
Its default value is 5051
```

## `backend`
Backend to use. The default is `generic`, which conforms to the OpenAI REST API, and is supported by most LLM providers. Choosing a more specific backend may provide additional functionality. Other possible values are generic, legacy, llamacpp, koboldcpp, openai, dummy.

```
It is a porcelain option in the backend group. You can set it with
`/set backend VALUE` or provide it as a command line parameter with
`--backend`
Its default value is generic
```

## `cache_prompt`
Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests. Because (depending on the backend) the logits are **not** guaranteed to be bit-for-bit identical for different batch sizes (prompt processing vs. token generation) enabling this option can cause nondeterministic results.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set cache_prompt VALUE` or provide it as a command line
parameter with `--cache_prompt`
Its default value is True
```

## `character_folder`
character folder to load at startup. The folder may contain a `system_msg` file, a `config.json`, and a `tools.py`, as well as various other files used as file variables. See the examples and documentation for more.

```
It is a porcelain option in the characters group. You can set it with
`/set character_folder VALUE` or provide it as a command line
parameter with `--character_folder`
Its default value is ""
```

## `chat_ai`
Name  the AI will have when chatting. Has various effects on the prompt when chat mode is enabled. This is usually set automatically in the config.json file of a character folder.

```
It is a plumbing option in the characters group. You can set it with
`/set chat_ai VALUE` or provide it as a command line parameter with
`--chat_ai`
Its default value is ""
```

## `chat_show_ai_prompt`
Controls wether to show AI prompt in chat mode. Specifically, assuming chat_ai = 'Bob', setting chat_show_ai_prompt to True will show 'Bob: ' in front of the AI's responses. Note that this is always sent to the back-end (in chat mode), this parameter merely controls wether it is shown.

```
It is a plumbing option in the interface group. You can set it with
`/set chat_show_ai_prompt VALUE` or provide it as a command line
parameter with `--chat_show_ai_prompt`
Its default value is True
```

## `chat_user`
Username you wish to be called when chatting in 'chat' mode. It will also replace occurrences of {chat_user} anywhere in the character files. If you don't provide one here, your username will be determined by your OS session login.

```
It is a porcelain option in the general group. You can set it with
`/set chat_user VALUE` or provide it as a command line parameter with
`--chat_user`
Its default value is user
```

## `cli_prompt`
String to show at the bottom as command prompt. Can be empty.

```
It is a plumbing option in the interface group. You can set it with
`/set cli_prompt VALUE` or provide it as a command line parameter with
`--cli_prompt`
Its default value is  {{current_tokens}} > 
```

## `cli_prompt_color`
Color of the prompt. Uses names of standard ANSI terminal colors. Requires --color to be enabled.

```
It is a plumbing option in the interface group. You can set it with
`/set cli_prompt_color VALUE` or provide it as a command line
parameter with `--cli_prompt_color`
Its default value is none
```

## `color`
Enable colored output.

```
It is a porcelain option in the interface group. You can set it with
`/set color VALUE` or provide it as a command line parameter with
`--color`
Its default value is True
```

## `config_file`
Path to a config fail in JSON format, containing a dictionary with OPTION : VALUE pairs to be loaded on startup. Same as /loadconfig or /loadoptions. To produce an example config-file, try /saveconfig example.json.

```
It is a porcelain option in the general group. You can set it with
`/set config_file VALUE` or provide it as a command line parameter
with `--config_file`
Its default value is ""
```

## `dry_multiplier`
Set the DRY (Don't Repeat Yourself) repetition penalty multiplier.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set dry_multiplier VALUE` or provide it as a command line
parameter with `--dry_multiplier`
Its default value is 0.8
```

## `dynamic_file_vars`
Dynamic file vars are strings of the form {[FILE1]}. If FILE1 is found, the entire expression is replaced with the contents of FILE1. This is dynamic in the sense that the contents of FILE1 are loaded each time the replacement is encountered, which is different from the normal file vars with {{FILENAME}}, which are loaded once during character initialization. Replacement happens in user inputs only. In particular, dynamic file vars are ignored in system messages or saved chats. If you want the LLM to get file contents, use tools. disabling this means no replacement happens. This can be a security vulnerability, so it is disabled by default on the API.

```
It is a plumbing option in the templates group. You can set it with
`/set dynamic_file_vars VALUE` or provide it as a command line
parameter with `--dynamic_file_vars`
Its default value is True
```

## `endpoint`
Address of backend http endpoint. This is a URL that is dependent on the backend you use, though the default of localhost:8080 works for most, including Llama.cpp and Kobold.cpp. If you want to connect to an online provider that is not part of the explicitly supported backends, this is where you would supply their API address.

```
It is a porcelain option in the backend group. You can set it with
`/set endpoint VALUE` or provide it as a command line parameter with
`--endpoint`
Its default value is http://localhost:8080
```

## `expand_user_input`
Expand variables in user input. E.g. {$var} will be replaced with content of var. Variables are initialized from character folders (i.e. file 'memory' will be {$memory}), or can be set manually with the /varfile command or --varfile option. See also --dynamic_file_vars.

```
It is a plumbing option in the interface group. You can set it with
`/set expand_user_input VALUE` or provide it as a command line
parameter with `--expand_user_input`
Its default value is True
```

## `force_params`
Force sending of sample parameters, even when they are seemingly not supported by the backend (use to debug or with generic

```
It is a plumbing option in the backend group. You can set it with
`/set force_params VALUE` or provide it as a command line parameter
with `--force_params`
Its default value is False
```

## `frequency_penalty`
Repeat alpha frequency penalty.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set frequency_penalty VALUE` or provide it as a command line
parameter with `--frequency_penalty`
Its default value is 0.0
```

## `grammar_file`
Grammar file used to restrict generation output. Grammar format is GBNF.

```
It is a plumbing option in the generation group. You can set it with
`/set grammar_file VALUE` or provide it as a command line parameter
with `--grammar_file`
Its default value is ""
```

## `hide`
Hides some unnecessary output, providing a more immersive experience. Same as typing /hide.

```
It is a plumbing option in the interface group. You can set it with
`/set hide VALUE` or provide it as a command line parameter with
`--hide`
Its default value is False
```

## `hint`
Hint for the AI. This string will be appended to the prompt behind the scenes. It's the first thing the AI sees. Try setting it to 'Of course,' to get a more compliant AI. Also refered to as 'prefill'.

```
It is a porcelain option in the generation group. You can set it with
`/set hint VALUE` or provide it as a command line parameter with
`--hint`
Its default value is ""
```

## `hint_sticky`
If disabled, hint will be shown to the AI as part of prompt, but will be omitted from the story.

```
It is a plumbing option in the generation group. You can set it with
`/set hint_sticky VALUE` or provide it as a command line parameter
with `--hint_sticky`
Its default value is True
```

## `http`
Enable a small webserver with minimal UI. By default, you'll find it at localhost:5050.

```
It is a porcelain option in the interface group. You can set it with
`/set http VALUE` or provide it as a command line parameter with
`--http`
Its default value is False
Setting it to True will start the corresponding service.
```

## `http_host`
Hostname to bind to if --http is enabled.

```
It is a plumbing option in the interface group. You can set it with
`/set http_host VALUE` or provide it as a command line parameter with
`--http_host`
Its default value is localhost
```

## `http_override`
If enabled, the values of --audio_websock, --tts_websock, --audio_websock_host, --audio_websock_port, --tts_websock_host, --tts_websock_port will be overriden if --http is provided. Use --no-http_override to disable this, so you can set your own host/port values for the websock services or disable them entirely.

```
It is a plumbing option in the interface group. You can set it with
`/set http_override VALUE` or provide it as a command line parameter
with `--http_override`
Its default value is True
```

## `http_port`
Port for the web server to listen on if --http is provided. By default, the --audio_websock_port will be --http_port+1, and --tts_websock_port will be --http_port+2, e.g. 5051 and 5052.

```
It is a plumbing option in the interface group. You can set it with
`/set http_port VALUE` or provide it as a command line parameter with
`--http_port`
Its default value is 5050
```

## `image_watch`
Enable watching of a directory for new images. If a new image appears in the folder, the image will be loaded with id 0 and sent to the backend. works with multimodal models only (like llava).

```
It is a porcelain option in the images group. You can set it with
`/set image_watch VALUE` or provide it as a command line parameter
with `--image_watch`
Its default value is False
Setting it to True will start the corresponding service.
```

## `image_watch_dir`
Directory that will be watched for new image files when --image_watch is enabled.

```
It is a plumbing option in the images group. You can set it with `/set
image_watch_dir VALUE` or provide it as a command line parameter with
`--image_watch_dir`
Its default value is /home/marius/Pictures/Screenshots/
```

## `image_watch_hint`
If image_watch is enabled, this string will be sent to the backend as start of the AI response whenever a new image is detected and automatically described. This allows you to guide or solicit the AI by setting it to e.g. 'Of course, this image show' or similar. Default is ''

```
It is a plumbing option in the images group. You can set it with `/set
image_watch_hint VALUE` or provide it as a command line parameter with
`--image_watch_hint`
Its default value is ""
```

## `image_watch_msg`
If image_watch is enabled, this message will be automatically send to the backend whenever a new image is detected. Set this to '' to disable automatic messages, while still keeping the automatic update the image with id 0.

```
It is a plumbing option in the images group. You can set it with `/set
image_watch_msg VALUE` or provide it as a command line parameter with
`--image_watch_msg`
Its default value is Can you describe this image?
```

## `include`
Include paths that will be searched for character folders named with the /start command or the --character_folder command line argument.

```
It is a porcelain option in the characters group. You can set it with
`/set include VALUE` or provide it as a command line parameter with
`--include`
Its default value is ['/home/marius/.local/share/ghostbox/chars', '/home/marius/prog/ai/ghostbox/ghostbox/data/chars/', 'chars/']
```

## `json`
Force generation output to be in JSON format. This is equivalent to using -g with a json.gbnf grammar file, but this option is provided for convenience.

```
It is a porcelain option in the generation group. You can set it with
`/set json VALUE` or provide it as a command line parameter with
`--json`
Its default value is False
```

## `log_time`
Print timing and performance statistics to stderr with every generation. Auto enabled for the API.

```
It is a plumbing option in the general group. You can set it with
`/set log_time VALUE` or provide it as a command line parameter with
`--log_time`
Its default value is False
```

## `max_context_length`
Maximum number of tokens to keep in context.

```
It is a porcelain option in the generation group. You can set it with
`/set max_context_length VALUE` or provide it as a command line
parameter with `--max_context_length`
Its default value is 32768
```

## `max_length`
Set the maximum number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set max_length VALUE` or provide it as a command line
parameter with `--max_length`
Its default value is -1
```

## `min_p`
The minimum probability for a token to be considered, relative to the probability of the most likely token.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set min_p VALUE` or provide it as a command line parameter
with `--min_p`
Its default value is 0.05
```

## `mirostat`
Enable Mirostat sampling, controlling perplexity during text generation.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set mirostat VALUE` or provide it as a command line
parameter with `--mirostat`
Its default value is 0
```

## `mirostat_eta`
Set the Mirostat learning rate, parameter eta.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set mirostat_eta VALUE` or provide it as a command line
parameter with `--mirostat_eta`
Its default value is 0.1
```

## `mirostat_tau`
Set the Mirostat target entropy, parameter tau.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set mirostat_tau VALUE` or provide it as a command line
parameter with `--mirostat_tau`
Its default value is 5.0
```

## `mode`
Mode of operation. Changes various things behind-the-scenes. Values are currently 'default', or 'chat'.

```
It is a plumbing option in the templates group. You can set it with
`/set mode VALUE` or provide it as a command line parameter with
`--mode`
Its default value is default
```

## `model`
LLM to use for requests. This only works if the backend supports choosing models.

```
It is a porcelain option in the backend group. You can set it with
`/set model VALUE` or provide it as a command line parameter with
`--model`
```

## `multiline`
Makes multiline mode the dfault, meaning that newlines no longer trigger a message being sent to the backend. instead, you must enter the value of --multiline_delimiter to trigger a send.

```
It is a porcelain option in the interface group. You can set it with
`/set multiline VALUE` or provide it as a command line parameter with
`--multiline`
Its default value is False
```

## `multiline_delimiter`
String that signifies the end of user input. This is only relevant for when --multiline is enabled. By default this is a backslash, inverting the normal behaviour of backslashes allowing to enter a newline ad-hoc while in multiline mode. This option is intended to be used by scripts to change the delimiter to something less common.

```
It is a plumbing option in the interface group. You can set it with
`/set multiline_delimiter VALUE` or provide it as a command line
parameter with `--multiline_delimiter`
Its default value is \
```

## `presence_penalty`
Repeat alpha presence penalty.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set presence_penalty VALUE` or provide it as a command line
parameter with `--presence_penalty`
Its default value is 0.0
```

## `prompt`
If provided, process the prompt and exit.

```
It is a porcelain option in the generation group. You can set it with
`/set prompt VALUE` or provide it as a command line parameter with
`--prompt`
Its default value is None
```

## `prompt_format`
Prompt format template to use. The default is 'auto', which means ghostbox will .let the backend handle templating, which is usually the right choice. You can still use other settings, like 'raw', to experiment. This is ignored if you use the generic or openai backend. Note: Prompt format templates used to be more important in the early days of LLMs, as confusion was rampant and mistakes were not uncommon even in official releases. Nowadays, it is quite safe to use the official templates. You may still want to use this option for experimentation, however.

```
It is a plumbing option in the templates group. You can set it with
`/set prompt_format VALUE` or provide it as a command line parameter
with `--prompt_format`
Its default value is auto
```

## `quiet`
Prevents printing and TTS vocalization of generations. Often used with the API when you want to handle generation results yourself and don't want printing to console.

```
It is a porcelain option in the interface group. You can set it with
`/set quiet VALUE` or provide it as a command line parameter with
`--quiet`
Its default value is False
```

## `repeat_penalty`
Control the repetition of token sequences in the generated text.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set repeat_penalty VALUE` or provide it as a command line
parameter with `--repeat_penalty`
Its default value is 1.1
```

## `samplers`
The order the samplers should be applied in. An array of strings representing sampler type names. If a sampler is not set, it will not be used. If a sampler is specified more than once, it will be applied multiple times.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set samplers VALUE` or provide it as a command line
parameter with `--samplers`
Its default value is ['min_p', 'xtc', 'dry', 'temperature']
```

## `smart_context`
Enables ghostbox version of smart context, which means dropping text at user message boundaries when the backend's context is exceeded. If you disable this, it usually means the backend will truncate the raw message. Enabling smart context means better responses and longer processing time due to cache invalidation, disabling it means worse responses with faster processing time. Note from marius: Beware I haven't looked at this in a while, since newer models all have very large contexts.

```
It is a plumbing option in the generation group. You can set it with
`/set smart_context VALUE` or provide it as a command line parameter
with `--smart_context`
Its default value is True
```

## `stderr`
Wether printing to stderr is enabled. You may want to disable this when building terminal applications using the API.

```
It is a plumbing option in the interface group. You can set it with
`/set stderr VALUE` or provide it as a command line parameter with
`--stderr`
Its default value is True
```

## `stop`
Specify a JSON array of stopping strings. These words will not be included in the completion, so make sure to add them to the prompt for the next iteration.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set stop VALUE` or provide it as a command line parameter
with `--stop`
Its default value is []
```

## `stream`
Enable streaming mode. This will print generations by the LLM piecemeal, instead of waiting for a full generation to complete. Results may be printed per-token, per-sentence, or otherwise, according to --stream_flush.

```
It is a porcelain option in the generation group. You can set it with
`/set stream VALUE` or provide it as a command line parameter with
`--stream`
Its default value is True
```

## `stream_flush`
When to flush the streaming buffer. When set to 'token', will print each token immediately. When set to 'sentence', it will wait for a complete sentence before printing. This can be useful for TTS software. Default is 'token'.

```
It is a plumbing option in the generation group. You can set it with
`/set stream_flush VALUE` or provide it as a command line parameter
with `--stream_flush`
Its default value is token
```

## `temperature`
Adjust the randomness of the generated text.

```
It is a porcelain option in the samplingparameters group. You can set
it with `/set temperature VALUE` or provide it as a command line
parameter with `--temperature`
Its default value is 0.8
```

## `template_include`
Include paths that will be searched for prompt templates. You can specify a template to use with the -T option.

```
It is a plumbing option in the templates group. You can set it with
`/set template_include VALUE` or provide it as a command line
parameter with `--template_include`
Its default value is ['/home/marius/.local/share/ghostbox/templates', '/home/marius/prog/ai/ghostbox/ghostbox/data/templates/', 'templates/']
```

## `text_ai_color`
Color for the generated text, as long as --color is enabled. Most ANSI terminal colors are supported.

```
It is a plumbing option in the interface group. You can set it with
`/set text_ai_color VALUE` or provide it as a command line parameter
with `--text_ai_color`
Its default value is none
```

## `text_ai_style`
Style for the generated text, as long as --color is enabled. Most ANSI terminal styles are supported.

```
It is a plumbing option in the interface group. You can set it with
`/set text_ai_style VALUE` or provide it as a command line parameter
with `--text_ai_style`
Its default value is bright
```

## `tools_forbidden`
Blacklist certain tools. Specify multiple times to forbid several tools. The default blacklist contains some common module imports that can pollute a tools.py namespace. You can override this in a character folders config.json if necessary.

```
It is a plumbing option in the tools group. You can set it with `/set
tools_forbidden VALUE` or provide it as a command line parameter with
`--tools_forbidden`
Its default value is ['List', 'Dict', 'launch_nukes']
```

## `tools_hint`
Text that will be appended to the system prompt when use_tools is true.

```
It is a plumbing option in the tools group. You can set it with `/set
tools_hint VALUE` or provide it as a command line parameter with
`--tools_hint`
Its default value is ""
```

## `tools_inject_dependency_function`
API only. Set a callback function to be called whenever an tool-using Ai is initialized. The callback will receive one argument: The tools.py module. You can use this to inject dependency or modify the module after it is loaded.

```
It is a plumbing option in the tools group. You can set it with `/set
tools_inject_dependency_function VALUE` or provide it as a command
line parameter with `--tools_inject_dependency_function`
Its default value is ""
```

## `tools_inject_ghostbox`
Inject a reference to ghostbox itself into an AI's tool module. This will make the '_ghostbox_plumbing' identifier available in the tools module and point it to the running ghostbox Plumbing instance. Disabling this will break many of the standard AI tools that ship with ghostbox.

```
It is a plumbing option in the tools group. You can set it with `/set
tools_inject_ghostbox VALUE` or provide it as a command line parameter
with `--tools_inject_ghostbox`
Its default value is True
```

## `tools_unprotected_shell_access`
Allow an AI to run shell commands, even if not logged in to their own account. The safe way of doing this is to create an account on your system with the same name as the AI, and then run this program under their account. If you don't want to do that, and you are ok with an AI deleting your files through accident or malice, set this flag to true.

```
It is a plumbing option in the tools group. You can set it with `/set
tools_unprotected_shell_access VALUE` or provide it as a command line
parameter with `--tools_unprotected_shell_access`
Its default value is False
```

## `top_p`
Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.

```
It is a porcelain option in the samplingparameters group. You can set
it with `/set top_p VALUE` or provide it as a command line parameter
with `--top_p`
Its default value is 0.95
```

## `tts`
Enable text to speech on generated text.

```
It is a porcelain option in the tts group. You can set it with `/set
tts VALUE` or provide it as a command line parameter with `--tts`
Its default value is False
Setting it to True will start the corresponding service.
```

## `tts_additional_arguments`
Additional command line arguments that will be passed to the tts_program.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_additional_arguments VALUE` or provide it as a command line
parameter with `--tts_additional_arguments`
Its default value is ""
```

## `tts_interrupt`
Stop an ongoing TTS whenever a new generation is spoken. When set to false, will queue messages instead.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_interrupt VALUE` or provide it as a command line parameter with
`--tts_interrupt`
Its default value is True
```

## `tts_language`
Set the TTS voice language. Right now, this is only relevant for kokoro.

```
It is a porcelain option in the tts group. You can set it with `/set
tts_language VALUE` or provide it as a command line parameter with
`--tts_language`
Its default value is en
```

## `tts_model`
The TTS model to use. This is ignored unless you use ghostbox-tts as your tts_program. Options are:  zonos, kokoro, xtts, polly

```
It is a porcelain option in the tts group. You can set it with `/set
tts_model VALUE` or provide it as a command line parameter with
`--tts_model`
Its default value is kokoro
```

## `tts_output_method`
How to play the generated speech. Using the --http argument automatically sets this to websock.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_output_method VALUE` or provide it as a command line parameter
with `--tts_output_method`
Its default value is default
```

## `tts_program`
Path to a TTS (Text-to-speech) program to verbalize generated text. The TTS program should read lines from standard input. Many examples are provided in scripts/ghostbox-tts-* . The ghostbox-tts script offers a native solution using various supported models.

```
It is a porcelain option in the tts group. You can set it with `/set
tts_program VALUE` or provide it as a command line parameter with
`--tts_program`
Its default value is ghostbox-tts
```

## `tts_rate`
Speaking rate for TTS voice program. Is passed to tts_program as environment variable. Note that speaking rate is not supported by all TTS engines.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_rate VALUE` or provide it as a command line parameter with
`--tts_rate`
Its default value is 50
```

## `tts_subtitles`
Enable printing of generated text while TTS is enabled.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_subtitles VALUE` or provide it as a command line parameter with
`--tts_subtitles`
Its default value is True
```

## `tts_tortoise_quality`
Quality preset. tortoise-tts only. Can be 'ultra_fast', 'fast' (default), 'standard', or 'high_quality'

```
It is a plumbing option in the tts group. You can set it with `/set
tts_tortoise_quality VALUE` or provide it as a command line parameter
with `--tts_tortoise_quality`
Its default value is fast
```

## `tts_voice`
Voice file to use for TTS. Default is 'random', which is a special value that picks a random available voice for your chosen tts_program. The value of tts_voice will be changed at startup if random is chosen, so when you find a voice you like you can find out its name with /lsoptions and checking tts_voice.

```
It is a porcelain option in the tts group. You can set it with `/set
tts_voice VALUE` or provide it as a command line parameter with
`--tts_voice`
Its default value is random
```

## `tts_voice_dir`
Directory to check first for voice file. Note: This doesn't currently work with all TTS engines, as some don't use files for voices.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_voice_dir VALUE` or provide it as a command line parameter with
`--tts_voice_dir`
Its default value is voices
```

## `tts_volume`
Volume for TTS voice program. Is passed to tts_program as environment variable.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_volume VALUE` or provide it as a command line parameter with
`--tts_volume`
Its default value is 1.0
```

## `tts_websock`
Enable websock as the output method for TTS. This is equivalent to `--tts_output_method websock`.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_websock VALUE` or provide it as a command line parameter with
`--tts_websock`
Its default value is False
```

## `tts_websock_host`
The address to bind to for the underlying TTS program when using websock as output method. ghostbox-tts only. This option is normally overriden by --http.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_websock_host VALUE` or provide it as a command line parameter with
`--tts_websock_host`
Its default value is localhost
```

## `tts_websock_port`
The port to listen on for the underlying TTS program when using websock as output method. ghostbox-tts only. This option is normally overriden by --http.

```
It is a plumbing option in the tts group. You can set it with `/set
tts_websock_port VALUE` or provide it as a command line parameter with
`--tts_websock_port`
Its default value is 5052
```

## `tts_zonos_model`
The Zonos TTS model offers two architecural variants: A pure transformer implementation or a transformer-mamba hybrid variant. Hybrid usually gives the best results, but requires flash attention. This option has no effect on non-zonos TTS engines. Options are: hybrid, transformer

```
It is a plumbing option in the tts group. You can set it with `/set
tts_zonos_model VALUE` or provide it as a command line parameter with
`--tts_zonos_model`
Its default value is hybrid
```

## `use_tools`
Enable use of tools, i.e. model may call python functions. This will do nothing if tools.py isn't present in the char directory. If tools.py is found, this will be automatically enabled.

```
It is a porcelain option in the tools group. You can set it with `/set
use_tools VALUE` or provide it as a command line parameter with
`--use_tools`
Its default value is False
```

## `var_file`
Files that will be added to the list of variables that can be expanded. E.g. -Vmemory means {$memory} will be expanded to the contents of file memory, provided expand_user_input is set. Can be used to override values set in character folders. Instead of using this, you can also just type {[FILENAME]} to have it be automatically expanded with the contents of FILENAME, provided --dynamic_file_vars is enabled.

```
It is a porcelain option in the interface group. You can set it with
`/set var_file VALUE` or provide it as a command line parameter with
`--var_file`
Its default value is []
```

## `verbose`
Show additional output for various things.

```
It is a plumbing option in the general group. You can set it with
`/set verbose VALUE` or provide it as a command line parameter with
`--verbose`
Its default value is False
```

## `warn_audio_activation_phrase`
Warn if audio is being transcribed, but no activation phrase is found. Normally this only will warn once. Set to -1 if you want to be warned every time.

```
It is a plumbing option in the audio group. You can set it with `/set
warn_audio_activation_phrase VALUE` or provide it as a command line
parameter with `--warn_audio_activation_phrase`
Its default value is True
```

## `warn_hint`
Warn if you have a hint set.

```
It is a plumbing option in the generation group. You can set it with
`/set warn_hint VALUE` or provide it as a command line parameter with
`--warn_hint`
Its default value is True
```

## `warn_trailing_space`
Warn if the prompt that is sent to the backend ends on a space. This can cause e.g. excessive emoticon use by the model.

```
It is a plumbing option in the generation group. You can set it with
`/set warn_trailing_space VALUE` or provide it as a command line
parameter with `--warn_trailing_space`
Its default value is True
```

## `warn_unsupported_sampling_parameter`
Warn if you have set an option that is usually considered a sampling parameter, but happens to be not supported by the chose nbackend.

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set warn_unsupported_sampling_parameter VALUE` or provide it
as a command line parameter with
`--warn_unsupported_sampling_parameter`
Its default value is True
```

## `websock`
Enable sending and receiving commands on a websock server running on --websock_host and --websock_port. This is enabled automatically with --http.

```
It is a plumbing option in the interface group. You can set it with
`/set websock VALUE` or provide it as a command line parameter with
`--websock`
Its default value is False
Setting it to True will start the corresponding service.
```

## `websock_host`
The hostname that the websocket server binds to.

```
It is a plumbing option in the interface group. You can set it with
`/set websock_host VALUE` or provide it as a command line parameter
with `--websock_host`
Its default value is localhost
```

## `websock_port`
The port that the websock server will listen on. By default, this is the http port +100.

```
It is a plumbing option in the interface group. You can set it with
`/set websock_port VALUE` or provide it as a command line parameter
with `--websock_port`
Its default value is 5150
```

## `whisper_model`
Name of the model to use for transcriptions using the openai whisper model. Default is 'base.en'. For a list of model names, see https://huggingface.co/openai/whisper-large

```
It is a plumbing option in the audio group. You can set it with `/set
whisper_model VALUE` or provide it as a command line parameter with
`--whisper_model`
Its default value is base.en
```

## `xtc_probability`
Set the chance for token removal via XTC sampler.
XTC means 'exclude top choices'. This sampler, when it triggers, removes all but one tokens above a given probability threshold. Recommended for creative tasks, as language tends to become less stereotypical, but can make a model less effective at structured output or intelligence-based tasks.
See original xtc PR by its inventor https://github.com/oobabooga/text-generation-webui/pull/6335

```
It is a plumbing option in the samplingparameters group. You can set
it with `/set xtc_probability VALUE` or provide it as a command line
parameter with `--xtc_probability`
Its default value is 0.5
```

