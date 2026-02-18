from dataclasses import dataclass
from enum import Enum, StrEnum
from pydantic import BaseModel, Field, model_serializer
import copy
from pydantic.types import Json
from typing import *
from .util import *
if TYPE_CHECKING:
    from ghostbox import get_ghostbox_data    

    

class Property(BaseModel):
    description: str
    type: str


class Parameters(BaseModel):
    type: str = "object"
    properties: Dict[str, Property]
    required: List[str] = []


class Function(BaseModel):
    name: str
    description: str
    # this wants jsonschema object
    parameters: Parameters


class Tool(BaseModel):
    type: str = "function"
    function: Function


class ImageRef(BaseModel):
    """This is used internally to represent an image context. It is usually discarded after one use.
    To see how images are saved in the history, see ImageContent, ChatContentComplex etc.
    """

    # This is an actual URL, e.g. a filepath
    url: str
    # the base64 encoded binary data.
    data: bytes


class ImageURL(BaseModel):
    # confusingly, this may be an URL, or just base64 image data
    # in this format:
    # f"data:image/{ext};base64,{base64_image}"
    url: str


class VideoURL(BaseModel):
    # in keeping with image, this may be a URL (e.g. filepath) or base64 encoded data.
    # in the case of video, I suppose raw data is somewhat less likely.
    url: str
    

class ChatContentComplex(BaseModel):
    """Contentfield of a ChatMessage, at least when the content is not a mere string."""

    type: Literal["text", "image_url"]
    # FIXME: I've seen multiple versions of this with text and content so we do both. the new llama.cpp vision implementation wants text
    content: str
    text: Optional[str] = None 
    image_url: Optional[ImageURL] = None
    video_url: Optional[VideoURL] = None    

    def get_text(self) -> str:
        """Simple helper to extract text from a complex message."""
        if self.text:
            return self.text
        else:
            return self.content

ChatContent = str | List[ChatContentComplex] | Dict[str, Any]


class FunctionCall(BaseModel):
    name: str
    # this is weird but it really is str, no idea why not dict
    arguments: str


class ToolCall(BaseModel):
    type: str = "function"
    function: FunctionCall
    # FIXME: I don't quite understand id field yet
    id: str = ""


class ChatMessage(BaseModel):
    role: Literal["system", "assistant", "user", "tool"]
    content: Optional[ChatContent] = None
    tool_calls: List[ToolCall] = []
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def get_text(self) -> str:
        """Simple helper to extract the text from a possibly complex message. This may drop other parts."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, ChatContentComplex):
            return self.content.get_text()
        elif isinstance(self.content, dict):
            return str(self.content)
        return ""

    def map_content(self, map_function: Callable[[str], str]) -> 'ChatMessage':
        """Returns a copy of the chat message with map_function applied to its contents."""
        msg = copy.deepcopy(self)
        match msg.content:
            case None:
                pass
            case str() as w:
                msg.content = map_function(w)
            case list():
                for i in range(len(msg.content)):
                    msg.content[i].content = map_function(msg.content[i].content)
            case dict():
                # this actually never happens
                raise RuntimeError("Dict not supported for ChatMessage content.")
        return msg
                    
    @model_serializer
    def ser_model(self) -> Dict[str, Any]:
        # we basically want exclude_none=True by default
        return {k:v for k, v in dict(self).items() if v is not None and v != []}
    
    @staticmethod
    def make_image_message(text: str, images: List[ImageRef], **kwargs: object) -> 'ChatMessage':
        from ghostbox.util import getImageExtension

        complex_content_list = []
        for image_ref in images:
            ext = getImageExtension(image_ref.url, default="png")
            base64_image = image_ref.data.decode("utf-8")
            image_content = ChatContentComplex(
                type="image_url",
                content="",
                image_url=ImageURL(
                    url=(
                        f"data:image/{ext};base64,{base64_image}"
                        if image_ref.data is not None
                        else image_ref.url
                    )
                ),
            )
            complex_content_list.append(image_content)

        # don't forget the prompt
        complex_content_list.append(ChatContentComplex(type="text", content=text, text=text))

        # FIXME: not sure why mypy complains about kwargs here
        return ChatMessage(role="user", content=complex_content_list, **kwargs) # type: ignore


class LLMBackend(StrEnum):
    generic = "generic"
    legacy = "legacy"
    llamacpp = "llamacpp"
    openai = "openai"
    google = "google"
    deepseek = "deepseek"
    qwen = "qwen"
    iflow = "iflow"
    dummy = "dummy"

# these are the models supported by ghostbox-tts
class TTSModel(StrEnum):
    zonos = "zonos"
    kokoro = "kokoro"
    xtts = "xtts"
    polly = "polly"
    orpheus = "orpheus"

class ZonosTTSModel(StrEnum):
    hybrid = "hybrid"
    transformer = "transformer"

# these are ways of playing sound that are supported by ghostbox-tts
class TTSOutputMethod(StrEnum):
    default = "default"
    websock = "websock"


class PromptFormatTemplateSpecialValue(StrEnum):
    auto = "auto"
    guess = "guess"
    raw = "raw"

class ArgumentType(StrEnum):
    Porcelain = "Porcelain"
    Plumbing = "Plumbing"
    
class ArgumentGroup(StrEnum):
    General = "General"
    Generation = "Generation"
    Interface = "Interface"
    Characters = "Characters"
    Templates = "Templates"
    TTS = "TTS"
    Audio = "Audio"
    Images = "Images"
    Tools = "Tools"
    Backend = "Backend"
    SamplingParameters = "SamplingParameters"
    LlamaCPP = "LlammaCPP"
    OpenAI = "OpenAI"
    Google = "Google"

class ArgumentTag(BaseModel):
    """Metadata associated with a command line argument."""

    name: str = ""
    type: ArgumentType
    group: ArgumentGroup

    # this is for e.g. streaming or temperature.
    very_important: bool = False

    # wether changing the value of this argument may start a service
    service: bool = False

    # for inclusion in the message of the day/tip
    motd: bool = False

    # basically, if its a command or option
    is_option: bool = True
    default_value: Optional[Any] = None

    # same help that is printed in terminal on --help
    help: str = ""

    def show_type(self) -> str:
        if self.is_option:
            return (
                "It is a "
                + self.type.name.lower()
                + " option in the "
                + self.group.name.lower()
                + " group."
            )
        return (
            "It is a "
            + self.type.name.lower()
            + " command in the "
            + self.group.name.lower()
            + " group."
        )

    def show_description(self) -> str:
        w = ""
        w += self.show_type()
        w += (
            "\nYou can set it with `/set "
            + self.name
            + " VALUE` or provide it as a command line parameter with `--"
            + self.name
            + "`"
        )
        return w


class SamplingParameterSpec(BaseModel):
    """Sampling parameters can be provided to backends to influence a model's inference behaviour.
    Most commonly this is temperature, presence penalty etc. However, here we take sampling parameter in the broadest sense, including samplers, CFG and control vectors.
    Note that this class provides only the specification of a sampling_parameter. This is for documentation and to keep track of which backend supports which parameters.
    """

    name: str
    description: str
    default_value: Any


class Timings(BaseModel):
    """Performance statistics for LLM backends.
    Most backends give timing statistics, though the format and particular stats vary. This class unifies the interface and boils it down to only the stats we care about.
    """

    prompt_n: int
    predicted_n: int
    cached_n: Optional[int] = None
    truncated: bool
    prompt_ms: float
    predicted_ms: float
    predicted_per_second: float
    predicted_per_token_ms: float

    original_timings: Optional[Dict[str, Any]] = {}

    def total_n(self) -> int:
        return self.prompt_n + self.predicted_n

    def total_ms(self) -> float:
        return self.prompt_ms + self.predicted_ms


api_default_options = {
    "color": False,
    "verbose": False,
    "stderr": True,
    "log_time": True,
    "cli_prompt": "",
    "dynamic_file_vars": False,
    "max_context_length": 2**15,
    "__api__": True # hidden option indicating that ghsotbox was started from Ghostbox class
}

class BrokenBackend(Exception):
    pass


class ModelStats(BaseModel):
    """Model information, like name and so on (example: 'gemini-2.5-flash'). Usually used with cloud providers."""

    name: str = Field(
        description = "The exact name of the model."
    )

    display_name: str = Field(
        default = "",
        description = "A more user friendly rendition of the model name."
    )

    description: str = Field(
        default = "",
        description = "Short description of the model's capabilities."
    )
        

class Config(BaseModel):
    # General
    include: List[str] = Field(
        default_factory=lambda: [userCharDir(), get_ghostbox_data("chars/"), "chars/"],
        description="Include paths that will be searched for character folders named with the /start command or the --character_folder command line argument.",
        json_schema_extra={
            "argparse": {
                "short": "-I",
                "long": "--include",
                "action": "append",
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Characters, motd=True)
            }
        }
    )
    template_include: List[str] = Field(
        default_factory=lambda: [userTemplateDir(), get_ghostbox_data("templates/"), "templates/"],
        description="Include paths that will be searched for prompt templates. You can specify a template to use with the -T option.",
        json_schema_extra={
            "argparse": {
                "long": "--template_include",
                "action": "append",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Templates)
            }
        }
    )
    history: bool = Field(
        default=True,
        description="If false, do not append messages to chat history. This is used mostly in the API to send system messages that won't clutter up the user's chat history.",
        json_schema_extra={
            "argparse": {
                "long": "--history",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.General)
            }
        }
    )
    history_retroactive_vars: bool = Field(
        default=False,
        description="If true, vars that you can set with e.g. Ghostbox.set_vars will be kept in the history in their raw magic string form, and replaced only before being sent to the backend. This allows you to retroactively change the content of past prompts.",
        json_schema_extra={
            "argparse": {
                "long": "--history_retroactive_vars",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.General)
            }
        }
    )
    history_force_alternating_roles: bool = Field(
        default=True,
        description="Force chat history to keep alternating roles between 'assistant' and 'user'. This can be important as many backends require this server-side, throwing a server error if it is not observed. Enabling this will enforce the alternation through a variety of strategies, some of which may alter your messages.",
        json_schema_extra={
            "argparse": {
                "long": "--history_force_alternating_roles",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.General)
            }
        }
    )
    prompt_format: str = Field(
        default="auto",
        description="Prompt format template to use. The default is 'auto', which means ghostbox will .let the backend handle templating, which is usually the right choice. You can still use other settings, like 'raw', to experiment. This is ignored if you use the generic or openai backend. Note: Prompt format templates used to be more important in the early days of LLMs, as confusion was rampant and mistakes were not uncommon even in official releases. Nowadays, it is quite safe to use the official templates. You may still want to use this option for experimentation, however.",
        json_schema_extra={
            "argparse": {
                "short": "-T",
                "long": "--prompt_format",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Templates)
            }
        }
    )
    stop: List[str] = Field(
        default_factory=list,
        description="Forbidden strings that will stop the LLM backend generation.",
        json_schema_extra={
            "argparse": {
                "short": "-s",
                "long": "--stop",
                "action": "append",
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Generation, motd=True)
            }
        }
    )
    character_folder: str = Field(
        default="",
        description="character folder to load at startup. The folder may contain a `system_msg` file, a `config.json`, and a `tools.py`, as well as various other files used as file variables. See the examples and documentation for more.",
        json_schema_extra={
            "argparse": {
                "short": "-c",
                "long": "--character_folder",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Characters, very_important=True, motd=True)
            }
        }
    )
    prompt: Optional[str] = Field(
        default=None,
        description="If provided, process the prompt and exit.",
        json_schema_extra={
            "argparse": {
                "short": "-p",
                "long": "--prompt",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Generation)
            }
        }
    )
    endpoint: str = Field(
        default="http://localhost:8080",
        description="Address of backend http endpoint. This is a URL that is dependent on the backend you use, though the default of localhost:8080 works for most, including Llama.cpp and Kobold.cpp. If you want to connect to an online provider that is not part of the explicitly supported backends, this is where you would supply their API address.",
        json_schema_extra={
            "argparse": {
                "long": "--endpoint",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Backend)
            }
        }
    )
    client: bool = Field(
        default=False,
        description="Run ghostbox in client mode, connecting to the remote address specified with --remote_host and --remote_port. The remote host must run a ghostbox instance started with --http.",
        json_schema_extra={
            "argparse": {
                "long": "--client",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.General)
            }
        }
    )
    remote_host: str = Field(
        default="localhost",
        description="Remote address to connect to with --client.",
        json_schema_extra={
            "argparse": {
                "long": "--remote_host",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.General)
            }
        }
    )
    remote_port: int = Field(
        default=5150,
        description="Remote port to connect to in client mode.",
        json_schema_extra={
            "argparse": {
                "long": "--remote_port",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.General)
            }
        }
    )
    backend: LLMBackend = Field(
        default=LLMBackend.generic,
        description="Backend to use. The default is `generic`, which conforms to the OpenAI REST API, and is supported by most LLM providers. Choosing a more specific backend may provide additional functionality. Other possible values are " + ", ".join([e.value for e in LLMBackend]) + ".",
        json_schema_extra={
            "argparse": {
                "long": "--backend",
                "type": str,
                "choices": [e.value for e in LLMBackend],
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Backend, very_important=True)
            }
        }
    )
    api_key: str = Field(
        default="",
        description="API key for various services. (e.g. OpenAI, Google's AI Studio). Be sure to specify the right --backend option. If using a specific backend and a specific api key is set, the specific key will be used over the general one.",
        json_schema_extra={
            "argparse": {
                "long": "--api_key",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.OpenAI)
            }
        }
    )
    google_api_key: str = Field(
        default="",
        description="API key for google's AI Studio. https://aistudio.google.com. If this value is set, it will override the --api_key when using google's api.",
        json_schema_extra={
            "argparse": {
                "long": "--google_api_key",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Google)
            }
        }
    )
    deepseek_api_key: str = Field(
        default="",
        description="API key for Deepseek https://deepseek.com",
        json_schema_extra={
            "argparse": {
                "long": "--deepseek_api_key",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Backend)
            }
        }
    )
    qwen_api_key: str = Field(
        default="",
        description="API key for Qwen (https://qwen.ai). You can also set this via the DASHSCOPE_API_KEY environment variable.",
        json_schema_extra={
            "argparse": {
                "long": "--qwen_api_key",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Backend)
            }
        }
    )
    iflow_api_key: str = Field(
        default="",
        description="API key for the iFlow cloud provider (https://iflow.cn). You can also set the IFLOW_API_KEY environment variable.",
        json_schema_extra={
            "argparse": {
                "long": "--iflow_api_key",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Backend)
            }
        }
    )
    iflow_prefered_model: str = Field(
        default="qwen3-coder-plus",
        description="Prefered model to use with the Iflow backend (https://iflow.cn).",
        json_schema_extra={
            "argparse": {
                "long": "--iflow_prefered_model",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Backend)
            }
        }
    )
    google_prefered_model: str = Field(
        default="models/gemini-2.5-flash",
        description="Prefered model to use with google backend. This will only be used if --model is not set.",
        json_schema_extra={
            "argparse": {
                "long": "--google_prefered_model",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Google)
            }
        }
    )
    qwen_prefered_model: str = Field(
        default="qwen3-vl-plus",
        description="Prefered model to use with qwen backend. This will only be used if --model is not set.",
        json_schema_extra={
            "argparse": {
                "long": "--qwen_prefered_model",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Backend)
            }
        }
    )
    max_length: int = Field(
        default=300,
        description="Number of tokens to request from backend for generation. Generation is stopped when this number is exceeded. Negative values mean generation is unlimited and will terminate when the backend generates a stop token.",
        json_schema_extra={
            "argparse": {
                "long": "--max_length",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Generation, very_important=True)
            }
        }
    )
    max_context_length: int = Field(
        default=32768,
        description="Maximum number of tokens to keep in context.",
        json_schema_extra={
            "argparse": {
                "long": "--max_context_length",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Generation, very_important=True)
            }
        }
    )
    chat_user: str = Field(
        default="user",
        description="Username you wish to be called when chatting in 'chat' mode. It will also replace occurrences of {chat_user} anywhere in the character files. If you don't provide one here, your username will be determined by your OS session login.",
        json_schema_extra={
            "argparse": {
                "short": "-u",
                "long": "--chat_user",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.General, very_important=True)
            }
        }
    )
    mode: str = Field(
        default="default",
        description="Mode of operation. Changes various things behind-the-scenes. Values are currently 'default', or 'chat'.",
        json_schema_extra={
            "argparse": {
                "short": "-M",
                "long": "--mode",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Templates)
            }
        }
    )
    force_params: bool = Field(
        default=False,
        description="Force sending of sample parameters, even when they are seemingly not supported by the backend (use to debug or with generic",
        json_schema_extra={
            "argparse": {
                "long": "--force_params",
                "boolean_optional_action": True,
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Backend)
            }
        }
    )
    llamacpp_thinking_json_fix: bool = Field(
        default=True,
        description="Llama.cpp does (currently, october 2025) not support structured output (e.g. via a json schema) with simultaneous reasoning. This can degrade the output quality. If this is enabled, ghostbox attempts a workaround to retrieve correct json while still letting the model go through it's reasoning. If this option is disabled, the json schema uses the default method of having llama.cpp generate a grammar, which forces the model to adhere to the schema exactly. Note that this is only relevant if enable_thinking is true",
        json_schema_extra={
            "argparse": {
                "long": "--llamacpp_thinking_json_fix",
                "boolean_optional_action": True,
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Backend)
            }
        }
    )
    llamacpp_auto_enable_thinking: bool = Field(
        default=True,
        description="If true, will query the llama.cpp server to determine if a thinking model is in use, and attempt to set the enable_thinking option accordingly. Ignored for other backends.",
        json_schema_extra={
            "argparse": {
                "long": "--llamacpp_auto_enable_thinking",
                "boolean_optional_action": True,
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Backend)
            }
        }
    )
    model: str = Field(
        default="",
        description="LLM to use for requests. This only works if the backend supports choosing models.",
        json_schema_extra={
            "argparse": {
                "short": "-m",
                "long": "--model",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Backend, very_important=True)
            }
        }
    )
    grammar_file: str = Field(
        default="",
        description="Grammar file used to restrict generation output. Grammar format is GBNF.",
        json_schema_extra={
            "argparse": {
                "short": "-g",
                "long": "--grammar_file",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Generation, motd=True)
            }
        }
    )
    chat_ai: str = Field(
        default="",
        description="Name the AI will have when chatting. Has various effects on the prompt when chat mode is enabled. This is usually set automatically in the config.json file of a character folder.",
        json_schema_extra={
            "argparse": {
                "long": "--chat_ai",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Characters)
            }
        }
    )
    stream: bool = Field(
        default=True,
        description="Enable streaming mode. This will print generations by the LLM piecemeal, instead of waiting for a full generation to complete. Results may be printed per-token, per-sentence, or otherwise, according to --stream_flush.",
        json_schema_extra={
            "argparse": {
                "long": "--stream",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Generation, very_important=True)
            }
        }
    )
    http: bool = Field(
        default=False,
        description="Enable a small webserver with minimal UI. By default, you'll find it at localhost:5050.",
        json_schema_extra={
            "argparse": {
                "long": "--http",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Interface, service=True)
            }
        }
    )
    websock: bool = Field(
        default=False,
        description="Enable sending and receiving commands on a websock server running on --websock_host and --websock_port. This is enabled automatically with --http.",
        json_schema_extra={
            "argparse": {
                "long": "--websock",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface, service=True, motd=True)
            }
        }
    )
    websock_host: str = Field(
        default="localhost",
        description="The hostname that the websocket server binds to.",
        json_schema_extra={
            "argparse": {
                "long": "--websock_host",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    websock_port: int = Field(
        default=5150,
        description="The port that the websock server will listen on. By default, this is the http port +100.",
        json_schema_extra={
            "argparse": {
                "long": "--websock_port",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    http_host: str = Field(
        default="localhost",
        description="Hostname to bind to if --http is enabled.",
        json_schema_extra={
            "argparse": {
                "long": "--http_host",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    http_port: int = Field(
        default=5050,
        description="Port for the web server to listen on if --http is provided. By default, the --audio_websock_port will be --http_port+1, and --tts_websock_port will be --http_port+2, e.g. 5051 and 5052.",
        json_schema_extra={
            "argparse": {
                "long": "--http_port",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    http_override: bool = Field(
        default=True,
        description="If enabled, the values of --audio_websock, --tts_websock, --audio_websock_host, --audio_websock_port, --tts_websock_host, --tts_websock_port will be overriden if --http is provided. Use --no-http_override to disable this, so you can set your own host/port values for the websock services or disable them entirely.",
        json_schema_extra={
            "argparse": {
                "long": "--http_override",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    multiline: bool = Field(
        default=False,
        description="Makes multiline mode the dfault, meaning that newlines no longer trigger a message being sent to the backend. instead, you must enter the value of --multiline_delimiter to trigger a send.",
        json_schema_extra={
            "argparse": {
                "long": "--multiline",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Interface)
            }
        }
    )
    multiline_delimiter: str = Field(
        default="\\",
        description="String that signifies the end of user input. This is only relevant for when --multiline is enabled. By default this is a backslash, inverting the normal behaviour of backslashes allowing to enter a newline ad-hoc while in multiline mode. This option is intended to be used by scripts to change the delimiter to something less common.",
        json_schema_extra={
            "argparse": {
                "long": "--multiline_delimiter",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    color: bool = Field(
        default=True,
        description="Enable colored output.",
        json_schema_extra={
            "argparse": {
                "long": "--color",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Interface, motd=True)
            }
        }
    )
    text_ai_color: str = Field(
        default="none",
        description="Color for the generated text, as long as --color is enabled. Most ANSI terminal colors are supported.",
        json_schema_extra={
            "argparse": {
                "long": "--text_ai_color",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface, motd=True)
            }
        }
    )
    text_ai_style: str = Field(
        default="bright",
        description="Style for the generated text, as long as --color is enabled. Most ANSI terminal styles are supported.",
        json_schema_extra={
            "argparse": {
                "long": "--text_ai_style",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    dynamic_file_vars: bool = Field(
        default=True,
        description="Dynamic file vars are strings of the form {[FILE1]}. If FILE1 is found, the entire expression is replaced with the contents of FILE1. This is dynamic in the sense that the contents of FILE1 are loaded each time the replacement is encountered, which is different from the normal file vars with {{FILENAME}}, which are loaded once during character initialization. Replacement happens in user inputs only. In particular, dynamic file vars are ignored in system messages or saved chats. If you want the LLM to get file contents, use tools. disabling this means no replacement happens. This can be a security vulnerability, so it is disabled by default on the API.",
        json_schema_extra={
            "argparse": {
                "long": "--dynamic_file_vars",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Templates, motd=True)
            }
        }
    )
    dynamic_file_vars_unsafe: bool = Field(
        default=False,
        description="If true, will recursively expand dynamic file variables. This allows you to have nested includes of files. Disabled by default because this may lead to accidental prompt injections and general confusion.",
        json_schema_extra={
            "argparse": {
                "long": "--dynamic_file_vars_unsafe",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Templates, motd=True)
            }
        }
    )
    dynamic_file_vars_max_depth: int = Field(
        default=5,
        description="Maximum recursion depths for the expansion of nested dynamic file variables. If dynamic_file_vars_unsafe is false, this has no effect. Setting this to 0 or lower can lead to infinite recursion.",
        json_schema_extra={
            "argparse": {
                "long": "--dynamic_file_vars_max_depth",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Templates, motd=True)
            }
        }
    )
    warn_trailing_space: bool = Field(
        default=True,
        description="Warn if the prompt that is sent to the backend ends on a space. This can cause e.g. excessive emoticon use by the model.",
        json_schema_extra={
            "argparse": {
                "long": "--warn_trailing_space",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Generation)
            }
        }
    )
    warn_unsupported_sampling_parameter: bool = Field(
        default=True,
        description="Warn if you have set an option that is usually considered a sampling parameter, but happens to be not supported by the chose nbackend.",
        json_schema_extra={
            "argparse": {
                "long": "--warn_unsupported_sampling_parameter",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters)
            }
        }
    )
    warn_audio_activation_phrase: bool = Field(
        default=True,
        description="Warn if audio is being transcribed, but no activation phrase is found. Normally this only will warn once. Set to -1 if you want to be warned every time.",
        json_schema_extra={
            "argparse": {
                "long": "--warn_audio_activation_phrase",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    warn_hint: bool = Field(
        default=True,
        description="Warn if you have a hint set.",
        json_schema_extra={
            "argparse": {
                "long": "--warn_hint",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Generation)
            }
        }
    )
    json_grammar: bool = Field(
        default=False,
        description="Force generation output to be in JSON format. This is equivalent to using -g with a json.gbnf grammar file, but this option is provided for convenience.",
        json_schema_extra={
            "argparse": {
                "long": "--json_grammar",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Generation, motd=True)
            }
        }
    )
    response_format: Union[str, Dict[str, Any]] = Field(
        default="text",
        description="Used internally to request JSON schemas as response format (see OpenAI API). Setting it to 'text' will yield default text generation.",
        json_schema_extra={
            "argparse": {
                "long": "--response_format",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Generation, motd=True)
            }
        }
    )
    stream_flush: str = Field(
        default="token",
        description="When to flush the streaming buffer. When set to 'token', will print each token immediately. When set to 'sentence', it will wait for a complete sentence before printing. When set to 'flex', will act like 'sentence', but prebuffer a minimum amount of characters before flushing, according to stream_flush_flex_value. Default is 'token'.",
        json_schema_extra={
            "argparse": {
                "long": "--stream_flush",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Generation, motd=True)
            }
        }
    )
    stream_flush_flex_value: int = Field(
        default=50,
        description="How many characters (not tokens) at least to buffer before flushing the queue, when stream_flush is set to 'flex'.",
        json_schema_extra={
            "argparse": {
                "long": "--stream_flush_flex_value",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Generation, motd=False)
            }
        }
    )
    cli_prompt: str = Field(
        default=" 0 > ",
        description="String to show at the bottom as command prompt. Can be empty.",
        json_schema_extra={
            "argparse": {
                "long": "--cli_prompt",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface, motd=True)
            }
        }
    )
    cli_prompt_color: str = Field(
        default="none",
        description="Color of the prompt. Uses names of standard ANSI terminal colors. Requires --color to be enabled.",
        json_schema_extra={
            "argparse": {
                "long": "--cli_prompt_color",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    hint: str = Field(
        default="",
        description="Hint for the AI. This string will be appended to the prompt behind the scenes. It's the first thing the AI sees. Try setting it to 'Of course,' to get a more compliant AI. Also refered to as 'prefill'.",
        json_schema_extra={
            "argparse": {
                "long": "--hint",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Generation)
            }
        }
    )
    hint_sticky: bool = Field(
        default=True,
        description="If disabled, hint will be shown to the AI as part of prompt, but will be omitted from the story.",
        json_schema_extra={
            "argparse": {
                "long": "--hint_sticky",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Generation)
            }
        }
    )
    tts: bool = Field(
        default=False,
        description="Enable text to speech on generated text.",
        json_schema_extra={
            "argparse": {
                "long": "--tts",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.TTS, very_important=True, service=True)
            }
        }
    )
    tts_model: TTSModel = Field(
        default=TTSModel.kokoro,
        description="The TTS model to use. This is ignored unless you use ghostbox-tts as your tts_program. Options are:  " + ", ".join([e.value for e in TTSModel]) + ".",
        json_schema_extra={
            "argparse": {
                "long": "--tts_model",
                "type": str,
                "choices": [e.value for e in TTSModel],
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_zonos_model: ZonosTTSModel = Field(
        default=ZonosTTSModel.hybrid,
        description="The Zonos TTS model offers two architecural variants: A pure transformer implementation or a transformer-mamba hybrid variant. Hybrid usually gives the best results, but requires flash attention. This option has no effect on non-zonos TTS engines. Options are: " + ", ".join([e.value for e in ZonosTTSModel]) + ".",
        json_schema_extra={
            "argparse": {
                "long": "--tts_zonos_model",
                "type": str,
                "choices": [e.value for e in ZonosTTSModel],
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_orpheus_model: str = Field(
        default="",
        description="Filepath to gguf file or huggingface repo name of the orpheus model you wish to use with the tts. Leave empty for some reasonable defaults. If you don't use the default, and set this to a repo, the underlying ghostbox-tts might silently download the snapshot, so check with /ttsdebug. This option is ignored unless you actually set tts_model to orpheus.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_orpheus_model",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_llm_server: str = Field(
        default="",
        description="Address of another LLM server that serves a TTS model on an OpenAI compatible backend. This is only relevant if you use a tts model that needs an LLM as backend. This is currently the case with orpheus. You can also let ghostbox-tts spawn a server on its own if you have llama-server in your path, by setting tts_orpheus_model to an orpheus gguf file or huggingface repository.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_llm_server",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_language: str = Field(
        default="en",
        description="Set the TTS voice language. Right now, this is only relevant for kokoro.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_language",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_output_method: TTSOutputMethod = Field(
        default=TTSOutputMethod.default,
        description="How to play the generated speech. Using the --http argument automatically sets this to websock.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_output_method",
                "type": str,
                "choices": [e.value for e in TTSOutputMethod],
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_filter: List[str] = Field(
        default_factory=list,
        description="List of strings to filter out before passing the LLM generation to the underlying tts. Strings in the list will remain in the generated text, but will not be read out by the TTS engine. You can use this for responsive AI assistrant's that can output certain signal words e.g. '<USER_REQUEST>'.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_filter",
                "nargs": "+",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_websock: bool = Field(
        default=False,
        description="Enable websock as the output method for TTS. This is equivalent to `--tts_output_method websock`.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_websock",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_websock_host: str = Field(
        default="localhost",
        description="The address to bind to for the underlying TTS program when using websock as output method. ghostbox-tts only. This option is normally overriden by --http.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_websock_host",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_websock_port: int = Field(
        default=5052,
        description="The port to listen on for the underlying TTS program when using websock as output method. ghostbox-tts only. This option is normally overriden by --http.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_websock_port",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_interrupt: bool = Field(
        default=True,
        description="Stop an ongoing TTS whenever a new generation is spoken. When set to false, will queue messages instead.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_interrupt",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_program: str = Field(
        default="ghostbox-tts",
        description="Path to a TTS (Text-to-speech) program to verbalize generated text. The TTS program should read lines from standard input. Many examples are provided in scripts/ghostbox-tts-* . The ghostbox-tts script offers a native solution using various supported models.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_program",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_clone_dir: str = Field(
        default="",
        description="Directory to check first for wave files used in voice cloning. Note that voice cloning isn't supported by all tts models.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_clone_dir",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_tortoise_quality: str = Field(
        default="fast",
        description="Quality preset. tortoise-tts only. Can be 'ultra_fast', 'fast' (default), 'standard', or 'high_quality'",
        json_schema_extra={
            "argparse": {
                "long": "--tts_tortoise_quality",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_volume: float = Field(
        default=1.0,
        description="Volume for TTS voice program. Is passed to tts_program as environment variable.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_volume",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_modify_system_msg: bool = Field(
        default=True,
        description="If enabled, instructions specific to the underlying TTS model will be appended to the system prompt. This may e.g. point out usage of tags like <laugh> and <cough> to the LLM. Support varies depending on model.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_modify_system_msg",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_rate: int = Field(
        default=50,
        description="Speaking rate for TTS voice program. Is passed to tts_program as environment variable. Note that speaking rate is not supported by all TTS engines.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_rate",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_additional_arguments: str = Field(
        default="",
        description="Additional command line arguments that will be passed to the tts_program.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_additional_arguments",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    image_watch: bool = Field(
        default=False,
        description="Enable watching of a directory for new images. If a new image appears in the folder, the image will be loaded with id 0 and sent to the backend. works with multimodal models only (like llava).",
        json_schema_extra={
            "argparse": {
                "long": "--image_watch",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Images, service=True, motd=True)
            }
        }
    )
    image_watch_clear_history: bool = Field(
        default=True,
        description="Clear the current story/message hisotry when a new image is automatically detected.",
        json_schema_extra={
            "argparse": {
                "long": "--image_watch_clear_history",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Images, service=True, motd=True)
            }
        }
    )
    image_watch_dir: str = Field(
        default=os.path.expanduser("~/Pictures/Screenshots/"),
        description="Directory that will be watched for new image files when --image_watch is enabled.",
        json_schema_extra={
            "argparse": {
                "long": "--image_watch_dir",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Images)
            }
        }
    )
    image_watch_msg: str = Field(
        default="Can you describe this image?",
        description="If image_watch is enabled, this message will be automatically send to the backend whenever a new image is detected. Set this to '' to disable automatic messages, while still keeping the automatic update the image with id 0.",
        json_schema_extra={
            "argparse": {
                "long": "--image_watch_msg",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Images, motd=True)
            }
        }
    )
    image_watch_hint: str = Field(
        default="",
        description="If image_watch is enabled, this string will be sent to the backend as start of the AI response whenever a new image is detected and automatically described. This allows you to guide or solicit the AI by setting it to e.g. 'Of course, this image show' or similar. Default is ''",
        json_schema_extra={
            "argparse": {
                "long": "--image_watch_hint",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Images, motd=True)
            }
        }
    )
    whisper_model: str = Field(
        default="base.en",
        description="Name of the model to use for transcriptions using the openai whisper model. Default is 'base.en'. For a list of model names, see https://huggingface.co/openai/whisper-large",
        json_schema_extra={
            "argparse": {
                "long": "--whisper_model",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    tts_voice: str = Field(
        default="random",
        description="Voice to use for TTS. Default is 'random', which is a special value that picks a random available voice for your chosen tts_program. The value of tts_voice will be changed at startup if random is chosen, so when you find a voice you like you can find out its name with /lsoptions and checking tts_voice. To get a list of voices, start ghostbox with your desired tts model and do /lsvoices.",
        json_schema_extra={
            "argparse": {
                "short": "-y",
                "long": "--tts_voice",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.TTS)
            }
        }
    )
    tts_subtitles: bool = Field(
        default=True,
        description="Enable printing of generated text while TTS is enabled.",
        json_schema_extra={
            "argparse": {
                "long": "--tts_subtitles",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.TTS)
            }
        }
    )
    config_file: str = Field(
        default="",
        description="Path to a config fail in JSON format, containing a dictionary with OPTION : VALUE pairs to be loaded on startup. Same as /loadconfig or /loadoptions. To produce an example config-file, try /saveconfig example.json.",
        json_schema_extra={
            "argparse": {
                "long": "--config_file",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.General)
            }
        }
    )
    chat_show_ai_prompt: bool = Field(
        default=True,
        description="Controls wether to show AI prompt in chat mode. Specifically, assuming chat_ai = 'Bob', setting chat_show_ai_prompt to True will show 'Bob: ' in front of the AI's responses. Note that this is always sent to the back-end (in chat mode), this parameter merely controls wether it is shown.",
        json_schema_extra={
            "argparse": {
                "long": "--chat_show_ai_prompt",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    smart_context: bool = Field(
        default=True,
        description="Enables ghostbox version of smart context, which means dropping text at user message boundaries when the backend's context is exceeded. If you disable this, it usually means the backend will truncate the raw message. Enabling smart context means better responses and longer processing time due to cache invalidation, disabling it means worse responses with faster processing time. Note from marius: Beware I haven't looked at this in a while, since newer models all have very large contexts.",
        json_schema_extra={
            "argparse": {
                "long": "--smart_context",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Generation, motd=True)
            }
        }
    )
    hide: bool = Field(
        default=False,
        description="Hides some unnecessary output, providing a more immersive experience. Same as typing /hide.",
        json_schema_extra={
            "argparse": {
                "long": "--hide",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface, motd=True)
            }
        }
    )
    sound_output_device_index: Optional[int] = Field(
        default=None,
        description="Index of sound output device for playback using pyaudio. If left unspecified, the default output device will be automatically determined. Use --sound_list_output_devices to see the device indices you can use here.",
        json_schema_extra={
            "argparse": {
                "long": "--sound_output_device_index",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.General)
            }
        }
    )
    sound_input_device_index: Optional[int] = Field(
        default=None,
        description="Index of sound input device for recording using pyaudio. If left unspecified, the default output device will be automatically determined. Use --sound_list_output_devices to see the device indices you can use here.",
        json_schema_extra={
            "argparse": {
                "long": "--sound_input_device_index",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.General)
            }
        }
    )
    sound_list_output_devices: bool = Field(
        default=False,
        description="List output devices and their indices. You can use the numbers as arguments to --sound_output_device_index.",
        json_schema_extra={
            "argparse": {
                "long": "--sound_list_output_devices",
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.General)
            }
        }
    )
    audio: bool = Field(
        default=False,
        description="Enable automatic transcription of audio input using openai whisper model. Obviously, you need a mic for this.",
        json_schema_extra={
            "argparse": {
                "long": "--audio",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Audio, service=True, very_important=True)
            }
        }
    )
    audio_silence_threshold: int = Field(
        default=2000,
        description="An integer value denoting the threshold for when automatic audio transcription starts recording. (default 2000)",
        json_schema_extra={
            "argparse": {
                "long": "--audio_silence_threshold",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio, motd=True)
            }
        }
    )
    audio_activation_phrase: str = Field(
        default="",
        description="When set, the phrase must be detected in the beginning of recorded audio, or the recording will be ignored. Phrase matching is fuzzy with punctuation removed.",
        json_schema_extra={
            "argparse": {
                "long": "--audio_activation_phrase",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio, motd=True)
            }
        }
    )
    audio_activation_period_ms: int = Field(
        default=0,
        description="Period in milliseconds where no further activation phrase is necessary to trigger a response. The period starts after any interaction with the AI, spoken or otherwise.",
        json_schema_extra={
            "argparse": {
                "long": "--audio_activation_period_ms",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    audio_interrupt: bool = Field(
        default=True,
        description="Stops generation and TTS when you start speaking. Does not require activation phrase.",
        json_schema_extra={
            "argparse": {
                "long": "--audio_interrupt",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    audio_activation_phrase_keep: bool = Field(
        default=True,
        description="If false and an activation phrase is set, the triggering phrase will be removed in messages that are sent to the backend.",
        json_schema_extra={
            "argparse": {
                "long": "--audio_activation_phrase_keep",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    audio_show_transcript: bool = Field(
        default=True,
        description="Show transcript of recorded user speech when kaudio transcribing is enabled. When disabled, you can still see the full transcript with /log or /print.",
        json_schema_extra={
            "argparse": {
                "long": "--audio_show_transcript",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    audio_websock: bool = Field(
        default=False,
        description="Enable to listen for audio on an HTTP websocket at the given `--websock_url`, instead of recording audio from a microphone. This can be used to stream audio through a website. This is enabled by default with the --http option unless you also supply --no-http_overrid.",
        json_schema_extra={
            "argparse": {
                "long": "--audio_websock",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    audio_websock_host: str = Field(
        default="localhost",
        description="The address to bind to when `--audio_websock` is enabled. You can stream audio to this endpoint using the websocket protocol for audio transcription. Normally overriden by --http.",
        json_schema_extra={
            "argparse": {
                "long": "--audio_websock_host",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    audio_websock_port: int = Field(
        default=5051,
        description="The port to listen on when `--audio_websock` is enabled. You can stream audio to this endpoint using the websocket protocol for audio transcription. Normally overriden by --http.",
        json_schema_extra={
            "argparse": {
                "long": "--audio_websock_port",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Audio)
            }
        }
    )
    verbose: bool = Field(
        default=False,
        description="Show additional output for various things.",
        json_schema_extra={
            "argparse": {
                "long": "--verbose",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.General)
            }
        }
    )
    log_time: bool = Field(
        default=False,
        description="Print timing and performance statistics to stderr with every generation. Auto enabled for the API.",
        json_schema_extra={
            "argparse": {
                "long": "--log_time",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.General)
            }
        }
    )
    quiet: bool = Field(
        default=False,
        description="Prevents printing and TTS vocalization of generations. Often used with the API when you want to handle generation results yourself and don't want printing to console.",
        json_schema_extra={
            "argparse": {
                "short": "-q",
                "long": "--quiet",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Interface)
            }
        }
    )
    history_drop_on_generation_error: bool = Field(
        default=True,
        description="Drop the last user message in history when encountering a generation error (no message from the AI). This means you have to resend your prompt manually, but it keeps the chat history clean.",
        json_schema_extra={
            "argparse": {
                "long": "--history_drop_on_generation_error",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Interface)
            }
        }
    )
    stderr: bool = Field(
        default=True,
        description="Wether printing to stderr is enabled. You may want to disable this when building terminal applications using the API.",
        json_schema_extra={
            "argparse": {
                "long": "--stderr",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    stdout: bool = Field(
        default=True,
        description="Wether printing to stdout is enabled. You may want to disable this when building terminal applications using the API.",
        json_schema_extra={
            "argparse": {
                "long": "--stdout",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    expand_user_input: bool = Field(
        default=True,
        description="Expand variables in user input. E.g. {$var} will be replaced with content of var. Variables are initialized from character folders (i.e. file 'memory' will be {$memory}), or can be set manually with the /varfile command or --varfile option. See also --dynamic_file_vars.",
        json_schema_extra={
            "argparse": {
                "long": "--expand_user_input",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Interface)
            }
        }
    )
    tools_unprotected_shell_access: bool = Field(
        default=False,
        description="Allow an AI to run shell commands, even if not logged in to their own account. The safe way of doing this is to create an account on your system with the same name as the AI, and then run this program under their account. If you don't want to do that, and you are ok with an AI deleting your files through accident or malice, set this flag to true.",
        json_schema_extra={
            "argparse": {
                "long": "--tools_unprotected_shell_access",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Tools)
            }
        }
    )
    tools_forbidden: List[str] = Field(
        default_factory=lambda: ["Any", "List", "Dict", "launch_nukes"],
        description="Blacklist certain tools. Specify multiple times to forbid several tools. The default blacklist contains some common module imports that can pollute a tools.py namespace. You can override this in a character folders config.json if necessary.",
        json_schema_extra={
            "argparse": {
                "short": "-d",
                "long": "--tools_forbidden",
                "action": "append",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Tools, very_important=True, motd=True)
            }
        }
    )
    tools_hint: str = Field(
        default="",
        description="Text that will be appended to the system prompt when use_tools is true.",
        json_schema_extra={
            "argparse": {
                "long": "--tools_hint",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Tools)
            }
        }
    )
    tools_inject_dependency_function: str = Field(
        default="",
        description="API only. Set a callback function to be called whenever an tool-using Ai is initialized. The callback will receive one argument: The tools.py module. You can use this to inject dependency or modify the module after it is loaded.",
        json_schema_extra={
            "argparse": {
                "long": "--tools_inject_dependency_function",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Tools, motd=True)
            }
        }
    )
    tools_inject_ghostbox: bool = Field(
        default=True,
        description="Inject a reference to ghostbox itself into an AI's tool module. This will make the '_ghostbox_plumbing' identifier available in the tools module and point it to the running ghostbox Plumbing instance. Disabling this will break many of the standard AI tools that ship with ghostbox.",
        json_schema_extra={
            "argparse": {
                "long": "--tools_inject_ghostbox",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.Tools)
            }
        }
    )
    use_tools: bool = Field(
        default=False,
        description="Enable use of tools, i.e. model may call python functions. This will do nothing if tools.py isn't present in the char directory. If tools.py is found, this will be automatically enabled.",
        json_schema_extra={
            "argparse": {
                "long": "--use_tools",
                "boolean_optional_action": True,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Tools, very_important=True, motd=True)
            }
        }
    )
    var_file: List[str] = Field(
        default_factory=list,
        description="Files that will be added to the list of variables that can be expanded. E.g. -Vmemory means {$memory} will be expanded to the contents of file memory, provided expand_user_input is set. Can be used to override values set in character folders. Instead of using this, you can also just type {[FILENAME]} to have it be automatically expanded with the contents of FILENAME, provided --dynamic_file_vars is enabled.",
        json_schema_extra={
            "argparse": {
                "short": "-x",
                "long": "--var_file",
                "action": "append",
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.Interface, motd=True)
            }
        }
    )

    # Sampling Parameters (from backends.py)
    temperature: float = Field(
        default=0.8,
        description="Adjust the randomness of the generated text.",
        json_schema_extra={
            "argparse": {
                "long": "--temperature",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.SamplingParameters, very_important=True, is_option=True)
            }
        }
    )
    dynatemp_range: float = Field(
        default=0.0,
        description="Dynamic temperature range. The final temperature will be in the range of `[temperature - dynatemp_range; temperature + dynatemp_range]`",
        json_schema_extra={
            "argparse": {
                "long": "--dynatemp_range",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    dynatemp_exponent: float = Field(
        default=1.0,
        description="Dynamic temperature exponent.",
        json_schema_extra={
            "argparse": {
                "long": "--dynatemp_exponent",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    top_k: int = Field(
        default=40,
        description="Limit the next token selection to the K most probable tokens.",
        json_schema_extra={
            "argparse": {
                "long": "--top_k",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    top_p: float = Field(
        default=0.95,
        description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.",
        json_schema_extra={
            "argparse": {
                "long": "--top_p",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Porcelain, group=ArgumentGroup.SamplingParameters, very_important=True, is_option=True)
            }
        }
    )
    min_p: float = Field(
        default=0.05,
        description="The minimum probability for a token to be considered, relative to the probability of the most likely token.",
        json_schema_extra={
            "argparse": {
                "long": "--min_p",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    n_indent: int = Field(
        default=0,
        description="Specify the minimum line indentation for the generated text in number of whitespace characters. Useful for code completion tasks.",
        json_schema_extra={
            "argparse": {
                "long": "--n_indent",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    n_keep: int = Field(
        default=0,
        description="Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded. The number excludes the BOS token. By default, this value is set to `0`, meaning no tokens are kept. Use `-1` to retain all tokens from the prompt.",
        json_schema_extra={
            "argparse": {
                "long": "--n_keep",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    typical_p: float = Field(
        default=1.0,
        description="Enable locally typical sampling with parameter p.",
        json_schema_extra={
            "argparse": {
                "long": "--typical_p",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    repeat_penalty: float = Field(
        default=1.1,
        description="Control the repetition of token sequences in the generated text.",
        json_schema_extra={
            "argparse": {
                "long": "--repeat_penalty",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, very_important=True, is_option=True)
            }
        }
    )
    repeat_last_n: int = Field(
        default=64,
        description="Last n tokens to consider for penalizing repetition.",
        json_schema_extra={
            "argparse": {
                "long": "--repeat_last_n",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    presence_penalty: float = Field(
        default=0.0,
        description="Repeat alpha presence penalty.",
        json_schema_extra={
            "argparse": {
                "long": "--presence_penalty",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, very_important=True, is_option=True)
            }
        }
    )
    frequency_penalty: float = Field(
        default=0.0,
        description="Repeat alpha frequency penalty.",
        json_schema_extra={
            "argparse": {
                "long": "--frequency_penalty",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, very_important=True, is_option=True)
            }
        }
    )
    dry_multiplier: float = Field(
        default=0.8,
        description="Set the DRY (Don't Repeat Yourself) repetition penalty multiplier.",
        json_schema_extra={
            "argparse": {
                "long": "--dry_multiplier",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    dry_base: float = Field(
        default=1.75,
        description="Set the DRY repetition penalty base value.",
        json_schema_extra={
            "argparse": {
                "long": "--dry_base",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    dry_allowed_length: int = Field(
        default=2,
        description="Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length).",
        json_schema_extra={
            "argparse": {
                "long": "--dry_allowed_length",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    dry_penalty_last_n: int = Field(
        default=-1,
        description="How many tokens to scan for repetitions.",
        json_schema_extra={
            "argparse": {
                "long": "--dry_penalty_last_n",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    dry_sequence_breakers: List[str] = Field(
        default_factory=lambda: ["\n", ":", "\"", "*"],
        description="Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted.",
        json_schema_extra={
            "argparse": {
                "long": "--dry_sequence_breakers",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    xtc_probability: float = Field(
        default=0.5,
        description="Set the chance for token removal via XTC sampler.\nXTC means 'exclude top choices'. This sampler, when it triggers, removes all but one tokens above a given probability threshold. Recommended for creative tasks, as language tends to become less stereotypical, but can make a model less effective at structured output or intelligence-based tasks.\nSee original xtc PR by its inventor https://github.com/oobabooga/text-generation-webui/pull/6335",
        json_schema_extra={
            "argparse": {
                "long": "--xtc_probability",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    xtc_threshold: float = Field(
        default=0.1,
        description="Set a minimum probability threshold for tokens to be removed via XTC sampler.\nXTC means 'exclude top choices'. This sampler, when it triggers, removes all but one tokens above a given probability threshold. Recommended for creative tasks, as language tends to become less stereotypical, but can make a model less effective at structured output or intelligence-based tasks.\nSee original xtc PR by its inventor https://github.com/oobabooga/text-generation-webui/pull/6335",
        json_schema_extra={
            "argparse": {
                "long": "--xtc_threshold",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    mirostat: int = Field(
        default=0,
        description="Enable Mirostat sampling, controlling perplexity during text generation.",
        json_schema_extra={
            "argparse": {
                "long": "--mirostat",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    mirostat_tau: float = Field(
        default=5.0,
        description="Set the Mirostat target entropy, parameter tau.",
        json_schema_extra={
            "argparse": {
                "long": "--mirostat_tau",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    mirostat_eta: float = Field(
        default=0.1,
        description="Set the Mirostat learning rate, parameter eta.",
        json_schema_extra={
            "argparse": {
                "long": "--mirostat_eta",
                "type": float,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    grammar: Optional[str] = Field(
        default=None,
        description="Set grammar for grammar-based sampling.",
        json_schema_extra={
            "argparse": {
                "long": "--grammar",
                "type": str,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    grammar_lazy: bool = Field(
        default=False,
        description="This parameter controls whether the grammar (specified by `grammar` or `json_schema`) is applied strictly from the beginning of generation, or if its activation is deferred until a specific trigger is encountered.",
        json_schema_extra={
            "argparse": {
                "long": "--grammar_lazy",
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    grammar_triggers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="This parameter defines the conditions under which a lazy grammar (when `grammar_lazy` is `true`) should become active. Each object in the array represents a single trigger.",
        json_schema_extra={
            "argparse": {
                "long": "--grammar_triggers",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    preserved_tokens: List[int] = Field(
        default_factory=list,
        description="A list of token pieces to be preserved during sampling and grammar processing.",
        json_schema_extra={
            "argparse": {
                "long": "--preserved_tokens",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    enable_thinking: bool = Field(
        default=True,
        description="Turn on reasoning for thinking models, disable it otherwise (if possible).",
        json_schema_extra={
            "argparse": {
                "long": "--enable_thinking",
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    chat_template_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"enable_thinking": True},
        description="This parameter allows you to pass arbitrary key-value arguments directly to the Jinja chat template used for prompt formatting. These arguments can be used within the Jinja template to control conditional logic, insert dynamic content, or modify the template's behavior.",
        json_schema_extra={
            "argparse": {
                "long": "--chat_template_kwargs",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Set a JSON schema for grammar-based sampling (e.g. `{"items": {"type": "string"}, "minItems": 10, "maxItems": 100}` of a list of strings, or `{}` for any JSON). See [tests](../../tests/test-json-schema-to-grammar.cpp) for supported features.',
        json_schema_extra={
            "argparse": {
                "long": "--json_schema",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    seed: int = Field(
        default=-1,
        description="Set the random number generator (RNG) seed.",
        json_schema_extra={
            "argparse": {
                "long": "--seed",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    ignore_eos: bool = Field(
        default=False,
        description="Ignore end of stream token and continue generating.",
        json_schema_extra={
            "argparse": {
                "long": "--ignore_eos",
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    logit_bias: List[List[Union[int, float, bool]]] = Field(
        default_factory=list,
        description='Modify the likelihood of a token appearing in the generated text completion. For example, use `"logit_bias": [[15043,1.0]]` to increase the likelihood of the token \'Hello\', or `"logit_bias": [[15043,-1.0]]` to decrease its likelihood. Setting the value to false, `"logit_bias": [[15043,false]]` ensures that the token `Hello` is never produced. The tokens can also be represented as strings, e.g. `[["Hello, World!",-0.5]]` will reduce the likelihood of all the individual tokens that represent the string `Hello, World!`, just like the `presence_penalty` does.',
        json_schema_extra={
            "argparse": {
                "long": "--logit_bias",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    n_probs: int = Field(
        default=0,
        description="If greater than 0, the response also contains the probabilities of top N tokens for each generated token given the sampling settings. Note that for temperature < 0 the tokens are sampled greedily but token probabilities are still being calculated via a simple softmax of the logits without considering any other sampler settings.",
        json_schema_extra={
            "argparse": {
                "long": "--n_probs",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    min_keep: int = Field(
        default=0,
        description="If greater than 0, force samplers to return N possible tokens at minimum.",
        json_schema_extra={
            "argparse": {
                "long": "--min_keep",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    t_max_predict_ms: int = Field(
        default=0,
        description="Set a time limit in milliseconds for the prediction (a.k.a. text-generation) phase. The timeout will trigger if the generation takes more than the specified time (measured since the first token was generated) and if a new-line character has already been generated. Useful for FIM applications.",
        json_schema_extra={
            "argparse": {
                "long": "--t_max_predict_ms",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    id_slot: int = Field(
        default=-1,
        description="Assign the completion task to an specific slot. If is -1 the task will be assigned to a Idle slot.",
        json_schema_extra={
            "argparse": {
                "long": "--id_slot",
                "type": int,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    cache_prompt: bool = Field(
        default=True,
        description="Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests. Because (depending on the backend) the logits are **not** guaranteed to be bit-for-bit identical for different batch sizes (prompt processing vs. token generation) enabling this option can cause nondeterministic results.",
        json_schema_extra={
            "argparse": {
                "long": "--cache_prompt",
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    return_tokens: bool = Field(
        default=False,
        description="Return the raw generated token ids in the `tokens` field. Otherwise `tokens` remains empty.",
        json_schema_extra={
            "argparse": {
                "long": "--return_tokens",
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    samplers: List[str] = Field(
        default_factory=lambda: ["min_p", "xtc", "dry", "temperature"],
        description="The order the samplers should be applied in. An array of strings representing sampler type names. If a sampler is not set, it will not be used. If a sampler is specified more than once, it will be applied multiple times.",
        json_schema_extra={
            "argparse": {
                "short": "-S",
                "long": "--samplers",
                "nargs": "*",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, very_important=True, motd=True, is_option=True)
            }
        }
    )
    timings_per_token: bool = Field(
        default=False,
        description="Include prompt processing and text generation speed information in each response.",
        json_schema_extra={
            "argparse": {
                "long": "--timings_per_token",
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    post_sampling_probs: Optional[Any] = Field(
        default=None,
        description="Returns the probabilities of top `n_probs` tokens after applying sampling chain.",
        json_schema_extra={
            "argparse": {
                "long": "--post_sampling_probs",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    response_fields: Optional[List[str]] = Field(
        default=None,
        description='A list of response fields, for example: `"response_fields": ["content", "generation_settings/n_predict"]`. If the specified field is missing, it will simply be omitted from the response without triggering an error. Note that fields with a slash will be unnested; for example, `generation_settings/n_predict` will move the field `n_predict` from the `generation_settings` object to the root of the response and give it a new name.',
        json_schema_extra={
            "argparse": {
                "long": "--response_fields",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    lora: List[Dict[str, Any]] = Field(
        default_factory=list,
        description='A list of LoRA adapters to be applied to this specific request. Each object in the list must contain `id` and `scale` fields. For example: `[{"id": 0, "scale": 0.5}, {"id": 1, "scale": 1.1}]`. If a LoRA adapter is not specified in the list, its scale will default to `0.0`. Please note that requests with different LoRA configurations will not be batched together, which may result in performance degradation.',
        json_schema_extra={
            "argparse": {
                "long": "--lora",
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )
    add_generation_prompt: bool = Field(
        default=True,
        description="Include the prompt used to generate in the result.",
        json_schema_extra={
            "argparse": {
                "long": "--add_generation_prompt",
                "type": bool,
                "tag": ArgumentTag(type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True)
            }
        }
    )

    # Hidden/Internal options (not directly exposed via CLI, but used internally or via API)

    #__api__: bool = Field(default=False, exclude=True)    
    user_config: str = Field(default="", exclude=True)


# --- TypedDict for kwargs (for mypy static analysis) ---
# This must manually mirror the Config class fields.
class ConfigKwargs(TypedDict, total=False):
    include: List[str]
    template_include: List[str]
    history: bool
    history_retroactive_vars: bool
    history_force_alternating_roles: bool
    prompt_format: str
    stop: List[str]
    character_folder: str
    prompt: Optional[str]
    endpoint: str
    client: bool
    remote_host: str
    remote_port: int
    backend: LLMBackend
    api_key: str
    google_api_key: str
    deepseek_api_key: str
    qwen_api_key: str
    iflow_api_key: str
    iflow_prefered_model: str
    google_prefered_model: str
    qwen_prefered_model: str
    max_length: int
    max_context_length: int
    chat_user: str
    mode: str
    force_params: bool
    llamacpp_thinking_json_fix: bool
    llamacpp_auto_enable_thinking: bool
    model: str
    grammar_file: str
    chat_ai: str
    stream: bool
    http: bool
    websock: bool
    websock_host: str
    websock_port: int
    http_host: str
    http_port: int
    http_override: bool
    multiline: bool
    multiline_delimiter: str
    color: bool
    text_ai_color: str
    text_ai_style: str
    dynamic_file_vars: bool
    dynamic_file_vars_unsafe: bool
    dynamic_file_vars_max_depth: int
    warn_trailing_space: bool
    warn_unsupported_sampling_parameter: bool
    warn_audio_activation_phrase: bool
    warn_hint: bool
    json_grammar: bool
    response_format: Union[str, Dict[str, Any]]
    stream_flush: str
    stream_flush_flex_value: int
    cli_prompt: str
    cli_prompt_color: str
    hint: str
    hint_sticky: bool
    tts: bool
    tts_model: TTSModel
    tts_zonos_model: ZonosTTSModel
    tts_orpheus_model: str
    tts_llm_server: str
    tts_language: str
    tts_output_method: TTSOutputMethod
    tts_filter: List[str]
    tts_websock: bool
    tts_websock_host: str
    tts_websock_port: int
    tts_interrupt: bool
    tts_program: str
    tts_clone_dir: str
    tts_tortoise_quality: str
    tts_volume: float
    tts_modify_system_msg: bool
    tts_rate: int
    tts_additional_arguments: str
    image_watch: bool
    image_watch_clear_history: bool
    image_watch_dir: str
    image_watch_msg: str
    image_watch_hint: str
    whisper_model: str
    tts_voice: str
    tts_subtitles: bool
    config_file: str
    chat_show_ai_prompt: bool
    smart_context: bool
    hide: bool
    sound_output_device_index: Optional[int]
    sound_input_device_index: Optional[int]
    sound_list_output_devices: bool
    audio: bool
    audio_silence_threshold: int
    audio_activation_phrase: str
    audio_activation_period_ms: int
    audio_interrupt: bool
    audio_activation_phrase_keep: bool
    audio_show_transcript: bool
    audio_websock: bool
    audio_websock_host: str
    audio_websock_port: int
    warn_audio_activation_phrase: bool
    verbose: bool
    log_time: bool
    quiet: bool
    history_drop_on_generation_error: bool
    stderr: bool
    stdout: bool
    expand_user_input: bool
    tools_unprotected_shell_access: bool
    tools_forbidden: List[str]
    tools_hint: str
    tools_inject_dependency_function: str
    tools_inject_ghostbox: bool
    use_tools: bool
    var_file: List[str]
    temperature: float
    dynatemp_range: float
    dynatemp_exponent: float
    top_k: int
    top_p: float
    min_p: float
    n_indent: int
    n_keep: int
    typical_p: float
    repeat_penalty: float
    repeat_last_n: int
    presence_penalty: float
    frequency_penalty: float
    dry_multiplier: float
    dry_base: float
    dry_allowed_length: int
    dry_penalty_last_n: int
    dry_sequence_breakers: List[str]
    xtc_probability: float
    xtc_threshold: float
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    grammar: Optional[str]
    grammar_lazy: bool
    grammar_triggers: List[Dict[str, Any]]
    preserved_tokens: List[int]
    enable_thinking: bool
    chat_template_kwargs: Dict[str, Any]
    json_schema: Optional[Dict[str, Any]]
    seed: int
    ignore_eos: bool
    logit_bias: List[List[Union[int, float, bool]]]
    n_probs: int
    min_keep: int
    t_max_predict_ms: int
    id_slot: int
    cache_prompt: bool
    return_tokens: bool
    samplers: List[str]
    timings_per_token: bool
    post_sampling_probs: Optional[Any]
    response_fields: Optional[List[str]]
    lora: List[Dict[str, Any]]
    add_generation_prompt: bool
    user_config: str    
