from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, model_serializer
import copy
from pydantic.types import Json
from typing import *


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

ChatContent = str | List[ChatContentComplex] | Dict


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
        if type(self.content) == str:
            return self.content
        elif type(self.content) == ChatContentComplex:
            return self.content.get_text()
        elif type(self.content) == dict:
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
    def make_image_message(text: str, images: List[ImageRef], **kwargs) -> 'ChatMessage':
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

        return ChatMessage(role="user", content=complex_content_list, **kwargs)


LLMBackend = Enum("LLMBackend", "generic legacy llamacpp koboldcpp openai google deepseek dummy")

# these are the models supported by ghostbox-tts
TTSModel = Enum("TTSModel", "zonos kokoro xtts polly orpheus")
ZonosTTSModel = Enum("ZonosTTSModel", "hybrid transformer")

# these are ways of playing sound that are supported by ghostbox-tts
TTSOutputMethod = Enum("TTSOutputMethod", "default websock")


# this isn't used yet anywhere, but it's nice to have here already for documentation
PromptFormatTemplateSpecialValue = Enum(
    "PromptFormatTemplateSpecialValue", "auto guess raw"
)


ArgumentType = Enum("ArgumentType", "Porcelain Plumbing")
ArgumentGroup = Enum(
    "ArgumentGroup",
    "General Generation Interface Characters Templates TTS Audio Images Tools Backend SamplingParameters LlamaCPP OpenAI Google",
)


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
        
        
