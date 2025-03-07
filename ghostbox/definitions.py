
from enum import Enum
from pydantic import BaseModel
from typing import *


LLMBackend = Enum("LLMBackend", "generic legacy llamacpp koboldcpp openai dummy")

# these are the models supported by ghostbox-tts
TTSModel = Enum("TTSModel", "zonos kokoro xtts")

# these are ways of playing sound that are supported by ghostbox-tts
TTSOutputMethod = Enum("TTSOutputMethod", "default websock")


# this isn't used yet anywhere, but it's nice to have here already for documentation
PromptFormatTemplateSpecialValue = Enum("PromptFormatTemplateSpecialValue", "auto guess raw")


ArgumentType = Enum("ArgumentType", "Porcelain Plumbing")
ArgumentGroup = Enum("ArgumentGroup", "General Generation Interface Characters Templates TTS Audio Images Tools Backend SamplingParameters LlamaCPP OpenAI")

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
            return "It is a " + self.type.name.lower() + " option in the " + self.group.name.lower() + " group."
        return "It is a " + self.type.name.lower() + " command in the " + self.group.name.lower() + " group."
    
    def show_description(self) -> str:
        w = ""
        w += self.show_type()
        w += "\nYou can set it with `/set " + self.name + " VALUE` or provide it as a command line parameter with `--" + self.name + "`"
        return w
        
    
class SamplingParameterSpec(BaseModel):
    """Sampling parameters can be provided to backends to influence a model's inference behaviour.
    Most commonly this is temperature, presence penalty etc. However, here we take sampling parameter in the broadest sense, including samplers, CFG and control vectors.
    Note that this class provides only the specification of a sampling_parameter. This is for documentation and to keep track of which backend supports which parameters."""

    name: str
    description: str
    default_value: Any
    
api_default_options = {
    "color" : False,
    "verbose" : False,
    "log_time" : True,
    "cli_prompt" : "",
    "dynamic_file_vars": False,
    "max_context_length":32000
}


