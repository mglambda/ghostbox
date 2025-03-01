
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
ArgumentGroup = Enum("ArgumentGroup", "Hyperparameter Terminal Characters TTS Audio Images Backend LlamaCPP OpenAI")

class ArgumentTag(BaseModel):
    """Metadata associated with a command line argument."""
    type: ArgumentType
    group: ArgumentGroup

    # this is for e.g. streaming or temperature.
    very_important: bool = False
    
    # wether changing the value of this argument may start a service
    service: bool = False

    # for inclusion in the message of the day/tip
    motd: bool = False

    # same help that is printed in terminal on --help
    help: str = ""
    
    
api_default_options = {
    "color" : False,
    "verbose" : False,
    "log_time" : True,
    "cli_prompt" : "",
    "dynamic_file_vars": False
}


