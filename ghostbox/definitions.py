
from enum import Enum

LLMBackend = Enum("LLMBackend", "llamacpp koboldcpp openai openai_generic dummy")

api_default_options = {
    "color" : False,
    "verbose" : False,
    "log_time" : True,
    "cli_prompt" : ""
}
    
