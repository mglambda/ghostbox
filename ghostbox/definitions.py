
from enum import Enum

LLMBackend = Enum("LLMBackend", "llamacpp koboldcpp openai dummy")

api_default_options = {
    "color" : False,
    "verbose" : False,
    "log_time" : True,
    "cli_prompt" : ""
}
    
