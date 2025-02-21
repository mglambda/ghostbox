
from enum import Enum

LLMBackend = Enum("LLMBackend", "generic legacy llamacpp koboldcpp openai dummy")

# this isn't used yet anywhere, but it's nice to have here already for documentation
PromptFormatTemplateSpecialValue = Enum("PromptFormatTemplateSpecialValue", "auto guess raw")

api_default_options = {
    "color" : False,
    "verbose" : False,
    "log_time" : True,
    "cli_prompt" : ""
}
    
