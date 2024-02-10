import time, requests
from abc import ABC, abstractmethod
from functools import *
from ghostbox.util import *
from ghostbox.streaming import *

# defined here for convenience, and as reference. These parameters should be handled by the generate method as payload.
default_params = {
    "repeat_penalty": 1.1,
    "repeat_last_n" : 64,
    "penalize_nl" : True,
    "presence_penalty" : 0.0,
    "frequency_penalty" : 0.0,
#    "penalty_prompt" : "", # they want this to be 'null'??
    "mirostat" : 0, # 0=disabled, 1= mirostat, 2=mirostat 2.0
    "mirostat_tau" : 5.0,
    "temperature": 0.7,
    "top_p": 0.92,
    "top_k": 0,
    "min_p" : 0.05,
    "typical_p": 1.0,
    "tfs_z": 1,

    # this one is an issue, koboldcpp has this    
    # see also https://github.com/ggerganov/llama.cpp/discussions/3914    
    #"sampler_order": [6, 0, 1, 3, 4, 2, 5],    
    "n_keep" : 0,
    "stop" : [],
    "tfz" : 1.0,
    "seed" : -1,
    "ignore_eos" : False,
    "logit_bias" : [],
    "n_probs" : 0,
    "slot_id" : -1, # this is very llama specific

    # sort of content-y parameters
    #"system_prompt" : "", # this seems to crash llama    
    "prompt" : "",
    #"stream" : False, #temporarily commented because it conflicts with argparse definition
    "grammar" : "",
    "image_data" : [],
    "n_predict" : 0,
    #"n_ctx" : 0, # this one seems to have no effect on llama
    "cache_prompt" : True
}

class AIBackend(ABC):
    """Abstract interface to a backend server, like llama.cpp or openai etc. Use the Program.make*Payload methods to make corresponding payloads. All backends must handle those dictionaries, but not all backends may support all features."""

    @abstractmethod
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.last_error = ""

    def getLastError(self):
        return self.last_error
        
    @abstractmethod
    def getName(self):
        """Returns the name of the backend. This can be compared to the --backend command line option."""
        pass
        
    @abstractmethod        
    def generate(self, payload):
        """Takes a payload dictionary similar to default_params. Returns a result object specific to the backend. Use handleResult to unpack its content."""
        pass

    @abstractmethod
    def handleGenerateResult(self, result):
        """Takes a result from the generate method and returns the generated string."""
        pass

    @abstractmethod
    def generateStreaming(self, payload, callback=lambda w: print(w)):
        """Takes a payload dictionary similar to default_params, and returns nothing. This will begin generating in streaming mode. The callback provided will be called with individual tokens that are being generated. This function is non-blocking"""
        pass
                                      
    @abstractmethod
    def tokenize(self, w):
        """Takes a string w and returns a list of tokens."""
        pass

    def health(self):
        """Returns a string indicating the status of the backend."""
        pass
    
    
class LlamaCPPBackend(AIBackend):
    """Bindings for the formidable Llama.cpp program."""

    def __init__(self, endpoint="http://localhost:8080"):
        super().__init__(endpoint)
        self._lastResult = None

    def getLastJSON(self):
        return self._lastResult

    def getName(self):
        return "llama.cpp"
    
    def generate(self, payload):
        super().generate(payload)
        # dict is llama format by default, so we don't need to do anything here
        return requests.post(self.endpoint + "/completion", json=payload)

    def handleGenerateResult(self, result):
        if result.status_code == 200:
            self._lastResult = result.json()
            return result.json()['content']
        self.last_error = "HTTP request with status code " + str(r.status_code)
        return None

    def generateStreaming(self, payload, callback=lambda w: print(w), flag=None):
        return streamPrompt(lambda d: callback(d["content"]), flag, self.endpoint + "/completion", payload)
    
    def tokenize(self, w):
        r = requests.post(self.endpoint + "/tokenize", json={"content" : w})
        if r.status_code == 200:
            return r.json()["tokens"]
        return []
    
    def health(self):
        r = requests.get(self.endpoint + "/health")
        if r.status_code == 200:
            return r.json()["status"]
        return "error " + str(r.status_code)

