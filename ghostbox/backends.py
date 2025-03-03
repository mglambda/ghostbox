import time, requests, threading
from abc import ABC, abstractmethod
from functools import *
from pydantic import BaseModel
from ghostbox.util import *
from ghostbox.definitions import *
from ghostbox.streaming import *

# Server request parameters
# defined here for convenience, and as reference. These parameters should be handled by the generate method as payload.
# note that these are based on the parameters for the llama.cpp server. Since ghostbox was developed with llama.cpp as its main backend, these serve as the default and
#  reference. Other backends need to translate these to their respective parameters if necessary.
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
    # update: I think n_ctx has been renamed to max_tokens - they aren't documenting this.
    #  Anyhow, max_tokens would match the OAI API.
    "cache_prompt" : True
}

class Timings(BaseModel):
    """Performance statistics for LLM backends.
Most backends give timing statistics, though the format and particular stats vary. This class unifies the interface and boils it down to only the stats we care about."""


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
    

class AIBackend(ABC):
    """Abstract interface to a backend server, like llama.cpp or openai etc. Use the Program.make*Payload methods to make corresponding payloads. All backends must handle those dictionaries, but not all backends may support all features."""

    @abstractmethod
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.stream_done = threading.Event()
        self.last_error = ""

    @abstractmethod
    def getLastError(self):
        return self.last_error

    @abstractmethod
    def getLastJSON(self):
        pass
    
    def waitForStream(self):
        self.stream_done.wait()
        
    @abstractmethod
    def getName(self):
        """Returns the name of the backend. This can be compared to the --backend command line option."""
        pass

    @abstractmethod
    def getMaxContextLength(self):
        """Returns the default setting for maximum context length that is set serverside. Often, this should be the maximum context the model is trained on.
        This should return -1 if the backend is unable to determine the maximum context (some backends don't let you query this at all)."""
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
        """Takes a payload dictionary similar to default_params and begins streaming generated tokens to a callback function. Returns True if there was a HTTP error, which you can check with getLastError().
        callback - A function taking one string argument, which will be the generated tokens.
        payload - A dictionary similar to default_params. If this doesn't contain "stream" : True, this function may fail or have no effect.
        returns - True on status code != 200"""
        pass
                                      
    @abstractmethod
    def tokenize(self, w):
        """Takes a string w and returns a list of tokens as ints."""
        pass

    @abstractmethod
    def detokenize(self, ts):
        """Takes a list of tokens as ints, and returns a string consisting of the tokens."""
        pass
    
    @abstractmethod
    def health(self):
        """Returns a string indicating the status of the backend."""
        pass

    @abstractmethod
    def timings(self, json=None) -> Optional[Timings]:
        """Returns performance statistics for this backend.
        The method can take a json parameter, which may be a return value of the getLastJSON method. In this case, timings for that result are returned.
        Otherwise, if called without the json parameter, timings for the last request are returned.
        The method may return none if there hasn't been a request yet to determine timings for and no json is provided."""
        pass
    
class LlamaCPPBackend(AIBackend):
    """Bindings for the formidable Llama.cpp based llama-server program."""

    def __init__(self, endpoint="http://localhost:8080"):
        super().__init__(endpoint)
        self._lastResult = None

    def getLastError(self):
        return super().getLastError()
        
    def getLastJSON(self):
        return self._lastResult

    def getName(self):
        #return "llama.cpp"
        return LLMBackend.llamacpp.name

    def getMaxContextLength(self):
        return -1
    
    def generate(self, payload):
        super().generate(payload)
        # dict is llama format by default, so we don't need to do anything here
        return requests.post(self.endpoint + "/completion", json=payload)

    def handleGenerateResult(self, result):
        if result.status_code != 200:
            self.last_error = "HTTP request with status code " + str(result.status_code)
            return None
        self._lastResult = result.json()
        return result.json()['content']

    def _makeLlamaCallback(self, callback):
        def f(d):
            if d["stop"]:
                self._lastResult = d
                return 
            callback(d["content"])
        return f

    def generateStreaming(self, payload, callback=lambda w: print(w)):
        self.stream_done.clear()
        r = streamPrompt(self._makeLlamaCallback(callback), self.stream_done, self.endpoint + "/completion", payload)
        if r.status_code != 200:
            self.last_error = "streaming HTTP request with status code " + str(r.status_code)
            self.stream_done.set()
            return True
        return False
    
    def tokenize(self, w):
        r = requests.post(self.endpoint + "/tokenize", json={"content" : w})
        if r.status_code == 200:
            return r.json()["tokens"]
        return []

    def detokenize(self, ts):
        r = requests.post(self.endpoint + "/detokenize", json={"tokens" : ts})
        if r.status_code == 200:
            return r.json()["content"]
        return []
    
    def health(self):
        r = requests.get(self.endpoint + "/health")
        if r.status_code != 200:
            return "error " + str(r.status_code)
        return r.json()["status"]

    def timings(self, json=None) -> Optional[Timings]:
        if json is None:
            if (json := self._lastResult) is None:
                return None
            time = json["timings"]
            return Timings(
                prompt_n= time["prompt_n"],
                predicted_n = time["predicted_n"],
                prompt_ms=time["prompt_ms"],
                predicted_ms=time["predicted_ms"],
                predicted_per_token_ms=time["predicted_per_token_ms"],
                predicted_per_second=time["predicted_per_second"],
                truncated=json["truncated"],
                cached_n=json["tokens_cached"],
                original_timings=time
                )
                
                
                
                
                
    
class OpenAILegacyBackend(AIBackend):
    """Backend for the official OpenAI API. The legacy version routes to /v1/completions, instead of the regular /v1/chat/completion."""
    
    def __init__(self, api_key, endpoint="https://api.openai.com"):
        super().__init__(endpoint)
        self.api_key = api_key
        self._lastResult = None

    def getLastError(self):
        return super().getLastError()

    def getLastJSON(self):
        return self._lastResult

    def getName(self):
        return LLMBackend.openai.name

    def getMaxContextLength(self):
        return -1
    
    def generate(self, payload):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": payload.get("model", "text-davinci-003"),
            "prompt": payload["prompt"],
            "max_tokens": payload.get("n_predict", 150),
            "temperature": payload.get("temperature", 0.7),
            "top_p": payload.get("top_p", 1.0),
            "frequency_penalty": payload.get("frequency_penalty", 0.0),
            "presence_penalty": payload.get("presence_penalty", 0.0)
        }
        response = requests.post(self.endpoint + "/v1/completions", headers=headers, json=data)
        if response.status_code != 200:
            self.last_error = f"HTTP request with status code {response.status_code}: {response.text}"
            return None
        self._lastResult = response.json()
        return response.json()

    def handleGenerateResult(self, result):
        if not result:
            return ""
        return result['choices'][0]['text']

    def generateStreaming(self, payload, callback=lambda w: print(w)):
        self.stream_done.clear()        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": payload.get("model", "text-davinci-003"),
            "prompt": payload["prompt"],
            "max_tokens": payload.get("n_predict", 150),
            "temperature": payload.get("temperature", 0.7),
            "top_p": payload.get("top_p", 1.0),
            "frequency_penalty": payload.get("frequency_penalty", 0.0),
            "presence_penalty": payload.get("presence_penalty", 0.0),
            "stream": True
        }
        #response = requests.post(self.endpoint + "/v1/completions", headers=headers, json=data)
        def openaiCallback(d):
            callback(d['choices'][0]['text'])
            
        response = streamPrompt(openaiCallback, self.stream_done, self.endpoint + "/v1/completions", json=data, headers=headers)
        if response.status_code != 200:
            self.last_error = f"HTTP request with status code {response.status_code}: {response.text}"
            self.stream_done.set()
            return True
        return False
    

    def tokenize(self, w):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {"prompt": w}
        response = requests.post(self.endpoint + "/v1/tokenize", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["tokens"]
        return []

    def detokenize(self, ts):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {"tokens": ts}
        response = requests.post(self.endpoint + "/v1/detokenize", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["content"]
        return []

    def health(self):
        # OpenAI API does not have a direct health check endpoint
        return "OpenAI API is assumed to be healthy."
    
    def timings(self) -> Optional[Timings]:
        # FIXME: not implemented yet
        return None
    
class OpenAIBackend(AIBackend):
    """Backend for the official OpenAI API. This is used for the company of Altman et al, but also serves as a general purpose API suported by various backends, including llama.cpp, llama-box, and many others."""
    
    def __init__(self, api_key, endpoint="https://api.openai.com"):
        super().__init__(endpoint)
        self.api_key = api_key
        self._lastResult = None

    def getLastError(self):
        return super().getLastError()

    def getLastJSON(self):
        return self._lastResult

    def getName(self):
        return LLMBackend.openai.name

    def getMaxContextLength(self):
        return -1
    
    def generate(self, payload):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": payload.get("model", "text-davinci-003"),
            "prompt": payload["prompt"],
            "max_tokens": payload.get("n_predict", 150), # FIXME: this is changed in the API for o1 and up
            "temperature": payload.get("temperature", 0.7),
            "top_p": payload.get("top_p", 1.0),
            "frequency_penalty": payload.get("frequency_penalty", 0.0),
            "presence_penalty": payload.get("presence_penalty", 0.0)
        }

        # the /V1/chat/completions endpoint expects structured data of user/assistant pairs        
        data |= self._dataFromPayload(payload)
        response = requests.post(self.endpoint + "/v1/chat/completions", headers=headers, json=data)
        if response.status_code != 200:
            self.last_error = f"HTTP request with status code {response.status_code}: {response.text}"
            return None
        self._lastResult = response.json()
        return response.json()

    def handleGenerateResult(self, result):
        if not result:
            return ""
        return result['choices'][0]["message"]["content"]

    def generateStreaming(self, payload, callback=lambda w: print(w)):
        self.stream_done.clear()        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": payload.get("model", "gpt-3"),
            "max_tokens": payload.get("n_predict", 150),
            "temperature": payload.get("temperature", 0.7),
            "top_p": payload.get("top_p", 1.0),
            "frequency_penalty": payload.get("frequency_penalty", 0.0),
            "presence_penalty": payload.get("presence_penalty", 0.0),
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        # the /V1/chat/completions endpoint expects structured data of user/assistant pairs        
        data |= self._dataFromPayload(payload)
        def openaiCallback(d):
            self._lastResult = d            
            choice = d["choices"][0]
            maybeChunk = choice["delta"].get("content", None)
            if maybeChunk is not None:
                callback(maybeChunk)
            
        response = streamPrompt(openaiCallback, self.stream_done, self.endpoint + "/v1/chat/completions", json=data, headers=headers)
        if response.status_code != 200:
            self.last_error = f"HTTP request with status code {response.status_code}: {response.text}"
            self.stream_done.set()            
            return True
        return False

    def _dataFromPayload(self, payload):
        """Take a payload dictionary from Plumbing and return dictionary with elements specific to the chat/completions endpoint.
        This expects payload to include the messages key, with various dictionaries in it, unlike other backends."""
        messages = [{"role": "system",
                     "content" : payload["system"]}]
        # story is list of dicts with role and content keys
        # we go through story one by one, mostly because of images
        for story_item in payload["story"]:
            if "image_id" in story_item:
                # images is more complicated, see https://platform.openai.com/docs/guides/vision
                # API wants the content field of an image message to be a list of dicts, not a string
                # the dicts have the type field, which determines wether its a user msg (text) or image (image-url)
                image_id = story_item["image_id"]
                image_content_list = []
                image_content_list.append({"type":"text",
                                           "content": story_item["content"]})
                if "images" not in payload or image_id not in payload["images"]:
                    printerr("warning: image with id " + str(image_id) + " not found.")
                    continue
                
                # actually packaging the image
                image_data = payload["images"][image_id]
                ext = getImageExtension(image_data["url"], default="png")
                base64_image = image_data["data"].decode("utf-8")
                image_content_list.append({"type":"image_url",
                                                 "image_url" : {"url": f"data:image/{ext};base64,{base64_image}"}})

                messages.append({ "role": story_item["role"], "content": image_content_list})
            else:
                messages.append(story_item)
               

        return {"messages": messages}
        

        
    def tokenize(self, w):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {"prompt": w}
        response = requests.post(self.endpoint + "/v1/tokenize", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["tokens"]
        return []

    def detokenize(self, ts):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {"tokens": ts}
        response = requests.post(self.endpoint + "/v1/detokenize", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["content"]
        return []

    def health(self):
        # OpenAI API does not have a direct health check endpoint
        return "OpenAI API is assumed to be healthy."

    def timings(self, json=None) -> Optional[Timings]:
        if json is None:
            if (json := self._lastResult) is None:
                return None
            time = json["timings"]
            return Timings(
                prompt_n= time["prompt_n"],
                predicted_n = time["predicted_n"],
                prompt_ms=time["prompt_ms"],
                predicted_ms=time["predicted_ms"],
                predicted_per_token_ms=time["predicted_per_token_ms"],
                predicted_per_second=time["predicted_per_second"],
                # unfortunately openai don't reveal these FIXME: might be able to figure out truncated
                truncated=False,
                cached_n=None,
                original_timings=time
                )
        
