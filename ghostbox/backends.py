import time, requests, threading
from abc import ABC, abstractmethod
from functools import *
from pydantic import BaseModel
from ghostbox.util import *
from ghostbox.definitions import *
from ghostbox.streaming import *

# this list is based on the llamacpp server. IMO most other backends are subsets of this.
_sampling_parameter_dict = [
    {"name": "temperature", "description": "Adjust the randomness of the generated text.", "default_value": 0.8},
    {"name": "dynatemp_range", "description": "Dynamic temperature range. The final temperature will be in the range of `[temperature - dynatemp_range; temperature + dynatemp_range]`", "default_value": 0.0},
    {"name": "dynatemp_exponent", "description": "Dynamic temperature exponent.", "default_value": 1.0},
    {"name": "top_k", "description": "Limit the next token selection to the K most probable tokens.", "default_value": 40},
    {"name": "top_p", "description": "Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.", "default_value": 0.95},
    {"name": "min_p", "description": "The minimum probability for a token to be considered, relative to the probability of the most likely token.", "default_value": 0.05},
# we rename this one     
#    {"name": "n_predict", "description": "Set the maximum number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache.", "default_value": -1},
{"name": "max_length", "description": "Set the maximum number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache.", "default_value": -1},    
    {"name": "n_indent", "description": "Specify the minimum line indentation for the generated text in number of whitespace characters. Useful for code completion tasks.", "default_value": 0},
    {"name": "n_keep", "description": "Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded. The number excludes the BOS token.", "default_value": 0},
#    {"name": "stream", "description": "Allows receiving each predicted token in real-time instead of waiting for the completion to finish (uses a different response format). To enable this, set to `true`.", "default_value": False},
    {"name": "stop", "description": "Specify a JSON array of stopping strings. These words will not be included in the completion, so make sure to add them to the prompt for the next iteration.", "default_value": []},
    {"name": "typical_p", "description": "Enable locally typical sampling with parameter p.", "default_value": 1.0},
    {"name": "repeat_penalty", "description": "Control the repetition of token sequences in the generated text.", "default_value": 1.1},
    {"name": "repeat_last_n", "description": "Last n tokens to consider for penalizing repetition.", "default_value": 64},
    {"name": "presence_penalty", "description": "Repeat alpha presence penalty.", "default_value": 0.0},
    {"name": "frequency_penalty", "description": "Repeat alpha frequency penalty.", "default_value": 0.0},
    {"name": "dry_multiplier", "description": "Set the DRY (Don't Repeat Yourself) repetition penalty multiplier.", "default_value": 0.0},
    {"name": "dry_base", "description": "Set the DRY repetition penalty base value.", "default_value": 1.75},
    {"name": "dry_allowed_length", "description": "Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length).", "default_value": 2},
    {"name": "dry_penalty_last_n", "description": "How many tokens to scan for repetitions.", "default_value": -1},
    {"name": "dry_sequence_breakers", "description": "Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted.", "default_value": ['\n', ':', '"', '*']},
    {"name": "xtc_probability", "description": "Set the chance for token removal via XTC sampler.", "default_value": 0.0},
    {"name": "xtc_threshold", "description": "Set a minimum probability threshold for tokens to be removed via XTC sampler.", "default_value": 0.1},
    {"name": "mirostat", "description": "Enable Mirostat sampling, controlling perplexity during text generation.", "default_value": 0},
    {"name": "mirostat_tau", "description": "Set the Mirostat target entropy, parameter tau.", "default_value": 5.0},
    {"name": "mirostat_eta", "description": "Set the Mirostat learning rate, parameter eta.", "default_value": 0.1},
    {"name": "grammar", "description": "Set grammar for grammar-based sampling.", "default_value": None},
    {"name": "json_schema", "description": "Set a JSON schema for grammar-based sampling (e.g. `{\"items\": {\"type\": \"string\"}, \"minItems\": 10, \"maxItems\": 100}` of a list of strings, or `{}` for any JSON). See [tests](../../tests/test-json-schema-to-grammar.cpp) for supported features.", "default_value": None},
    {"name": "seed", "description": "Set the random number generator (RNG) seed.", "default_value": -1},
    {"name": "ignore_eos", "description": "Ignore end of stream token and continue generating.", "default_value": False},
    {"name": "logit_bias", "description": "Modify the likelihood of a token appearing in the generated text completion.", "default_value": []},
    {"name": "n_probs", "description": "If greater than 0, the response also contains the probabilities of top N tokens for each generated token given the sampling settings.", "default_value": 0},
    {"name": "min_keep", "description": "If greater than 0, force samplers to return N possible tokens at minimum.", "default_value": 0},
    {"name": "t_max_predict_ms", "description": "Set a time limit in milliseconds for the prediction (a.k.a. text-generation) phase.", "default_value": 0},
    {"name": "image_data", "description": "An array of objects to hold base64-encoded image `data` and its `id`s to be reference in `prompt`.", "default_value": []},
    {"name": "id_slot", "description": "Assign the completion task to an specific slot. If is -1 the task will be assigned to a Idle slot.", "default_value": -1},
    {"name": "cache_prompt", "description": "Re-use KV cache from a previous request if possible.", "default_value": True},
    {"name": "return_tokens", "description": "Return the raw generated token ids in the `tokens` field.", "default_value": False},
    {"name": "samplers", "description": "The order the samplers should be applied in.", "default_value": ["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]},
    {"name": "timings_per_token", "description": "Include prompt processing and text generation speed information in each response.", "default_value": False},
    {"name": "post_sampling_probs", "description": "Returns the probabilities of top `n_probs` tokens after applying sampling chain.", "default_value": None},
    {"name": "response_fields", "description": "A list of response fields, for example: `\"response_fields\": [\"content\", \"generation_settings/n_predict\"]`.", "default_value": None},
    {"name": "lora", "description": "A list of LoRA adapters to be applied to this specific request.", "default_value": []},
]

## Construct the dictionary of Pydantic objects
sampling_parameters = {hp["name"]: SamplingParameterSpec(**hp) for hp in _sampling_parameter_dict}
# this is for fast copy and send to backend
default_params = {hp.name: hp.default_value for hp in sampling_parameters.values()}

# some reference lists for convenience
supported_parameters = {p:sampling_parameters[p] for p in "temperature frequency_penalty presence_penalty max_length repeat_penalty top_p stop".split(" ")}
sometimes_parameters = {p:sampling_parameters[p] for p in "xtc_probability dry_multiplier min_p mirostat mirostat_tau mirostat_eta".split(" ")}
sampling_parameter_tags = {p.name : ArgumentTag(name=p.name,type=ArgumentType.Plumbing, group=ArgumentGroup.SamplingParameters, is_option=True, default_value=p.default_value,help=p.description) for p in sampling_parameters.values()}
# some special ones
for p in supported_parameters.keys():
    sampling_parameter_tags[p].very_important = True

sampling_parameter_tags["temperature"].type = ArgumentType.Porcelain
sampling_parameter_tags["top_p"].type = ArgumentType.Porcelain

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

    @abstractmethod

    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        """Returns a dictionary of sampling_parameters that are supported by the model.
        The dictionary has the parameter names as keys. If a sampling_parameter is present in the dict, it is expected to be supported by the various generation methods."""
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
        # adjust slightly for our renames
        llama_paylod = payload | {"n_predict":payload["max_length"]}
        return requests.post(self.endpoint + "/completion", json=llama_payload)

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
        llama_paylod = payload | {"n_predict":payload["max_length"]}        
        r = streamPrompt(self._makeLlamaCallback(callback), self.stream_done, self.endpoint + "/completion", llama_payload)
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

    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        # llamacpp params are the default
        return sampling_parameters
        
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
            "max_tokens": payload.get("max_length", 150),
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
            "max_tokens": payload.get("max_length", 150),
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

    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        # restricted set
        # i just can't be bothered to test this
        supported = supported_parameters.keys()
        return {hp.name:hp for hp in sampling_parameters.values()
                if hp.name in supported}
        
class OpenAIBackend(AIBackend):
    """Backend for the official OpenAI API. This is used for the company of Altman et al, but also serves as a general purpose API suported by various backends, including llama.cpp, llama-box, and many others."""
    
    def __init__(self, api_key, endpoint="https://api.openai.com"):
        super().__init__(endpoint)
        self.api_key = api_key
        self._lastResult = None
        self._memoized_params = None

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
            "max_tokens": payload.get("max_length", 150), # FIXME: this is changed in the API for o1 and up
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
            "max_tokens": payload.get("max_length", 150),
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

    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        # I don't like doing this everytime
        if self._memoized_params is not None:
            return self._memoized_params
        
        # this is tricky because it really depends on the actual backend.
        # the openai class really is not specific enough for this
        supported = supported_parameters.keys()
        sometimes = sometimes_parameters.keys()
        d = {hp.name:hp for hp in sampling_parameters.values()
                if hp.name in supported}
        for param in sometimes:
            sp = sampling_parameters[param]
            d[param] = SamplingParameterSpec(name=sp.name, default_value=sp.default_value, description=sp.description + "\nNote: May not be supported in this backend. For full support, try out llama.cpp https://github.com/ggml-org/llama.cpp") 

            self._memoized_params = d
            return d
        
