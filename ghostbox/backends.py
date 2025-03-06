import time, requests, threading
from abc import ABC, abstractmethod
from functools import *
from pydantic import BaseModel
from ghostbox.util import *
from ghostbox.definitions import *
from ghostbox.streaming import *

# this list is based on the llamacpp server. IMO most other backends are subsets of this.
# the sampler values have been adjusted to more sane default options (no more top_p)
sampling_parameters = {
    "temperature": SamplingParameterSpec(
        name="temperature",
        description="Adjust the randomness of the generated text.",
        default_value=0.8,
    ),
    "dynatemp_range": SamplingParameterSpec(
        name="dynatemp_range",
        description="Dynamic temperature range. The final temperature will be in the range of `[temperature - dynatemp_range; temperature + dynatemp_range]`",
        default_value=0.0,
    ),
    "dynatemp_exponent": SamplingParameterSpec(
        name="dynatemp_exponent",
        description="Dynamic temperature exponent.",
        default_value=1.0,
    ),
    "top_k": SamplingParameterSpec(
        name="top_k",
        description="Limit the next token selection to the K most probable tokens.",
        default_value=40,
    ),
    "top_p": SamplingParameterSpec(
        name="top_p",
        description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.",
        default_value=0.95,
    ),
    "min_p": SamplingParameterSpec(
        name="min_p",
        description="The minimum probability for a token to be considered, relative to the probability of the most likely token.",
        default_value=0.05,
    ),
    # this one is renamed from n_predict
    "max_length": SamplingParameterSpec(
        name="max_length",
        description="Set the maximum number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache.",
        default_value=-1,
    ),
    "n_indent": SamplingParameterSpec(
        name="n_indent",
        description="Specify the minimum line indentation for the generated text in number of whitespace characters. Useful for code completion tasks.",
        default_value=0,
    ),
    "n_keep": SamplingParameterSpec(
        name="n_keep",
        description="Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded. The number excludes the BOS token. By default, this value is set to `0`, meaning no tokens are kept. Use `-1` to retain all tokens from the prompt.",
        default_value=0,
    ),
    #    "stream": SamplingParameterSpec(
    #        name="stream",
    #        description="Allows receiving each predicted token in real-time instead of waiting for the completion to finish (uses a different response format). To enable this, set to `true`.",
    #        default_value=False
    #    ),
    "stop": SamplingParameterSpec(
        name="stop",
        description="Specify a JSON array of stopping strings. These words will not be included in the completion, so make sure to add them to the prompt for the next iteration.",
        default_value=[],
    ),
    "typical_p": SamplingParameterSpec(
        name="typical_p",
        description="Enable locally typical sampling with parameter p.",
        default_value=1.0,
    ),
    "repeat_penalty": SamplingParameterSpec(
        name="repeat_penalty",
        description="Control the repetition of token sequences in the generated text.",
        default_value=1.1,
    ),
    "repeat_last_n": SamplingParameterSpec(
        name="repeat_last_n",
        description="Last n tokens to consider for penalizing repetition.",
        default_value=64,
    ),
    "presence_penalty": SamplingParameterSpec(
        name="presence_penalty",
        description="Repeat alpha presence penalty.",
        default_value=0.0,
    ),
    "frequency_penalty": SamplingParameterSpec(
        name="frequency_penalty",
        description="Repeat alpha frequency penalty.",
        default_value=0.0,
    ),
    "dry_multiplier": SamplingParameterSpec(
        name="dry_multiplier",
        description="Set the DRY (Don't Repeat Yourself) repetition penalty multiplier.",
        default_value=0.8,
    ),
    "dry_base": SamplingParameterSpec(
        name="dry_base",
        description="Set the DRY repetition penalty base value.",
        default_value=1.75,
    ),
    "dry_allowed_length": SamplingParameterSpec(
        name="dry_allowed_length",
        description="Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length).",
        default_value=2,
    ),
    "dry_penalty_last_n": SamplingParameterSpec(
        name="dry_penalty_last_n",
        description="How many tokens to scan for repetitions.",
        default_value=-1,
    ),
    "dry_sequence_breakers": SamplingParameterSpec(
        name="dry_sequence_breakers",
        description="Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted.",
        default_value=["\n", ":", '"', "*"],
    ),
    "xtc_probability": SamplingParameterSpec(
        name="xtc_probability",
        description="Set the chance for token removal via XTC sampler.\nXTC means 'exclude top choices'. This sampler, when it triggers, removes all but one tokens above a given probability threshold. Recommended for creative tasks, as language tends to become less stereotypical, but can make a model less effective at structured output or intelligence-based tasks.\nSee original xtc PR by its inventor https://github.com/oobabooga/text-generation-webui/pull/6335",
        default_value=0.5,
    ),
    "xtc_threshold": SamplingParameterSpec(
        name="xtc_threshold",
        description="Set a minimum probability threshold for tokens to be removed via XTC sampler.\nXTC means 'exclude top choices'. This sampler, when it triggers, removes all but one tokens above a given probability threshold. Recommended for creative tasks, as language tends to become less stereotypical, but can make a model less effective at structured output or intelligence-based tasks.\nSee original xtc PR by its inventor https://github.com/oobabooga/text-generation-webui/pull/6335",
        default_value=0.1,
    ),
    "mirostat": SamplingParameterSpec(
        name="mirostat",
        description="Enable Mirostat sampling, controlling perplexity during text generation.",
        default_value=0,
    ),
    "mirostat_tau": SamplingParameterSpec(
        name="mirostat_tau",
        description="Set the Mirostat target entropy, parameter tau.",
        default_value=5.0,
    ),
    "mirostat_eta": SamplingParameterSpec(
        name="mirostat_eta",
        description="Set the Mirostat learning rate, parameter eta.",
        default_value=0.1,
    ),
    "grammar": SamplingParameterSpec(
        name="grammar",
        description="Set grammar for grammar-based sampling.",
        default_value=None,
    ),
    "json_schema": SamplingParameterSpec(
        name="json_schema",
        description='Set a JSON schema for grammar-based sampling (e.g. `{"items": {"type": "string"}, "minItems": 10, "maxItems": 100}` of a list of strings, or `{}` for any JSON). See [tests](../../tests/test-json-schema-to-grammar.cpp) for supported features.',
        default_value=None,
    ),
    "seed": SamplingParameterSpec(
        name="seed",
        description="Set the random number generator (RNG) seed.",
        default_value=-1,
    ),
    "ignore_eos": SamplingParameterSpec(
        name="ignore_eos",
        description="Ignore end of stream token and continue generating.",
        default_value=False,
    ),
    "logit_bias": SamplingParameterSpec(
        name="logit_bias",
        description='Modify the likelihood of a token appearing in the generated text completion. For example, use `"logit_bias": [[15043,1.0]]` to increase the likelihood of the token \'Hello\', or `"logit_bias": [[15043,-1.0]]` to decrease its likelihood. Setting the value to false, `"logit_bias": [[15043,false]]` ensures that the token `Hello` is never produced. The tokens can also be represented as strings, e.g. `[["Hello, World!",-0.5]]` will reduce the likelihood of all the individual tokens that represent the string `Hello, World!`, just like the `presence_penalty` does.',
        default_value=[],
    ),
    "n_probs": SamplingParameterSpec(
        name="n_probs",
        description="If greater than 0, the response also contains the probabilities of top N tokens for each generated token given the sampling settings. Note that for temperature < 0 the tokens are sampled greedily but token probabilities are still being calculated via a simple softmax of the logits without considering any other sampler settings.",
        default_value=0,
    ),
    "min_keep": SamplingParameterSpec(
        name="min_keep",
        description="If greater than 0, force samplers to return N possible tokens at minimum.",
        default_value=0,
    ),
    "t_max_predict_ms": SamplingParameterSpec(
        name="t_max_predict_ms",
        description="Set a time limit in milliseconds for the prediction (a.k.a. text-generation) phase. The timeout will trigger if the generation takes more than the specified time (measured since the first token was generated) and if a new-line character has already been generated. Useful for FIM applications.",
        default_value=0,
    ),
    # this doesn't even work in llama
    #    "image_data": SamplingParameterSpec(
    #        name="image_data",
    #        description="An array of objects to hold base64-encoded image `data` and its `id`s to be reference in `prompt`. You can determine the place of the image in the prompt as in the following: `USER:[img-12]Describe the image in detail.\nASSISTANT:`. In this case, `[img-12]` will be replaced by the embeddings of the image with id `12` in the following `image_data` array: `{..., \"image_data\": [{\"data\": \"<BASE64_STRING>\", \"id\": 12}]}`. Use `image_data` only with multimodal models, e.g., LLaVA.",
    #        default_value=[]
    #    ),
    "id_slot": SamplingParameterSpec(
        name="id_slot",
        description="Assign the completion task to an specific slot. If is -1 the task will be assigned to a Idle slot.",
        default_value=-1,
    ),
    "cache_prompt": SamplingParameterSpec(
        name="cache_prompt",
        description="Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests. Because (depending on the backend) the logits are **not** guaranteed to be bit-for-bit identical for different batch sizes (prompt processing vs. token generation) enabling this option can cause nondeterministic results.",
        default_value=True,
    ),
    "return_tokens": SamplingParameterSpec(
        name="return_tokens",
        description="Return the raw generated token ids in the `tokens` field. Otherwise `tokens` remains empty.",
        default_value=False,
    ),
    "samplers": SamplingParameterSpec(
        name="samplers",
        description="The order the samplers should be applied in. An array of strings representing sampler type names. If a sampler is not set, it will not be used. If a sampler is specified more than once, it will be applied multiple times.",
        default_value=["min_p", "xtc", "dry", "temperature"],
    ),
    "timings_per_token": SamplingParameterSpec(
        name="timings_per_token",
        description="Include prompt processing and text generation speed information in each response.",
        default_value=False,
    ),
    "post_sampling_probs": SamplingParameterSpec(
        name="post_sampling_probs",
        description="Returns the probabilities of top `n_probs` tokens after applying sampling chain.",
        default_value=None,
    ),
    "response_fields": SamplingParameterSpec(
        name="response_fields",
        description='A list of response fields, for example: `"response_fields": ["content", "generation_settings/n_predict"]`. If the specified field is missing, it will simply be omitted from the response without triggering an error. Note that fields with a slash will be unnested; for example, `generation_settings/n_predict` will move the field `n_predict` from the `generation_settings` object to the root of the response and give it a new name.',
        default_value=None,
    ),
    "lora": SamplingParameterSpec(
        name="lora",
        description='A list of LoRA adapters to be applied to this specific request. Each object in the list must contain `id` and `scale` fields. For example: `[{"id": 0, "scale": 0.5}, {"id": 1, "scale": 1.1}]`. If a LoRA adapter is not specified in the list, its scale will default to `0.0`. Please note that requests with different LoRA configurations will not be batched together, which may result in performance degradation.',
        default_value=[],
    ),
}

# this is for fast copy and send to backend
default_params = {hp.name: hp.default_value for hp in sampling_parameters.values()}

# some reference lists for convenience
supported_parameters = {
    p: sampling_parameters[p]
    for p in "temperature frequency_penalty presence_penalty max_length repeat_penalty top_p stop".split(
        " "
    )
}
sometimes_parameters = {
    p: sampling_parameters[p]
    for p in "cache_prompt xtc_probability dry_multiplier min_p mirostat mirostat_tau mirostat_eta samplers".split(
        " "
    )
}
sampling_parameter_tags = {
    p.name: ArgumentTag(
        name=p.name,
        type=ArgumentType.Plumbing,
        group=ArgumentGroup.SamplingParameters,
        is_option=True,
        default_value=p.default_value,
        help=p.description,
    )
    for p in sampling_parameters.values()
}
# some special ones
for p in supported_parameters.keys():
    sampling_parameter_tags[p].very_important = True

sampling_parameter_tags["temperature"].type = ArgumentType.Porcelain
sampling_parameter_tags["top_p"].type = ArgumentType.Porcelain


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


class AIBackend(ABC):
    """Abstract interface to a backend server, like llama.cpp or openai etc. Use the Program.make*Payload methods to make corresponding payloads. All backends must handle those dictionaries, but not all backends may support all features."""

    @abstractmethod
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.stream_done = threading.Event()
        self.last_error = ""
        self._last_request = {}
        self._last_result = {}
        self._config = {}

    def configure(self, config: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Configure backend specific options thourhg a key -> value dictionary. Returns the current config of a backend.
        To see a list of possible keys, consult the backend specific configure method's documentation, or call this method without arguments. Passing keys to backends that do not support them will have no effect but is otherwise safe.
        Note that this is *not* the way to set various backend server options, like cache_prompt or temperature etc. Those should go in the payload.
        """
        self._config |= config
        return self._config

    def getLastError(self):
        return self.last_error

    def getLastJSON(self) -> Dict:
        """Returns the last json result sent by the backend."""
        return self._last_result

    def getLastRequest(self) -> Dict:
        """Returns the last payload dictionary that was sent to the server."""
        return self._last_request

    def waitForStream(self):
        self.stream_done.wait()

    @abstractmethod
    def getName(self):
        """Returns the name of the backend. This can be compared to the --backend command line option."""
        pass

    @abstractmethod
    def getMaxContextLength(self):
        """Returns the default setting for maximum context length that is set serverside. Often, this should be the maximum context the model is trained on.
        This should return -1 if the backend is unable to determine the maximum context (some backends don't let you query this at all).
        """
        pass

    @abstractmethod
    def generate(self, payload):
        """Takes a payload dictionary similar to default_params. Returns a result object specific to the backend. Use handleResult to unpack its content."""
        pass

    @abstractmethod
    def handleGenerateResult(self, result) -> Optional[Dict]:
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
        The method may return none if there hasn't been a request yet to determine timings for and no json is provided.
        """
        pass

    @abstractmethod
    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        """Returns a dictionary of sampling_parameters that are supported by the model.
        The dictionary has the parameter names as keys. If a sampling_parameter is present in the dict, it is expected to be supported by the various generation methods.
        """
        pass


class LlamaCPPBackend(AIBackend):
    """Bindings for the formidable Llama.cpp based llama-server program."""

    def __init__(self, endpoint="http://localhost:8080"):
        super().__init__(endpoint)
        self._config |= {
            # this means we will use /chat/completions, which applies the jinja templates etc. This is most often what we want.
            # if this option is false, the generate methods will use the /completions endpoint, which requires us to apply our own templates, which is great for experimentation.
            "llamacpp_use_chat_completion_endpoint": True
        }

    def getName(self):
        return LLMBackend.llamacpp.name

    def getMaxContextLength(self):
        return -1

    def generate(self, payload):
        super().generate(payload)
        # adjust slightly for our renames
        llama_payload = payload | {"n_predict": payload["max_length"]}

        if self._config["llamacpp_use_chat_completion_endpoint"]:
            endpoint_suffix = "/chat/completions"
            # /chat/completions expects a more OAI like payload
            llama_payload = OpenAIBackend.dataFromPayload(llama_payload)
        else:
            endpoint_suffix = "/completion"
            
        self._last_request = llama_payload
        return requests.post(self.endpoint + endpoint_suffix, json=llama_payload)

    def handleGenerateResult(self, result):
        if result.status_code != 200:
            self.last_error = "HTTP request with status code " + str(result.status_code)
            return None
        self._lastResult = result.json()

        import json
        print(json.dumps(result.json(), indent=4))
        if self._config["llamacpp_use_chat_completion_endpoint"]:
            # this one wants more oai like results
            return OpenAIBackend.handleGenerateResultOpenAI(result.json())

        # handling the /completion endpoint
        if (payload := result.json()["content"]) is not None:
            return payload
        if (payload := result.json().get("tool_calls", None)) is not None:
            return payload
        return None

    def _makeLlamaCallback(self, callback):
        def f(d):
            import json
            if d["stop"]:
                self._last_result = d                            
            callback(d["content"])
        return f

    def generateStreaming(self, payload, callback=lambda w: print(w)):
        self.stream_done.clear()
        llama_payload = payload | {"n_predict": payload["max_length"], "stream":True}

        def one_line_lambdas_for_python(r):
            # thanks guido
            self._last_result = r
            
        if self._config["llamacpp_use_chat_completion_endpoint"]:
            endpoint_suffix = "/chat/completions"
            # /chat/completions expects a more OAI like payload
            llama_payload |= OpenAIBackend.dataFromPayload(llama_payload)
            final_callback = OpenAIBackend.makeOpenAICallback(callback, last_result_callback=one_line_lambdas_for_python)
        else:
            endpoint_suffix = "/completion"
            final_callback =             self._makeLlamaCallback(callback)

        self._last_request = llama_payload

        r = streamPrompt(
            final_callback,
            self.stream_done,
            self.endpoint + endpoint_suffix,
            llama_payload,
        )
        if r.status_code != 200:
            self.last_error = "streaming HTTP request with status code " + str(
                r.status_code
            )
            self.stream_done.set()
            return True
        return False

    def tokenize(self, w):
        r = requests.post(self.endpoint + "/tokenize", json={"content": w})
        if r.status_code == 200:
            return r.json()["tokens"]
        return []

    def detokenize(self, ts):
        r = requests.post(self.endpoint + "/detokenize", json={"tokens": ts})
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
            if (json := self._last_result) is None:
                return None
            time = json["timings"]
            return Timings(
                prompt_n=time["prompt_n"],
                predicted_n=time["predicted_n"],
                prompt_ms=time["prompt_ms"],
                predicted_ms=time["predicted_ms"],
                predicted_per_token_ms=time["predicted_per_token_ms"],
                predicted_per_second=time["predicted_per_second"],
                truncated=json["truncated"],
                cached_n=json["tokens_cached"],
                original_timings=time,
            )

    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        # llamacpp params are the default
        return sampling_parameters


class OpenAILegacyBackend(AIBackend):
    """Backend for the official OpenAI API. The legacy version routes to /v1/completions, instead of the regular /v1/chat/completion."""

    def __init__(self, api_key, endpoint="https://api.openai.com"):
        super().__init__(endpoint)
        self.api_key = api_key

    def getName(self):
        return LLMBackend.openai.name

    def getMaxContextLength(self):
        return -1

    def generate(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = payload | {"stream": False}
        self._last_request = data
        response = requests.post(
            self.endpoint + "/v1/completions", headers=headers, json=data
        )
        if response.status_code != 200:
            self.last_error = (
                f"HTTP request with status code {response.status_code}: {response.text}"
            )
            return None
        self._lastResult = response.json()
        return response.json()

    def handleGenerateResult(self, result):
        if not result:
            return None

        if (payload := result["choices"][0]["message"]["content"]) is not None:
            return payload
        if (payload := result["choices"][0]["message"]["tool_calls"]) is not None:
            return payload
        return None

    def generateStreaming(self, payload, callback=lambda w: print(w)):
        self.stream_done.clear()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = payload | {"stream": True, "stream_options": {"include_usage": True}}
        self._last_request = data

        def openaiCallback(d):
            callback(d["choices"][0]["text"])

        response = streamPrompt(
            openaiCallback,
            self.stream_done,
            self.endpoint + "/v1/completions",
            json=data,
            headers=headers,
        )
        if response.status_code != 200:
            self.last_error = (
                f"HTTP request with status code {response.status_code}: {response.text}"
            )
            self.stream_done.set()
            return True
        return False

    def tokenize(self, w):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"prompt": w}
        response = requests.post(
            self.endpoint + "/v1/tokenize", headers=headers, json=data
        )
        if response.status_code == 200:
            return response.json()["tokens"]
        return []

    def detokenize(self, ts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"tokens": ts}
        response = requests.post(
            self.endpoint + "/v1/detokenize", headers=headers, json=data
        )
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
        return {
            hp.name: hp for hp in sampling_parameters.values() if hp.name in supported
        }


class OpenAIBackend(AIBackend):
    """Backend for the official OpenAI API. This is used for the company of Altman et al, but also serves as a general purpose API suported by various backends, including llama.cpp, llama-box, and many others."""

    def __init__(self, api_key, endpoint="https://api.openai.com"):
        super().__init__(endpoint)
        self.api_key = api_key
        self._memoized_params = None

    def getName(self):
        return LLMBackend.openai.name

    def getMaxContextLength(self):
        return -1

    def generate(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = payload | {"max_tokens": payload["max_length"], "stream": False}
        # the /V1/chat/completions endpoint expects structured data of user/assistant pairs
        data |= self.dataFromPayload(payload)
        self._last_request = data
        response = requests.post(
            self.endpoint + "/v1/chat/completions", headers=headers, json=data
        )
        if response.status_code != 200:
            self.last_error = (
                f"HTTP request with status code {response.status_code}: {response.text}"
            )
            return None
        self._lastResult = response.json()
        return response.json()


    def handleGenerateResult(self, result):
        # this is just so that others can use the openai specific handling, which is kind of an industry standard
        return self.handleGenerateResultOpenAI(result)
    
    @staticmethod
    def handleGenerateResultOpenAI(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not result:
            return None

        if (payload := result["choices"][0]["message"]["content"]) is not None:
            return payload
        if (payload := result["choices"][0]["message"]["tool_calls"]) is not None:
            return payload
        return None

    @staticmethod
    def makeOpenAICallback(callback, last_result_callback=lambda x: x):
        def openAICallback(d):
            last_result_callback(d)
            choice = d["choices"][0]
            maybeChunk = choice["delta"].get("content", None)
            if maybeChunk is not None:
                callback(maybeChunk)
                
        return openAICallback
    
    def generateStreaming(self, payload, callback=lambda w: print(w)):
        self.stream_done.clear()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = payload | {"stream": True, "stream_options": {"include_usage": True}}
        # the /V1/chat/completions endpoint expects structured data of user/assistant pairs
        data |= self.dataFromPayload(payload)
        self._last_request = data
        def one_line_lambdas_for_python(r):
            self._last_result = r
        
        response = streamPrompt(
            self.makeOpenAICallback(callback, last_result_callback=one_line_lambdas_for_python),
            self.stream_done,
            self.endpoint + "/v1/chat/completions",
            json=data,
            headers=headers,
        )
        if response.status_code != 200:
            self.last_error = (
                f"HTTP request with status code {response.status_code}: {response.text}"
            )
            self.stream_done.set()
            return True
        return False

    @staticmethod
    def dataFromPayload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Take a payload dictionary from Plumbing and return dictionary with elements specific to the chat/completions endpoint.
        This expects payload to include the messages key, with various dictionaries in it, unlike other backends.
        """
        messages = [{"role": "system", "content": payload["system"]}]
        # story is list of dicts with role and content keys
        # we go through story one by one, mostly because of images
        for story_item in payload["story"]:
            if "image_id" in story_item:
                # images is more complicated, see https://platform.openai.com/docs/guides/vision
                # API wants the content field of an image message to be a list of dicts, not a string
                # the dicts have the type field, which determines wether its a user msg (text) or image (image-url)
                image_id = story_item["image_id"]
                image_content_list = []
                image_content_list.append(
                    {"type": "text", "content": story_item["content"]}
                )
                if "images" not in payload or image_id not in payload["images"]:
                    printerr("warning: image with id " + str(image_id) + " not found.")
                    continue

                # actually packaging the image
                image_data = payload["images"][image_id]
                ext = getImageExtension(image_data["url"], default="png")
                base64_image = image_data["data"].decode("utf-8")
                image_content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{base64_image}"},
                    }
                )

                messages.append(
                    {"role": story_item["role"], "content": image_content_list}
                )
            else:
                messages.append(story_item)

        return {"messages": messages}

    def tokenize(self, w):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"prompt": w}
        response = requests.post(
            self.endpoint + "/v1/tokenize", headers=headers, json=data
        )
        if response.status_code == 200:
            return response.json()["tokens"]
        return []

    def detokenize(self, ts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"tokens": ts}
        response = requests.post(
            self.endpoint + "/v1/detokenize", headers=headers, json=data
        )
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

            if "timings" not in json:
                return None

            time = json["timings"]
            return Timings(
                prompt_n=time["prompt_n"],
                predicted_n=time["predicted_n"],
                prompt_ms=time["prompt_ms"],
                predicted_ms=time["predicted_ms"],
                predicted_per_token_ms=time["predicted_per_token_ms"],
                predicted_per_second=time["predicted_per_second"],
                # unfortunately openai don't reveal these FIXME: might be able to figure out truncated
                truncated=False,
                cached_n=None,
                original_timings=time,
            )

    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        # I don't like doing this everytime
        if self._memoized_params is not None:
            return self._memoized_params

        # this is tricky because it really depends on the actual backend.
        # the openai class really is not specific enough for this
        supported = supported_parameters.keys()
        sometimes = sometimes_parameters.keys()
        d = {hp.name: hp for hp in sampling_parameters.values() if hp.name in supported}
        for param in sometimes:
            sp = sampling_parameters[param]
            d[param] = SamplingParameterSpec(
                name=sp.name,
                default_value=sp.default_value,
                description=sp.description
                + "\nNote: May not be supported in this backend. For full support, try out llama.cpp https://github.com/ggml-org/llama.cpp",
            )

            self._memoized_params = d
            return d
