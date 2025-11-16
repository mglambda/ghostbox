import time, requests, threading, json
from abc import ABC, abstractmethod
from functools import *
from pydantic import BaseModel
from ghostbox.util import *
from ghostbox.definitions import *
from ghostbox.streaming import *
import traceback # Added for detailed error logging in GoogleBackend
import base64 # Added for image handling in GoogleBackend
from google.genai.types import Content, Part # Added for Google tokenize helper

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
    "grammar_lazy": SamplingParameterSpec(
        name="grammar_lazy",
        description="This parameter controls whether the grammar (specified by `grammar` or `json_schema`) is applied strictly from the beginning of generation, or if its activation is deferred until a specific trigger is encountered.",
        default_value=False,
    ),
    "grammar_triggers": SamplingParameterSpec(
        name="grammar_triggers",
        description="This parameter defines the conditions under which a lazy grammar (when `grammar_lazy` is `true`) should become active. Each object in the array represents a single trigger.",
        default_value=[],
    ),
    "preserved_tokens": SamplingParameterSpec(
        name="preserved_tokens",
        description="A list of token pieces to be preserved during sampling and grammar processing.",
        default_value=[],
    ),
    "enable_thinking": SamplingParameterSpec(
        name="enable_thinking",
        description="Turn on reasoning for thinking models, disable it otherwise (if possible).",
        default_value=True,
    ),            
    "chat_template_kwargs": SamplingParameterSpec(
        name="chat_template_kwargs",
        description="This parameter allows you to pass arbitrary key-value arguments directly to the Jinja chat template used for prompt formatting. These arguments can be used within the Jinja template to control conditional logic, insert dynamic content, or modify the template's behavior.",
        default_value={"enable_thinking": True},
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
    # new - only  got this from the git logs
    "add_generation_prompt": SamplingParameterSpec(
        name="add_generation_prompt",
        description="Include the prompt used to generate in the result.",
        default_value=True,
    ),
}
### end of big list

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
    for p in "cache_prompt xtc_probability dry_multiplier min_p mirostat mirostat_tau mirostat_eta samplers grammar_lazy grammar_triggers preserved_tokens chat_template_kwargs enable_thinking".split(
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


# These don't fit anywhere else and don't really need documentation
special_parameters = {"response_format": {"type": "text"},
                      "llamacpp_thinking_json_fix": True,
                      "model": ""}



class AIBackend(ABC):
    """Abstract interface to a backend server, like llama.cpp or openai etc. Use the Program.make*Payload methods to make corresponding payloads. All backends must handle those dictionaries, but not all backends may support all features."""

    @abstractmethod
    def __init__(self, endpoint: str, logger: Optional[Callable[[str], None]] = None):
        self.endpoint = endpoint
        self.stream_done = threading.Event()
        self.last_error = ""
        self._last_request = {}
        self._last_result = {}
        self._config = {}
        self.logger = logger

    def log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger(f"[{self.getName()}] " + msg)

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
    def timings(self, result_json=None) -> Optional[Timings]:
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

    def get_models(self) -> List[ModelStats]:
        """List the models supported by this backend.
        Mostly used with cloud providers. Many backends will return an empty list, which indicates that there isn't any model information available."""
        return []

class DummyBackend(AIBackend):

    def __init__(self, endpoint: str ="http://localhost:8080", **kwargs):
        super().__init__(endpoint, **kwargs)

    def getName(self) -> str:
        return "dummy"

    def getMaxContextLength(self) -> int:
        return -1

    def generate(self, payload) -> None:
        return None

    def handleGenerateResult(self, result):
        return None

    def generateStreaming(self, payload) -> None:
        return None
    def timings(self) -> Optional[Timings]:
        return None
    
    
    def health(self) -> str:
        return "This is a dummy backend. It's fine."

    def tokenize(self, w) -> List[int]:
        return []

    def detokenize(self, tokens) -> str:
        return ""
    
    def sampling_parameters(self):
        return {}
    
class LlamaCPPBackend(AIBackend):
    """Bindings for the formidable Llama.cpp based llama-server program."""

    def __init__(self, endpoint: str ="http://localhost:8080", **kwargs):
        super().__init__(endpoint, **kwargs)
        self._config |= {
            # this means we will use /chat/completions, which applies the jinja templates etc. This is most often what we want.
            # if this option is false, the generate methods will use the /completions endpoint, which requires us to apply our own templates, which is great for experimentation.
            "llamacpp_use_chat_completion_endpoint": True
        }
        self.log(f"Initialized llama.cpp backend with config : {json.dumps(self._config)}")
        
    def getName(self):
        return LLMBackend.llamacpp.name

    def getMaxContextLength(self):
        return -1


    @staticmethod
    def _alter_system_msg(llama_payload: Dict[str, Any], alter_function: Callable[[str], str]) -> None:
        """Apply the alter_function to the system prompt.
        alter_function must be reentrant."""
        
        try:
            llama_payload["system"] = alter_function(llama_payload["system"])
        except KeyError:
            # no biggie
            pass

        try:
            for message in llama_payload["messages"]:
                if message["role"] == "system":
                    message["content"] = alter_function(message["content"])
        except KeyError:
            pass
        
    @staticmethod
    def _fix_thinking_json(llama_payload: Dict[str, Any]) -> None:
        """Destructively modifies the llama_payload dict to disable structured output and contain the json schema in the prompt instead.
        Also sets the grammar to an experimental thinking json grammar, which allows the model to think and then only accepts general json syntax, i.e. general and not specifically the schema.
        """        

        response_format = llama_payload["response_format"]
        if not(isinstance(response_format, dict)):
            return

        if response_format.get("type", None) != "json_object":
            return

        if (schema := response_format.get("schema", None)) == None:
            return

        schema_str = json.dumps(schema)
        def append_schema(system_msg: str):
            return system_msg + f"""
When responding to the user, think step by step before giving a response. Your final response should adhere to the following JSON schema: The JSON you output must adhere to the following schema:

```json
        {schema_str}
```        
"""

        LlamaCPPBackend._alter_system_msg(llama_payload, append_schema)
        # keep llamacpp from applying grammar
        llama_payload["response_format"] = "text"
        # enable our own grammar
        llama_payload["grammar"] = getJSONThinkingGrammar()
        
        try:
            llama_payload["chat_template_kwargs"] |= {"enable_thinking":True}
        except KeyError:
            pass
        
        
    def generate(self, payload):
        # adjust slightly for our renames
        llama_payload = payload | {"n_predict": payload["max_length"]}

        if self._config["llamacpp_use_chat_completion_endpoint"]:
            endpoint_suffix = "/chat/completions"
        else:
            endpoint_suffix = "/completion"
            if "tools" in payload:
                printerr(
                    "warning: Tool use with a custom prompt_format and using llama.cpp backend is currently experimental. Set your prompt_format to 'auto' or use the generic backend for a stable experience."
                )

        if "tools" in llama_payload:
            # FIXME: this is because using tools seems to invalidate the cache in llamacpp. probably because they are putting tool instructions in the system prompt. this is an attempt to fix or at least mitigate that.
            # i.e. we can just cache the inevitable streaming, non-tool generation that follows tool use.
            # however this will still suck for multi-turn tool use
            llama_payload |= {"cache_prompt": False}

        # slight adjustment. I'm still unclear wether llama server supports this directly
        if "enable_thinking" in llama_payload:
            old = llama_payload.get("chat_template_kwargs", {})
            llama_payload["chat_template_kwargs"] = old | {"enable_thinking": llama_payload["enable_thinking"]}
            
        if llama_payload["llamacpp_thinking_json_fix"] and llama_payload["enable_thinking"] and isinstance(llama_payload["response_format"], dict):
            self.log(f"Applying thinking json fix.")
            self._fix_thinking_json(llama_payload)
            
        self._last_request = llama_payload
        final_endpoint = self.endpoint + endpoint_suffix
        self.log(f"generate to {final_endpoint}")
        self.log(f"Payload: {json.dumps(llama_payload, indent=4)}")        
        return requests.post(final_endpoint, json=llama_payload)

    def handleGenerateResult(self, result):
        if result.status_code != 200:
            self.last_error = "HTTP request with status code " + str(result.status_code)
            return None
        self._last_result = result.json()


        if self._config["llamacpp_use_chat_completion_endpoint"]:
            # this one wants more oai like results
            # FIXME: as of september 2025 this returns a weirdly structured message. the fixme is more a reminder to investigate
            llama_msg = OpenAIBackend.handleGenerateResultOpenAI(result.json())
            return llama_msg

        # handling the /completion endpoint
        if (payload := result.json()["content"]) is not None:
            return payload
        if (payload := result.json().get("tool_calls", None)) is not None:
            return payload
        return None

    def _makeLlamaCallback(self, callback):
        def f(d):
            if d["stop"]:
                self._last_result = d
            callback(d["content"])

        return f

    def generateStreaming(self, payload, callback=lambda w: print(w)):
        self.stream_done.clear()
        llama_payload = payload | {"n_predict": payload["max_length"], "stream": True}

        def one_line_lambdas_for_python(r):
            # thanks guido
            self._last_result = r

        if self._config["llamacpp_use_chat_completion_endpoint"]:
            endpoint_suffix = "/chat/completions"

            final_callback = OpenAIBackend.makeOpenAICallback(
                callback, last_result_callback=one_line_lambdas_for_python
            )
        else:
            endpoint_suffix = "/completion"
            final_callback = self._makeLlamaCallback(callback)

        if llama_payload["llamacpp_thinking_json_fix"] and llama_payload["enable_thinking"]:
            self.log(f"warning: llamacpp_thinking_json_fix is not supported with stream = True. Defaulting to using a grammar, which may disable reasoning.")
        self._last_request = llama_payload

        final_endpoint = self.endpoint + endpoint_suffix
        self.log(f"generateStreaming to {final_endpoint}")
        self.log(f"Payload: {json.dumps(llama_payload, indent=4)}")        
        r = streamPrompt(
            final_callback,
            self.stream_done,
            final_endpoint,
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
        self.log(f"tokenize {len(w)} characters.")
        r = requests.post(self.endpoint + "/tokenize", json={"content": w})
        if r.status_code == 200:
            return r.json()["tokens"]
        return []

    def detokenize(self, ts):
        self.log("detokenize with {len(ts)} tokens.")
        r = requests.post(self.endpoint + "/detokenize", json={"tokens": ts})
        if r.status_code == 200:
            return r.json()["content"]
        return []

    def health(self):
        r = requests.get(self.endpoint + "/health")
        if r.status_code != 200:
            return "error " + str(r.status_code)
        return r.json()["status"]

    def timings(self, result_json=None) -> Optional[Timings]:
        if result_json is None:
            if (json := self._last_result) is None:
                return None
        else:
            json = result_json

        if "timings" not in json:
            printerr("warning: Got weird server result: " + str(json))
            return

        time = json["timings"]
        # these are llama specific fields which aren't always available on the OAI endpoints
        truncated, cached_n = json.get("truncated", None), json.get(
            "tokens_cached", None
        )
        if (verbose := json.get("__verbose", None)) is not None:
            truncated, cached_n = verbose["truncated"], verbose["tokens_cached"]

            return Timings(
                prompt_n=time["prompt_n"],
                predicted_n=time["predicted_n"],
                prompt_ms=time["prompt_ms"],
                predicted_ms=time["predicted_ms"],
                predicted_per_token_ms=time["predicted_per_token_ms"],
                predicted_per_second=time["predicted_per_second"],
                truncated=truncated,
                cached_n=cached_n,
                original_timings=time,
            )

    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        # llamacpp params are the default
        return sampling_parameters

    def props(self) -> Dict[str, Any]:
        """Llama.cpp specific /props endpoint, giving server and model properties."""
        response = requests.get(self.endpoint + "/props")
        if response.status_code != 200:
            self.log(f"Couldn't get /props. Reason: {response.text}, text: {response.text}")
            return {}

        try:
            return response.json()
        except Exception as e:
            self.log(f"Couldn't parse json from /props endpoint. Reason: {e}")
        return {}

    def has_thinking_model(self) -> Optional[bool]:
        """Llama.cpp only. Queries the jinja template to determine if the model that llama is running suports thinking.
        Returns true if thinking is enabled, false if not, and none if it can't be determined."""
        
        props = self.props()
        if not props or "chat_template" not in props:
            return None

        template_str = props["chat_template"]
        if "enable_thinking" in template_str:
            return True
        return False
class OpenAILegacyBackend(AIBackend):
    """Backend for the official OpenAI API. The legacy version routes to /v1/completions, instead of the regular /v1/chat/completion."""

    def __init__(self, api_key: str, endpoint:str="https://api.openai.com", **kwargs):
        super().__init__(endpoint, **kwargs)
        self.api_key = api_key
        self.log(f"Initialized legacy OpenAI backend. This routes to /v1/completion and will not apply the chat template. Using config : {json.dumps(self._config)}")
        
    def getName(self):
        return LLMBackend.legacy.name

    def getMaxContextLength(self):
        return -1

    def generate(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = payload | {"max_tokens": payload["max_length"], "stream": False}
        self._last_request = data
        final_endpoint = self.endpoint + "/v1/completions"
        self.log(f"generate to {final_endpoint}")
        self.log(f"Payload: {json.dumps(data, indent=4)}")        
        response = requests.post(
            final_endpoint, headers=headers, json=data
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

        data = payload | {"max_tokens": payload["max_length"], "stream": True, "stream_options": {"include_usage": True}}
        self._last_request = data

        def openaiCallback(d):
            callback(d["choices"][0]["text"])
            self._last_result = d            


        final_endpoint = self.endpoint + "/v1/completions"
        self.log(f"generateStreaming to {final_endpoint}.")
        self.log(f"Payload: {json.dumps(data, indent=4)}")
        response = streamPrompt(
            openaiCallback,
            self.stream_done,
            final_endpoint,
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
        self.log(f"tokenize with {len(w)} characters.")
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
        self.log(f"detokenize with {len(ts)} tokens.")
        response = requests.post(
            self.endpoint + "/v1/detokenize", headers=headers, json=data
        )
        if response.status_code == 200:
            return response.json()["content"]
        return []

    def health(self):
        # OpenAI API does not have a direct health check endpoint
        return "OpenAI API is assumed to be healthy."

    def timings(self, result_json=None) -> Optional[Timings]:
        return OpenAIBackend.timings(self, result_json)


    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        # restricted set
        # i just can't be bothered to test this
        supported = supported_parameters.keys()
        return {
            hp.name: hp for hp in sampling_parameters.values() if hp.name in supported
        }


class OpenAIBackend(AIBackend):
    """Backend for the official OpenAI API. This is used for the company of Altman et al, but also serves as a general purpose API suported by various backends, including llama.cpp, llama-box, and many others."""

    def __init__(self, api_key: str, endpoint:str="https://api.openai.com", **kwargs):
        super().__init__(endpoint, **kwargs)
        self.api_key = api_key
        self._memoized_params = None
        api_str = "" if not(api_key) else " with api key " + api_key[:4] + ("x" * len(api_key[4:]))
        self.log(f"Initialized OpenAI compatible backend {api_str}. Routing to {endpoint}. Config is {self._config}")
                                                                            
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

        if "tools" in data:
            # see the llamacpp generate method fixme
            # this has no effect on the official OAI api anyway
            data |= {"cache_prompt": False}

        self._last_request = data
        final_endpoint = self.endpoint + "/v1/chat/completions"
        self.log(f"generate to {final_endpoint}.")
        self.log(f"Payload: {json.dumps(data, indent=4)}")        
        response = requests.post(
            final_endpoint, headers=headers, json=data
        )
        if response.status_code != 200:
            self.last_error = (
                f"HTTP request with status code {response.status_code}: {response.text}"
            )
            return None
        self._last_result = response.json()
        return response.json()

    def handleGenerateResult(self, result):
        # this is just so that others can use the openai specific handling, which is kind of an industry standard
        return self.handleGenerateResultOpenAI(result)

    @staticmethod
    def handleGenerateResultOpenAI(result: Dict[str, Any]) -> Any:
        # used to be Optional[Dict[str, Any]]:
        # now it's Optional[str|Dict]
        # it's not completely terrible, since it makes sense - either return the text that the AI generated, or a dict if it was tools, or none on error
        # but still, needs a rework FIXME
        if not result:
            return None

        if (
            payload := result["choices"][0]["message"].get("content", None)
        ) is not None:
            return payload
        if result["choices"][0]["message"].get("tool_calls", None) is not None:
            # consumers of this like applyTools expect a dict here
            # FIXME: this is a bad function since it returns sometimes str sometimes dict. this is a big refactor though, as it would involve rewriting a bunch of internal types in pydantic
            return result
        return None

    @staticmethod
    def makeOpenAICallback(callback, last_result_callback=lambda x: x):
        def openAICallback(d):
            last_result_callback(d)
            # FIXME: handle reasoning here
            choices = d["choices"]
            if not(choices):
                return
            choice = choices[0]

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

        data = payload | {"max_tokens": payload["max_length"], "stream": True, "stream_options": {"include_usage": True}}
        self._last_request = data

        def one_line_lambdas_for_python(r):
            self._last_result = r


        final_endpoint = self.endpoint + "/v1/chat/completions"
        self.log(f"generateStreaming to {final_endpoint}.")
        self.log(f"Payload: {json.dumps(data, indent=4)}")        
        response = streamPrompt(
            self.makeOpenAICallback(
                callback, last_result_callback=one_line_lambdas_for_python
            ),
            self.stream_done,
            final_endpoint,
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
        self.log(f"tokenize with {len(w)} characters.")
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
        self.log(f"detokenize with {len(ts)} tokens.")
        response = requests.post(
            self.endpoint + "/v1/detokenize", headers=headers, json=data
        )
        if response.status_code == 200:
            return response.json()["content"]
        return []

    def health(self):
        # OpenAI API does not have a direct health check endpoint
        return "OpenAI API is assumed to be healthy."


    def timings(self, result_json=None) -> Optional[Timings]:
        if result_json is None:
            if (json := self._last_result) is None:
                return None
        else:
            json = result_json


        if "__verbose" in json:
            verbose = json["__verbose"]
            time = verbose["timings"]
            truncated = verbose["truncated"]
            cached = verbose["tokens_cached"]
        elif "timings" in json:
            time = json["timings"]
            truncated = False
            cached = None            
        else:
            return None

        return Timings(
            prompt_n=time["prompt_n"],
            predicted_n=time["predicted_n"],
            prompt_ms=time["prompt_ms"],
            predicted_ms=time["predicted_ms"],
            predicted_per_token_ms=time["predicted_per_token_ms"],
            predicted_per_second=time["predicted_per_second"],
            # unfortunately openai don't reveal these, unless we got __verbose
            truncated=truncated,
            cached_n=cached,
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


class GoogleBackend(AIBackend):
    """Backend for google's AI Studio https://aistudio.google.com"""

    def __init__(self, api_key: str, **kwargs):
        super().__init__("https://aistudio.google.com", **kwargs)
        self.api_key = api_key
        if not(self.api_key):
            printerr("error: Google AI Studio requires an API key. Please set it with either the --google_api_key or the general --api_key option. You can get an API key at https://aistudio.google.com")
            raise BrokenBackend("Missing API key for google.")

        api_str = "" if not(api_key) else " with api key " + api_key[:4] + ("x" * len(api_key[4:]))
       
        self.log(f"Initializing Google compatible backend {api_str}. Routing to https://aistudio.google.com. Config is {self._config}")
        # fail early on imports
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            printerr("error: Could not import google's SDK. Do you have the google-genai package installed?\nTry\n\n```\npip install google-genai```\n")
            raise RuntimeError("Aborting due to failed imports.")
        
        self._memoized_params = None
        # Store the model name in _config for later use by tokenize and generate
        # The 'model' key should be present in kwargs if passed from Plumbing
        # We also fix it here to ensure it's a supported model name
        initial_model = kwargs.get('model', "")
        self._config['model'] = self.fix_model(initial_model)
        
                                                                            
    def getName(self):
        return LLMBackend.google.name

    def getMaxContextLength(self):
        return -1


    def _get_models(self) -> List[Any]:
        """Returns a list of names of supported models by google."""
        return self.client.models.list()

    def get_models(self) -> List[ModelStats]:
        models = self._get_models()
        return [ModelStats(
            name = model.name,
            display_name = model.display_name,
            description = model.description
        )
                for model in models]
    
    def fix_model(self, model: str) -> str:
        """Ensures the given model name is a valid one. Returns a default model with a warning if not."""
        models = [model.name for model in self._get_models()]
        if model not in models:
            default_model = models[0] if len(models) > 0 else "gemini-2.5-flash"
            printerr(f"warning: Model {model} not supported by Google. Defaulting to {default_model}")
            return default_model
        return model

    
    def content_from_chatmessage(self, msg: ChatMessage) -> 'google.genai.types.Content':
        from google.genai.types import Content, Part, FunctionCall, FunctionResponse, Blob
        import base64

        # Google GenAI roles: 'user', 'model', 'tool'
        # System messages are usually handled by system_instruction, but if they appear in history, they are treated as user messages by GenAI.
        role_map = {"user": "user", "assistant": "model", "tool": "tool", "system": "user"} 

        genai_parts = []

        # Handle text content
        if isinstance(msg.content, str) and msg.content:
            genai_parts.append(Part(text=msg.content))
        elif isinstance(msg.content, list): # Multimodal content
            for item in msg.content:
                if item.type == "text" and item.get_text():
                    genai_parts.append(Part(text=item.get_text()))
                elif item.type == "image_url" and item.image_url and item.image_url.url:
                    image_url_str = item.image_url.url
                    try:
                        # Expecting data URI: data:image/jpeg;base64,...
                        mime_type_part, base64_data_part = image_url_str.split(",", 1)
                        mime_type = mime_type_part.split(';')[0].split(':')[1]
                        image_bytes = base64.b64decode(base64_data_part)
                        genai_parts.append(Part(inline_data=Blob(mime_type=mime_type, data=image_bytes)))
                    except Exception as e:
                        self.log(f"warning: Could not parse image_url for Google GenAI: {image_url_str}. Error: {e}")
                        # Skip this image part if parsing fails
                        pass
                elif item.type == "video_url" and item.video_url and item.video_url.url:
                    video_url_str = item.video_url.url
                    try:
                        # Assuming data URI: data:video/mp4;base64,...
                        mime_type_part, base64_data_part = video_url_str.split(",", 1)
                        mime_type = mime_type_part.split(';')[0].split(':')[1]
                        video_bytes = base64.b64decode(base64_data_part)
                        genai_parts.append(Part(inline_data=Blob(mime_type=mime_type, data=video_bytes)))
                    except Exception as e:
                        self.log(f"warning: Could not parse video_url for Google GenAI: {video_url_str}. Error: {e}")
                        # Skip this video part if parsing fails
                        pass

                    
        # Handle tool calls (assistant requesting a tool)
        if msg.role == "assistant" and msg.tool_calls:
            for tool_call in msg.tool_calls:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    genai_parts.append(Part(function_call=FunctionCall(name=tool_call.function.name, args=function_args)))
                except json.JSONDecodeError:
                    self.log(f"warning: Could not decode tool arguments for Google GenAI: {tool_call.function.arguments}")
                    genai_parts.append(Part(text=f"Assistant requested tool {tool_call.function.name} with invalid arguments: {tool_call.function.arguments}"))
        
        # Handle tool results (tool role message)
        if msg.role == "tool" and msg.tool_name and msg.content is not None:
            tool_output = msg.content
            if isinstance(tool_output, (dict, list)):
                tool_output = json.dumps(tool_output) # Convert dict/list to JSON string
            
            genai_parts.append(Part(function_response=FunctionResponse(name=msg.tool_name, response={"result": tool_output})))
            # Note: GenAI's FunctionResponse `response` field is a dict.
            # We wrap the tool's content in a "result" key. This might need adjustment based on actual tool output structure.

        # Ensure at least one part if message has content but no specific parts were added
        if not genai_parts and msg.get_text():
            genai_parts.append(Part(text=msg.get_text()))
        
        # If still no parts, add an empty text part to avoid API errors for empty content
        if not genai_parts:
            genai_parts.append(Part(text=""))

        return Content(role=role_map.get(msg.role, "user"), parts=genai_parts)
    

    @staticmethod
    def get_safety_settings() -> List['google.genai.types.SafetySetting']:
        from google.genai import types
        # so the APi will error out with 400 invalid request if you set any other than the following (as per the docs)
        # kind of defeats the point of an enum. thanks, googl!
        supported_categories = [
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            types.HarmCategory.HARM_CATEGORY_HARASSMENT
        ]
        
        # FIXME: right now we just turn it all off. mayb let user configure?
        return [
                        types.SafetySetting(category=category,
                                            threshold=types.HarmBlockThreshold.OFF)
                        for category in list(types.HarmCategory)
                        if category in supported_categories
        ]

    def _post_generation(self):
        """Some processing that runs after generating text, shared between generation methods."""
        # Check for block/failure etc and notify the user
        if self._last_result is not None:
            if (prompt_feedback := self._last_result.get("prompt_feedback", None)) != None:
                printerr(f"warning: Got prompt feedback from server:\n{json.dumps(prompt_feedback, indent=4)}")

    def _prepare_generation_config(self, payload):
        """Prepare google's generate config for generation."""
        from google.genai import types, errors        
        generation_config = types.GenerateContentConfig(
            system_instruction=payload["system"],
            safety_settings=self.get_safety_settings(),
            temperature=payload.get("temperature", 0.8),
        )
        if payload.get("max_length", -1) > 0:
            generation_config.max_output_tokens = payload["max_length"]
        if payload.get("top_p") is not None:
            generation_config.top_p = payload["top_p"]
        if payload.get("top_k") is not None:
            generation_config.top_k = payload["top_k"]

        # structured output
        if payload["response_format"] != "text" and payload["response_format"] != "raw":
            generation_config.response_mime_type = "application/json"
            generation_config.response_json_schema = payload["response_format"]["schema"]
            
        return generation_config

    def _prepare_generation_contents(self, payload) -> Tuple[Any, Any]:
        """Prepare contents for google's generate_content method based on a payload.
        returns a pair of genai contents (for google) and serializable contents (for debugging/logging)."""
        from google.genai import types, errors                
        genai_contents = []
        for msg in payload["story"]:
            # System messages are passed via `system_instruction` argument, not in `contents` list.
            if msg.role == "system":
                continue
            genai_contents.append(self.content_from_chatmessage(msg))

        serializable_contents = self._serialize_content(genai_contents)
        return genai_contents, serializable_contents
    
    def generate(self, payload) -> Optional[Any]:
        from google.genai import types, errors
        
        generation_config = self._prepare_generation_config(payload)
        genai_contents, serializable_contents = self._prepare_generation_contents(payload)
            
        # Store the request for debugging
        self._last_request = {
            "model": payload["model"],
            "contents": [c.model_dump() for c in serializable_contents],
            "generation_config": generation_config.model_dump(),
            "safety_settings": [s.model_dump() for s in self.get_safety_settings()],
        }
        
        self.log(f"generate to Google GenAI. Payload: {json.dumps(self._last_request, indent=4)}")
        # ensure the model used here and the one for tokenization is the same
        self._config["model"] = self.fix_model(payload["model"])

        try:
            response = self.client.models.generate_content(
                contents=genai_contents,
                config=generation_config,
                model=self.fix_model(payload["model"])
            )
            self._last_result = response.model_dump()
            self._post_generation()
            return response
        except errors.APIError as e:
            self.last_error = f"Google API error: {e.code}, {e.message}"
            return None
        except Exception as e:
            self.last_error = f"Google API error: {e.__class__.__name__}: {e}\n{traceback.format_exc()}"
            self.log(self.last_error)
            return None


    def handleGenerateResult(self, result):
        return self.handleGenerateResultGoogle(result)


    @staticmethod
    def handleGenerateResultGoogle(result):
        # keep it simple
        try:
            # result.text aggregates all text parts from all candidates
            return result.text
        except Exception as e:
            printerr(f"warning: Exception while unpacking result from Google API. Traceback:\n{traceback.format_exc()}\n")
            return None


    def _serialize_content(self, genai_contents):
        """Serialize a google genai content type to json strings. This is mostly used for debugging and self.lastResult."""
        from google.genai import types
        # Before serializing to JSON, base64 encode any bytes data
        serializable_contents = []
        for c in genai_contents:
            serializable_parts = []
            for part in c.parts:
                if part.inline_data and isinstance(part.inline_data.data, bytes):
                    # no need to see binary data. just snip it.
                    serializable_parts.append(types.Part(text="<BINARY_BLOB>"))
                else:
                    serializable_parts.append(part)
            serializable_contents.append(Content(role=c.role, parts=serializable_parts))
        return serializable_contents

    def generateStreaming(self, payload, callback=lambda w: print(w)):
        self.stream_done.clear()
        from google.genai import types, errors

        generation_config = self._prepare_generation_config(payload)
        genai_contents, serializable_contents = self._prepare_generation_contents(payload)
            
        # Store the request for debugging
        self._last_request = {
            "model": payload["model"],
            "contents": [c.model_dump() for c in serializable_contents],
            "generation_config": generation_config.model_dump(),
            "safety_settings": [s.model_dump() for s in self.get_safety_settings()],
        }
        self.log(f"generateStreaming to Google GenAI. Payload: {json.dumps(self._last_request, indent=4)}")
        # ensure the model used here and the one for tokenization is the same
        self._config["model"] = self.fix_model(payload["model"])
        
        try:
            stream_response = self.client.models.generate_content_stream(
                contents=genai_contents,
                config=generation_config,
                model=self.fix_model(payload["model"]),
            )

            full_response_text = ""
            # The `stream_response` is an iterable of `GenerateContentResponse` objects.
            # Each `GenerateContentResponse` has a `candidates` list, and each candidate has `content`.
            # We need to extract text from these parts.
            for chunk in stream_response:
                if self.stream_done.is_set():
                    self.log("Streaming stopped by external signal.")
                    break
                
                # A chunk might have multiple candidates, usually just one.
                # A candidate's content might have multiple parts.
                if chunk.candidates:
                    candidate_content = chunk.candidates[0].content
                    for part in candidate_content.parts:
                        if part.text:
                            callback(part.text)
                            full_response_text += part.text
                        # For now, we only stream text. Tool_code or tool_response parts are not streamed as raw text.
                
                # Store the last chunk's model_dump for _last_result
                # This will be overwritten by the final aggregated response if available
                self._last_result = chunk.model_dump()

        except errors.APIError as e:
            self.last_error = f"Google API error: {e.code}, {e.message}"
            return True
        except Exception as e:
            self.last_error = f"Google API error: {e.__class__.__name__}: {e}"
            self.log(self.last_error + f"Full Traceback:\n{traceback.format_exc()}")
            return True # Indicate error

        finally:
            self.stream_done.set() # Signal completion
            self._post_generation()
        return False # No HTTP error
    
    def _make_content_from_raw_text(self, text: str) -> 'google.generativeai.types.Content':
        """Helper to create a Content object from a raw string for token counting."""
        return Content(role="user", parts=[Part(text=text)])

    def tokenize(self, w: str) -> List[int]:
        self.log(f"Attempting to tokenize {len(w)} characters for Google backend (only token count is supported).")
        try:
            content_to_count = self._make_content_from_raw_text(w)
            response = self.client.models.count_tokens(contents=[content_to_count],
                                                       model=self.fix_model(self._config.get("model", "")))
            token_count = response.total_tokens
            self.log(f"Token count for '{w[:50]}...' is {token_count}.")
            # Return a list of placeholder integers so len() works as expected
            return [0] * token_count
        except Exception as e:
            self.log(f"warning: Tokenization (counting) failed for Google backend: {e.__class__.__name__}: {e}\n{traceback.format_exc()}")
            return []

    def detokenize(self, ts: List[int]) -> str:
        self.log("warning: Detokenization is not directly supported by the Google GenAI API.")
        return ""
    
    def health(self):
        return "Google AI Studio API is assumed to be healthy."

    def timings(self, result_json=None) -> Optional[Timings]:
        # Google GenAI responses include usage metadata in the final response.
        # We can extract this to populate Timings.
        if result_json is None:
            if (json_data := self._last_result) is None:
                return None
        else:
            json_data = result_json

        try:
            usage_metadata = json_data.get("usage_metadata", {})
            prompt_token_count = usage_metadata.get("prompt_token_count", 0)
            candidates_token_count = usage_metadata.get("candidates_token_count", 0)
            total_token_count = usage_metadata.get("total_token_count", 0)

            # Google GenAI does not directly provide ms timings per token/prompt.
            # We can only infer total tokens.
            # For now, set ms timings to 0 or placeholder.
            return Timings(
                prompt_n=prompt_token_count,
                predicted_n=candidates_token_count,
                cached_n=None, # Not directly available
                truncated=False, # Not directly available
                prompt_ms=0.0,
                predicted_ms=0.0,
                predicted_per_second=0.0,
                predicted_per_token_ms=0.0,
                original_timings=usage_metadata,
            )
        except Exception as e:
            self.log(f"warning: Exception while extracting timings from Google API result: {e.__class__.__name__}: {e}\n{traceback.format_exc()}")
            return None
    
    def sampling_parameters(self) -> Dict[str, SamplingParameterSpec]:
        if self._memoized_params is not None:
            return self._memoized_params

        # Google GenAI's `GenerateContentConfig` supports:
        # temperature: float (0.0 - 1.0)
        # top_p: float (0.0 - 1.0)
        # top_k: int (positive)
        # max_output_tokens: int (positive) -> maps to Ghostbox's max_length
        # stop_sequences: List[str] -> maps to Ghostbox's stop

        google_supported_params = {
            "temperature": sampling_parameters["temperature"],
            "top_p": sampling_parameters["top_p"],
            "top_k": sampling_parameters["top_k"],
            "max_length": sampling_parameters["max_length"], 
            "stop": sampling_parameters["stop"],
        }
            
        self._memoized_params = google_supported_params
        return google_supported_params



class DeepseekBackend(OpenAIBackend):
    """Backend for the deepseek cloud LLM provider.
        This is a razor thin wrapper around the OpenAI ctype. It exists mostly to provide a list of model and to be future proof.
        """

    def __init__(self, api_key: str, endpoint:str="https://api.deepseek.com", **kwargs):
        super().__init__(api_key, endpoint, **kwargs)        

    def getName(self) -> str:
        return "Deepseek"

    @staticmethod
    def _fix_payload(deepseek_payload: Dict[str, Any]) -> None:
        """Modify a payload to be more pallatable to the deepseek API."""
        # FIXME: figure out the enum they expect
        if isinstance(deepseek_payload["response_format"], str):
            del deepseek_payload["response_format"]

        # clamp the length
        upper_bound = 65536
        n = deepseek_payload["max_length"]
        n = upper_bound if n == -1 else n
        deepseek_payload["max_length"] = min(upper_bound, max(1, n))


        if deepseek_payload["top_p"] <= 0.0 or deepseek_payload["top_p"] > 1.0:
            deepseek_payload["top_p"] = 0.95
                
    def generate(self, payload: Dict[str, Any]) -> Any:
        self._fix_payload(payload)
        return super().generate(payload)

    def generateStreaming(self, payload: Dict[str, Any], callback=lambda w: print(w)) -> None:
        self._fix_payload(payload)
        super().generateStreaming(payload, callback)
        
    
    def get_models(self) -> List[ModelStats]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get("https://api.deepseek.com/models", headers=headers)
        if response.status_code != 200:
            self.log(f"Got status code {response.status_code} during model query.")
            return []

        try:
            data = response.json()["data"]
            return [ModelStats(
                name=record["id"],
                display_name=record["id"]
            )
                    for record in data]
        except Exception as e:
            self.log(f"Couldn't get deepseek models. Reason: {e}")
        return []
    
    
            

        
