from typing import *
import traceback
import subprocess
import os
from glob import glob
import threading
import sys
import requests
import json
import time
import wave
import asyncio
import numpy as np
import torch
import queue
from huggingface_hub import snapshot_download
from ghostbox.tts_backends import TTSBackend
try:
    import feedwater
    _feedwater = True
except ModuleNotFoundError:
    _feedwater = False
    


class OrpheusBackend(TTSBackend):
    """Backend for the orpheus tts model.
        https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained
    https://github.com/canopyai/Orpheus-TTS

    Much of the code for this was taken from:
    https://github.com/isaiahbjork/orpheus-tts-local
    much respect.

        This implementation has been tested with llama.cpp as provider for the underlying llm. Any server that offers an OpenAI compatible endpoint should in principle work.
    """

    def __init__(self, **kwargs):
        """Creates a Orpheus TTS backend."""
        # some defaults for sampling parameters (meant for llamacpp through ghostbox)
        # you can experiment, but at least these are known to work
        default_config = {
            "temperature": 0.6,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "max_length": 1024,
            # this is factory default
            # "samplers": ["penalties", "top_p", "temperature"],
            # this is what we do, which is roughly 3x faster
            "samplers":["penalties", "min_p","temperature"],
            # orpheus specific stuff
            "available_voices": [
                "tara",
                "leah",
                "jess",
                "leo",
                "dan",
                "mia",
                "zac",
                "zoe",
            ],
            "special_tags": [
                "<laugh>",
                "<chuckle>",
                "<sigh>",
                "<cough>",
                "<sniffle>",
                "<groan>",
                "<yawn>",
                "<gasp>",
            ],
            "voice": "",  # we set tara as default below
            "volume": 1.0, # user can set this to change generated volume. we also boost the default volume by a bit
            "start_token_id": 128259,
            "end_token_ids": [128009, 128260, 128261, 128257],
            "custom_token_prefix": "<custom_token_",
            # this is the audio decoder
            "snac_model": "hubertsiuzdak/snac_24khz",
            "sample_rate": 24000,
        }

        if "config" in kwargs:
            kwargs["config"] = default_config | kwargs["config"]

        super().__init__(**kwargs)

        # we want to allow users to use undeifined voices
        # but we provide tara as default if they don't specify at all
        if self.config["voice"] == "":
            self.config["voice"] = "tara"


        # we boost the volume of all orpheus voices
        # as I find them too quiet
        # but tara is especially egregious
        if self.config["voice"] == "tara":
            self.volume_boost = 1.5
        else:
            # FIXME: keep experimenting, the other voices might not need a boost
            self.volume_boost = 1.25
                
            
        from ghostbox.util import printerr
        self._print_func = printerr
        # this is used for time stats when streaming, the default value below will be overriden
        self._start_time = time.time()
        
        self._init()

    def _print(self, w):
        self._print_func(w)

    def _init(self):
        self._print("Initializing orpheus TTS backend.")
        from snac import SNAC
        import ghostbox

        # SNAC model uses 24kHz

        self.model = SNAC.from_pretrained(self.config["snac_model"]).eval()
        # Check if CUDA is available and set device accordingly
        self.snac_device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._print(f"Using device: {self.snac_device}")
        self.model = self.model.to(self.snac_device)

        # figure out the llm server endpoint
        # if the user set it, it's easy
        if (llm_server := self.config["llm_server"]) != "":
            endpoint = llm_server
        else:
            self._print("Spawning LLM server...")
            self._spawn_llm_server()
            endpoint = "http://localhost:8181"

        self.box = ghostbox.from_openai_legacy(
            character_folder="",
            endpoint=endpoint,
            stdout=False,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_legnth=self.config["max_length"],
            samplers=self.config["samplers"],
            # samplers=["min_p","temperature", "xtc"],
            repeat_penalty=self.config["repeat_penalty"],
            prompt_format="raw",
            # for debug
            verbose=False,
        )

    def _spawn_llm_server(self) -> None:
        model_name = self._get_orpheus_model_name()
        executable = self.config["llm_server_executable"]
        max_context_length = self.config["max_length"]
        quant = (self.config["orpheus_quantization"][:2] + "_0").lower()
        port = 8181

        # load all of it or gtfo
        layers = 999

        # obviously this is specific to llamacpp
        command = f"{executable} --port {port} -c {max_context_length} -ngl {layers} -fa -ctk {quant} -ctv {quant} --mlock"
        cmd_list =             command.split(" ")
        # extend so we properly escape the model_name
        cmd_list.extend(["-m", model_name])
        self._print(f"Executing `{command}`.")
        global _feedwater
        if _feedwater:
            # the advantage of feedwater for our purposes is that it will 100% clean up and kill the process
            self._server_process = feedwater.run(" ".join(cmd_list))
        else:
            # with subprocess, the llama-server might not be killed at program end
            # which is annoying as it will hold like 3gigs of vram
            self._server_process = subprocess.Popen(
                cmd_list,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )


    def _get_orpheus_model_name(self) -> str:
        """Returns filepath to an orpheus model."""
        self._print(f"Considering orpheus model: {self.config["orpheus_model"]}")
        if (model := self.config["orpheus_model"]) != "":
            if os.path.isfile(model):
                return model
        else:
            # see the quant and just use some defaults
            quant = self.config["orpheus_quantization"]
            if quant == "Q4_K_M":
                model = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
            elif quant == "Q8_0":
                model = "lex-au/Orpheus-3b-FT-Q8_0.gguf"
            else:
                raise RuntimeError(
                    f"fatal error: Sorry, we don't support that quantization level yet ({quant})."
                )

        # assume model is a huggingface repo
        repo_dir = snapshot_download(model)
        try:
            return glob(os.path.join(repo_dir, "*.gguf"))[0]
        except:
            raise RuntimeError(f"fatal error: Could not find gguf file in directory '{repo_dir}' after downloading snapshot for '{model}'.")
            
        
    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_voices(self) -> List[str]:
        return self.config["available_voices"]

    def configure(self, **kwargs) -> None:
        # FIXME: too lazy to put the argument list here atm
        self.config |= kwargs

    def split_into_sentences(self, text: str) -> List[str]:
        # need to experiment what works best with orpheus
        # return super().split_into_sentences(text)
        return (text.strip()).split("\n")


    

    def can_stream(self) -> bool:
        return True

    def tts_to_generator(self, text:str) -> Iterator[bytes]:
        voice = os.path.basename(self.config["voice"])
        self._start_time = time.time()
        return self.tokens_decoder_sync(
            self.generate_tokens_from_api(
                prompt=text,
                voice=voice
            ),
        )

    
    def tts_to_file(
        self, text: str, file_path: str, language: str = "en", speaker_file: str = ""
    ) -> None:
        # speaker file is just a name in this case, and we default to tara
        if speaker_file == "":
            speaker_file = self.config["voice"]

        # FIXME: see _init
        speaker_file = os.path.basename(speaker_file)
            
        start_time = time.time()
        self.tokens_decoder_sync(
            self.generate_tokens_from_api(
                prompt=text,
                voice=speaker_file,
            ),
            output_file=file_path,
        )

        end_time = time.time()
        self._print(
            f"Speech generation completed in {end_time - start_time:.2f} seconds"
        )

    def format_prompt(self, prompt: str, voice: str) -> str:
        """Format prompt for Orpheus model with voice prefix and special tokens."""
        if voice not in self.config["available_voices"]:
            self._print(
                f"warning: Voice '{voice}' not recognized. Hope you know what you're doing!"
            )

        # Format similar to how engine_class.py does it with special tokens
        formatted_prompt = f"{voice}: {prompt}"

        # Add special token markers for the LM Studio API
        special_start = "<|audio|>"  # Using the additional_special_token from config
        special_end = "<|eot_id|>"  # Using the eos_token from config
        return f"{special_start}{formatted_prompt}{special_end}"

    def generate_tokens_from_api(
        self,
        prompt: str,
        voice: str,
    ):
        """Generate tokens using the API of a llama.cpp llama-server."""
        formatted_prompt = self.format_prompt(prompt, voice)

        token_counter = 0
        q = queue.Queue()
        done = threading.Event()
        done.clear()
        generation = ""

        def process_token(w):
            nonlocal token_counter
            q.put(w.strip())
            token_counter += 1

        def gen(w):
            nonlocal done
            nonlocal generation
            generation = w
            done.set()

        self.box.clear_history()
        with self.box.options(
            **{
                k: v
                for k, v in self.config.items()
                if k
                in "temperature top_p min_p repeat_penalty samplers max_length".split(
                    " "
                )
            }
        ):
            self.box.text_stream(
                formatted_prompt,
                chunk_callback=process_token,
                generation_callback=gen,
            )

        # have to do it this way because the original code used generators
        # and we are lazy


        self._print(f"`{prompt}`")
        while not (done.is_set()):
            try:
                token = q.get(timeout=1)
                yield token
            except queue.Empty:
                # without this threads don't react to signals which is annoying
                continue


        # debug
        # the ghostbox object will print timing information as long as box.timings == True
        #self._print(f"Token generation complete with {token_counter} tokens.")        
        #timings = self.box._plumbing.getBackend().timings()
        #self._print(f"tokens generated: {timings.predicted_n}\nt/s: {timings.predicted_per_second}")
        #self._print(f"timings: {json.dumps(timings.model_dump(), indent=4)}")
        # #self._print(generation)

    ### decoder methods below ###
    async def tokens_decoder(self, token_gen):
        """Asynchronous token decoder that converts token stream to audio stream."""
        buffer = []
        count = 0
        async for token_text in token_gen:
            token = self.turn_token_into_id(token_text, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Convert to audio when we have enough tokens
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = self.convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples

    def tokens_decoder_sync(self, syn_token_gen, output_file: Optional[str]=None):
        """Synchronous wrapper for the asynchronous token decoder."""
        audio_queue = queue.Queue()
        audio_segments = []

        # If output_file is provided, prepare WAV file
        wav_file = None
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            wav_file = wave.open(output_file, "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config["sample_rate"])

        # Convert the synchronous token generator into an async generator
        async def async_token_gen():
            for token in syn_token_gen:
                yield token

        async def async_producer():
            async for audio_chunk in self.tokens_decoder(async_token_gen()):
                audio_queue.put(audio_chunk)
            audio_queue.put(None)  # Sentinel to indicate completion

        def run_async():
            asyncio.run(async_producer())

        # Start the async producer in a separate thread
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

        # Process audio as it becomes available
        while True:
            try:
                audio = audio_queue.get(timeout=1)
                if audio is None:
                    break
            except queue.Empty:
                continue

            audio_segments.append(audio)

            # Write to WAV file if provided
            if wav_file:
                wav_file.writeframes(audio)
            else:
                yield audio

        # Close WAV file if opened
        if wav_file:
            wav_file.close()

        thread.join()

        # Calculate and print duration
        duration = (
            sum([len(segment) // (2 * 1) for segment in audio_segments])
            / self.config["sample_rate"]
        )
        self._print(f"Generated {len(audio_segments)} audio segments")
        self._print(f"Generated {duration:.2f} seconds of audio")
        if not(wav_file):
            # print the stats here
            end_time = time.time()
            self._print(
                f"Speech generation completed in {end_time - self._start_time:.2f} seconds"
            )
        
            


    def convert_to_audio(self, multiframe, count):
        """Convert token frames to audio."""
        frames = []
        if len(multiframe) < 7:
            return

        codes_0 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self.snac_device, dtype=torch.int32)

        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]

        for j in range(num_frames):
            i = 7 * j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor(
                    [frame[i]], device=self.snac_device, dtype=torch.int32
                )
            else:
                codes_0 = torch.cat(
                    [
                        codes_0,
                        torch.tensor(
                            [frame[i]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )

            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor(
                    [frame[i + 1]], device=self.snac_device, dtype=torch.int32
                )
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 4]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )
            else:
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 1]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 4]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )

            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor(
                    [frame[i + 2]], device=self.snac_device, dtype=torch.int32
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 3]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 5]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 6]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )
            else:
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 2]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 3]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 5]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 6]], device=self.snac_device, dtype=torch.int32
                        ),
                    ]
                )

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        # check that all tokens are between 0 and 4096 otherwise return *
        if (
            torch.any(codes[0] < 0)
            or torch.any(codes[0] > 4096)
            or torch.any(codes[1] < 0)
            or torch.any(codes[1] > 4096)
            or torch.any(codes[2] < 0)
            or torch.any(codes[2] > 4096)
        ):
            return

        with torch.inference_mode():
            audio_hat = self.model.decode(codes)

        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        # boost the volume as I find the default rather quiet
        # and also apply user volume
        audio_np = audio_np * self.volume_boost * self.config["volume"]
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes

    def turn_token_into_id(self, token_string, index):
        """Convert token string to numeric ID for audio processing."""
        # Strip whitespace
        token_string = token_string.strip()

        # Find the last token in the string
        last_token_start = token_string.rfind(self.config["custom_token_prefix"])

        if last_token_start == -1:
            return None

        # Extract the last token
        last_token = token_string[last_token_start:]

        # Process the last token
        if last_token.startswith(
            self.config["custom_token_prefix"]
        ) and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                return token_id
            except ValueError:
                return None
        else:
            return None
