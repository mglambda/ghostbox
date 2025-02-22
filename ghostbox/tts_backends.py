import traceback
from abc import ABC, abstractmethod
from functools import *
from typing import *
from ghostbox.util import *
from ghostbox.definitions import *

import nltk.data
nltk.download('punkt_tab')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def initTTS(model: str, config: Dict[str, Any] = {}):
    if model == "xtts":
        return XTTSBackend()
    elif model == "zonos":
        return ZonosBackend(config=config)
    # FIXME: more models
    
class TTSBackend(ABC):
    """Abstract interface to a TTS model, like xtts2 (now derelict), tortoise, or zonos."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]={}) -> None:
        pass

    @abstractmethod
    def tts_to_file(self, text: str, file_path: str, language:str = "en", speaker_file:str = "", seed: int = 420) -> None:
        """Given a message, writes the message spoken as audio to a wav file."""
        pass

    @abstractmethod
    def split_into_sentences(self, text:str) -> List[str]:
        """Returns a list of sentences, where a 'sentence' is any string the TTS backend wants to process as a chunk.
        The default implementation splits on common punctuation marks."""
        return tokenizer.tokenize(text)


    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Set parameters specific to a TTS model."""
        pass
    
        
class XTTSBackend(TTSBackend):
    """Bindings for the xtts2 model, which is currently (2025) in license limbo and should not be used for production purposes.
This immplementation remains here as a reference implementation."""

    def __init__(self, config: Dict[str, Any]={}):
        super().__init__()
        # fail importing these early
        import torch        
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        printerr("Using " + device)        
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)        

    def tts_to_file(self, text: str, file_path: str, language:str = "en", speaker_file:str = "", seed: int = 420) -> None:
        self.tts.tts_to_file(text=text, speaker_wav=speaker_file, language=language, file_path=file_path)
        
    def split_into_sentences(self, text:str) -> List[str]:
        return self.tts.synthesizer.split_into_sentences(msg)        

    def configure(self, **kwargs) -> None:
        super().configure(**kwargs)
        
class ZonosBackend(TTSBackend):
    """Bindings for the zonos v0.1 model. See https://github.com/Zyphra/Zonos"""

    def __init__(self, config: Dict[str, Any]={}):
        super().__init__(config=config)
        self._speakers = {}        
        # default config
        self.config = {"zonos_model": "Zyphra/Zonos-v0.1-transformer"}
        self._model_fallback = self.config["zonos_model"]
        self.configure(**config)
        self._init()

    def _init(self) -> None:
        printerr("Initializing zonos TTS model " + self.config["zonos_model"] + ".")        
        # fail importing these early
        import torch
        import torchaudio
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        from zonos.utils import DEFAULT_DEVICE as device
        printerr("Using " + str(device))        
        #self._model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
        #self._model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

        try:
            self._model = Zonos.from_pretrained(self.config["zonos_model"], device=device)
        except:
            if self.config["zonos_model"] != self._model_fallback:
                printerr("warning: Couldn't load model. Retrying with fallback. Below is the full traceback.")
                printerr(traceback.format_exc())                            
                self.configure(zonos_model=self._model_fallback)
                self._init()
                return
            printerr("error: Couldn't load model. Panicing, as there is no more fallback. Below is the full traceback. Goodbye.")
            printerr(traceback.format_exc())                            

    def configure(self, **kwargs) -> None:
        for key, value in kwargs.items():
            # we only configure options that are in the default config
            if key not in self.config:
                continue

            # some special cases
            if key == "zonos_model":
                if value == "transformer":
                    self.config["zonos_model"] = "Zyphra/Zonos-v0.1-transformer"
                elif value == "hybrid":
                    self.config["zonos_model"] = "Zyphra/Zonos-v0.1-hybrid"
                else:
                    self.config["zonos_model"] = value
            else:
                self.config[key] = value
        
    def tts_to_file(self, text: str, file_path: str, language:str = "en-us", speaker_file:str = "", seed:int = 420) -> None:
        import torch
        import torchaudio
        from zonos.conditioning import make_cond_dict

        # we want to support the 'en' code because xtts uses it
        if language == 'en':
            language = "en-us"

        if speaker_file not in self._speakers:
            self._create_speaker(speaker_file)
        speaker = self._speakers[speaker_file]

        torch.manual_seed(seed)
        e = self.get_default_emotions()
        e["other"] = 10.0
        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language, emotion=list(e.values()), pitch_std=200.0)
        conditioning = self._model.prepare_conditioning(cond_dict)
        codes = self._model.generate(conditioning)
        wavs = self._model.autoencoder.decode(codes).cpu()
        torchaudio.save(file_path, wavs[0], self._model.autoencoder.sampling_rate)
        
        
    def split_into_sentences(self, text:str) -> List[str]:
        ws = super().split_into_sentences(text)
        # debug
        print(str(ws))
        return ws
    

    def get_default_emotions(self):
        names = "happiness sadness disgust fear surprise anger other neutral".split(" ")
        emotions = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
        return dict(list(zip(names, emotions)))
        return 
        
    
    def _create_speaker(self, speaker_file):
        import torchaudio        
        wav, sampling_rate = torchaudio.load(speaker_file)
        self._speakers[speaker_file] = self._model.make_speaker_embedding(wav, sampling_rate)
