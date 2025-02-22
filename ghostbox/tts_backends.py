from abc import ABC, abstractmethod
from functools import *
from typing import *
from ghostbox.util import *
from ghostbox.definitions import *

def initTTS(model: str):
    if model == "xtts":
        return XTTSBackend()
    elif model == "zonos":
        return ZonosBackend()
    # FIXME: more models
    
class TTSBackend(ABC):
    """Abstract interface to a TTS model, like xtts2 (now derelict), tortoise, or zonos."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def tts_to_file(self, text: str, file_path: str, language:str = "en", speaker_wav:str = "") -> None:
        """Given a message, writes the message spoken as audio to a wav file."""
        pass

    @abstractmethod
    def split_into_sentences(self, text:str) -> List[str]:
        """Returns a list of sentences, where a 'sentence' is any string the TTS backend wants to process as a chunk.
        The default implementation splits on common punctuation marks."""
        def split(ws, split_char):
            return reduce(lambda xs, ys: xs + ys, [w.split(split_char) for w in ws], [])
        return reduce(split, ".!?:;", [text])

class XTTSBackend(TTSBackend):
    """Bindings for the xtts2 model, which is currently (2025) in license limbo and should not be used for production purposes.
This immplementation remains here as a reference implementation."""

    def __init__(self):
        super().__init__()
        # fail importing these early
        import torch        
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        printerr("Using " + device)        
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)        

    def tts_to_file(self, text: str, file_path: str, language:str = "en", speaker_wav:str = "") -> None:
        self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
        
    def split_into_sentences(self, text:str) -> List[str]:
        return self.tts.synthesizer.split_into_sentences(msg)        
