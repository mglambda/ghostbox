import traceback, threading
from abc import ABC, abstractmethod
from functools import *
from typing import *
from ghostbox.util import *
from ghostbox.definitions import *

class TTSOutput(ABC):
    """Manages output of TTS sound."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def play(self, filename: str, volume: float=1.0) -> None:
        pass

    def stop(self) -> None:
        """Instantly interrups and stops any ongoing playback. This method is thread safe."""
        pass
    
class DefaultTTSOutput(TTSOutput):
    """Local TTS sound output using pyaudio."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # fail early if pyaudio isn't available
        import pyaudio
        printerr("Using pyaudio for local playback.")
        self.stop_flag = threading.Event()
        self.pyaudio = pyaudio.PyAudio()        

    def play(self, filename: str, volume: float= 1.0) -> None:
        import wave, pyaudio
        
        wf = wave.open(filename, 'rb')
        p = self.pyaudio
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Read data in chunks
        chunk = 1024
        data = wf.readframes(chunk)
        # Play the audio data
        self.stop_flag.clear()
        while data:
            if self.stop_flag.isSet():
                break
            stream.write(data)
            data = wf.readframes(chunk)

        # Close the stream and PyAudio object
        stream.stop_stream()
        stream.close()

    def stop(self) -> None:
        """Instantly interrupts and stops all playback."""
        self.stop_flag.set()
