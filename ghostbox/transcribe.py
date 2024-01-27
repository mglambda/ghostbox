import whisper
import time, os, sys, contextlib, threading
import wave
import tempfile
from ctypes import *
import pyaudio
from pydub import pyaudioop
import audioop
import math
from collections import deque

def loadModel(name="base.en"):
    return whisper.load_model(name)

def getWhisperTranscription(filename, model):
    result = model.transcribe(filename, fp16=False)
    return result["text"].strip()
    
# unfortunately pyaudio will give a bunch of error messages, which is very irritating for using it in a shell program, so we supress the msgs
@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)



def record_on_detect(file_name, silence_limit=1, silence_threshold=2500, chunk=1024, rate=44100, prev_audio=1):
    """ Silence limit in seconds. The max ammount of seconds where
only silence is recorded. When this time passes the
recording finishes and the file is delivered.

The silence threshold intensity that defines silence
and noise signal (an int. lower than THRESHOLD is silence).
Previous audio (in seconds) to prepend. When noise
is detected, how much of previously recorded audio is
prepended. This helps to prevent chopping the beginning
of the phrase."""
    
    CHANNELS = 2
    FORMAT = pyaudio.paInt16

    with ignoreStderr():      
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(2),
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        output=False,
                        frames_per_buffer=chunk)

        listen = True
        started = False
        rel = rate/chunk
        frames = []

        prev_audio = deque(maxlen=int(prev_audio * rel))
        slid_window = deque(maxlen=int(silence_limit * rel))

        while listen:
            data = stream.read(chunk)
            slid_window.append(math.sqrt(abs(audioop.avg(data, 4))))

            if(sum([x > silence_threshold for x in slid_window]) > 0):
                if(not started):
                    # recording starts here
                    started = True
            elif (started is True):
                started = False
                listen = False
                prev_audio = deque(maxlen=int(0.5 * rel))

            if (started is True):
                frames.append(data)
            else:
                prev_audio.append(data)

        stream.stop_stream()
        stream.close()

        p.terminate()

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(list(prev_audio)))
    wf.writeframes(b''.join(frames))
    wf.close()
        

class WhisperTranscriber(object):
    def __init__(self, model, silence_threshold=2500):
        self.model = model
        self.silence_threshold = silence_threshold
        

    def TranscribeDirectly(self, input_msg=""):
        """Records audio directly from the microphone and then transcribes it to text using Whisper, returning that transcription."""    
        # Create a temporary file to store the recorded audio (this will be deleted once we've finished transcription)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav")

        sample_rate = 16000
        bits_per_sample = 16
        chunk_size = 1024
        audio_format = pyaudio.paInt16
        channels = 1

        def callback(in_data, frame_count, time_info, status):
            wav_file.writeframes(in_data)
            return None, pyaudio.paContinue

        # Open the wave file for writing
        wav_file = wave.open(temp_file.name, 'wb')
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)

        # Suppress ALSA warnings (https://stackoverflow.com/a/13453192)
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            return

        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        asound = cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)

        # Initialize PyAudio
        with ignoreStderr():    
            audio = pyaudio.PyAudio()

            # Start recording audio
            stream = audio.open(format=audio_format,
                                channels=channels,
                                rate=sample_rate,
                                input=True,
                                frames_per_buffer=chunk_size,
                                stream_callback=callback)

            input(input_msg)
            # Stop and close the audio stream
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # Close the wave file
            wav_file.close()

            # And transcribe the audio to text (suppressing warnings about running on a CPU)
            result = getWhisperTranscription(temp_file.name, self.model)
            temp_file.close()
            return result

    def TranscribeContinuously(self):
        """Starts recording continuously, transcribing audio when a given volume threshold is reached.
        This function is non-blocking, but returns a ContinuousTranscriber object, which runs asynchronously and must be polled to get the latest transcription (if any)."""
        return ContinuousTranscriber(self.model, self.silence_threshold)

class ContinuousTranscriber(object):
    def __init__(self, model, silence_threshold):
        self.model = model
        self.silence_threshold = silence_threshold
        self.buffer = []
        self.running = False
        self.resume_flag = threading.event()
        self.payload_flag = threading.Event()
        self._spawnThread()

    def _spawnThread(self):
        self.running = True
        self.resume_flag.set()
        self.payload_flag.clear()
        thread = threading.Thread(target=self._recordLoop, args=())
        thread.start()

def _recordLoop(self):
    while self.running:
        self.resume_flag.wait()
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav")
        self.buffer.append(record_on_detect(temp_file.name))
        self.payload_flag.set()
        
def pause(self):
    self.resume_flag.clear()

def resume(self):
    self.resume_flag.set()

def stop(self):
    # FIXME: what if resume_flag is false and thread is waiting on the flag? this might be a memory issue (thread never gets GCed). But if we set() the flag here, it records again and behaviour might be weird
    self.running = False

    def pop(self):
        """Returns a list of strings that were recorded and transcribed since the last time poll or pop was called.
        This function is non-blocking."""
        tmp = self.buffer
        self.buffer = [] #FIXME: race condition?
        return tmp

    def poll(self):
        """Returns a list of strings that were recorded and transcribed since the last time poll or pop was called. 
This function will block until new input is recorded."""
        self.payload_flag.wait()
        self.payload_flag.clear()
        return self.pop()
    
        
                               
