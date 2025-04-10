from typing import *
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from queue import Queue, Empty
import librosa        
import numpy as np
import websockets
from websockets.sync.client import connect, ClientConnection
import threading, time, sys, json, traceback
from ghostbox.util import printerr, get_default_microphone_sample_rate, get_default_output_sample_rate, is_output_format_supported, convert_int16_to_float

@dataclass
class RemoteMsg:
    text: str
    is_stderr: bool = False

client_cli_token = "TriggerCLIPrompt: "
_remote_info_token = "RemoteInfo: "
samplerate_token = "samplerate: "

class RemoteInfo(BaseModel):
    tts: bool
    tts_websock: bool
    tts_websock_host: str
    tts_websock_port: int
    audio: bool
    audio_websock: bool
    audio_websock_host: str
    audio_websock_port: int

    @staticmethod
    def show_json(data: str) -> str:
        return _remote_info_token + data

    @staticmethod
    def maybe_from_remote_payload(line: str) -> Optional['RemoteInfo']:
        try:
            data = json.loads(line[len(_remote_info_token):])
            info = RemoteInfo(**data)
        except:
            printerr("warning: Couldn't parse RemoteInfo.")
            printerr(traceback.format_exc())
            return None
        return info

    
@dataclass
class GhostboxClient:
    """Holds connection information and runs a remote session with a ghostbox started with --server.
    This is most often used with the --client command line option, and takes the --remote_host and --remote_port as arguments."""

    remote_host: str
    remote_port: int = 5150
    stdout_callback: Callable[[str], None] = lambda w: print(w, end="", flush=True)
    stderr_callback: Callable[[str], None] = printerr
    logging: bool = False
    running: bool = True
    # how many bytes are pushed to the output stream at a time. Note that the authority sits server side, as the server determines the size of the network packets. This value is merely used to set the output stream parameter.
    # FIXME: make this part of remoteinfo
    tts_chunk_size: int = 512
    # this is the sample rate that the server pushes audio data at. the server communicates it to us on connect
    tts_samplerate: Optional[float] = None
    

    # this is what prepends messages if they are supposed to be printed to standard error
    _stderr_token: str = "[|STDER|]:"
    # this will be computed on init
    _stderr_token_length: int = 0


    # this carries information about remote services
    _remote_info: Optional[RemoteInfo] = None
    
    # queue that carries messages that are sent from remote box, so it's the remote's stdout and stderr
    _message_queue: Queue[RemoteMsg] = field(default_factory=Queue)
    

    _print_thread: Optional[threading.Thread] = None
    _websock_thread: Optional[threading.Thread] = None
    _tts_thread: Optional[threading.Thread] = None
    _tts_playback_thread: Optional[threading.Thread] = None
    _audio_thread: Optional[threading.Thread] = None
    
    _websocket: Optional[ClientConnection] = None
    _tts_websocket: Optional[ClientConnection] = None    

    # wave chunks are on this queue
    _tts_play_queue: Queue[bytes] = field(default_factory=Queue)
    # if set, playback should stop immediately
    _tts_stop_flag: threading.Event = field(default_factory=threading.Event)
    # did the server get notified that we are done playing?
    _sent_done: bool = False
    
    

    def __post_init__(self):
        self._stderr_token_length = len(self._stderr_token)
        def print_loop():
            while self.running:
                try:
                    msg = self._message_queue.get(timeout=1)
                except Empty:
                    # don't delete this, it gets triggered on timeout, which we want because otherwise the thread won't react to signals
                    continue

                if msg.is_stderr:
                    self.stderr_callback(msg.text)
                else:
                    self.stdout_callback(msg.text)

        self._print_thread = threading.Thread(target=print_loop, daemon=True)
        self._print_thread.start()

        # this will establish a connection and handle messages
        self._websock_thread = threading.Thread(target=self._client_loop, daemon=True)
        self._websock_thread.start()
        
        # we need to know if we have to start audio and tts on the client box
        # we will start them if they are being served on the remote box
        # to find out what is served, we do a client handshake
        self._print("Determining remote services.")
        while self._remote_info is None:
            if self._websocket is not None:
                self.write_line("/client_handshake")
            time.sleep(1)

        info = self._remote_info
        no_services = True
        if info.tts and info.tts_websock:
            self._print("TTS service active.")
            no_services = False
            self._start_tts_client()

        if info.audio and info.audio_websock:
            self._print("Audio transcription service active.")
            no_services = False
            self._start_audio_client()
            

        if no_services:
            self._print("No services active.")
            
    def _print(self, log_msg) -> None:
        if self.logging:
            print(f"[Client] {log_msg} :: {time.strftime("%c")}", file=sys.stderr)
                     
                     
    def shutdown(self) -> None:
        self.running = False
        if self._websocket is not None:
            self._websocket.close()
            

            
    def uri(self) -> str:
        if self.remote_host.startswith("ws://"):
            prefix == ""
        elif self.remote_host.startswith("wss://"):
            raise RuntimeError(f"fatal error: SSL not supported yet for websockets. Change the uri '{self.remote_host}' to use ws:// instead.")
        else:
            prefix = "ws://"
            
        return f"{prefix}{self.remote_host}:{self.remote_port}"


    def pick_host(self, host_b) -> str:
        """Help pick the remote host in situations where we have several candidates.
        Mostly this is use to avoid using RemoteInfo hostnames that are localhost or 0.0.0.0.
        Example: ghostbox --client --remote_host galaxybrain.ai
        now let host_a="galaxybrain.ai"
        On galxybrain.ai, someone started the tts with tts_host="0.0.0.0", binding it to all available network interfaces, including galaxybrain.ai.
        Let host_b="0.0.0.0"
        Given this situation, obviously we want to connect to host_a. However, if host_b was a different hostname, like "universebrainhosting.ai", it might be that the tts is actually on a different machine, in which case we want host_b."""
        host_a = self.remote_host
        if host_b == "0.0.0.0" or host_b == "localhost":
            return host_a
        return host_b
    
        
    
    def _init_websocket(self) -> None:
        """Establishes the websocket connection to remote host."""
        if self._websocket is not None:
            self._print("Skipping initialization of websocket: Connection already established.")
            time.sleep(3)
            return

        self._print(f"Connecting to {self.uri()}.")
        try:
            self._websocket = connect(self.uri())
        except websockets.exceptions.ConnectionClosedError as e:
            self._print(f"Connection closed with error: {e}")
            time.sleep(3)
        except websockets.exceptions.InvalidURI as e:
            self._print(f"Invalid URI: {e}")
            time.sleep(3)
        #except websockets.exceptions.ConnectionRefusedError as e:
            #self._print(f"Connection refused: {e}")
            #time.sleep(3)
        except Exception as e:
            self._print(f"An error occurred while connecting: {e}")
            time.sleep(3)

        if self._websocket is not None:
            self._print(f"Successfully established connection to {self.uri()}.")

    def _client_loop(self) -> None:
        """Websock connection loop. Handles connecting and receiving messages from server."""
        while self.running:
            if self._websocket is None:
                self._init_websocket()
                
            try:
                msg = self._websocket.recv()
                if self._remote_info is None and msg.startswith(_remote_info_token):
                    # note that currently, updating the remote info while server is running is not supported
                    # only way to do that is to reconnect
                    if (info := RemoteInfo.maybe_from_remote_payload(msg)) is not None:
                        self._remote_info = info
                elif msg.startswith(self._stderr_token):
                    self._message_queue.put(RemoteMsg(text=msg[self._stderr_token_length:], is_stderr=True))
                elif msg.startswith( client_cli_token):
                    # this bypasses the message queue because we don't want # and color etc.
                    print(msg[len(client_cli_token):], flush=True, file=sys.stderr, end="")
                else:
                    self._message_queue.put(RemoteMsg(text=msg))

                    
            except websockets.exceptions.ConnectionClosedOK:
                self._print("Connection closed normally.")
                self._websocket = None
            except websockets.exceptions.ConnectionClosedError as e:
                self._print(f"Connection closed with error: {e}")
                self._websocket = None
            except Exception as e:
                self._print(f"An error occurred while receiving: {e}\n" + traceback.format_exc())
                time.sleep(5)
                # ??? what to do here

    def write_line(self, msg: str) -> None:
        """Sends a message to the remote ghostbox.
        This function does not block, and the message is not guaranteed to arrive."""
        try:
            self._websocket.send(msg)
        except websockets.exceptions.ConnectionClosedError as e:
            self._print(f"Connection closed with error: {e}")

    def tts_websocket_send(self, msg) -> None:
        """Sends a message to the remote tts if it is connected."""
        try:
            if self._tts_websocket is not None:
                self._tts_websocket.send(msg)
        except:
            self._print("Could not send message to tts: " + traceback.format_exc())
            
    def _start_tts_client(self) -> None:
        """Starts a websock client that connects to the remote TTS host and launches a handler loop that plays audio on the local machine."""
        import pyaudio
        import wave

        p = pyaudio.PyAudio()
        stream = None

        info = self._remote_info
        if not info or not info.tts_websock:
            self._print("TTS service not available.")
            return

        uri = f"ws://{self.pick_host(info.tts_websock_host)}:{info.tts_websock_port}"
        self._print(f"Connecting to TTS WebSocket at {uri}.")

        def handle_tts_msgs_loop():
            while self.running:
                try:
                    if self._tts_websocket is None:
                        self._tts_websocket = connect(uri)
                        
                    data = self._tts_websocket.recv()
                    if data == "stop":
                        self._print("Received stop signal.")
                        self._tts_play_queue.queue.clear()
                        self._tts_stop_flag.set()
                    elif type(data) == str and data.startswith(samplerate_token):
                        try:
                            self.tts_samplerate = float(data[len(samplerate_token):])
                            self._print(f"Set tts sample rate to {self.tts_samplerate} herz from server message.")
                        except:
                            self._print("error parsing sample rate!")
                    elif data == "ok":
                        self._sent_done = True
                    else:
                        self._print(f"Queueing {len(data)} bytes.")
                        self._tts_stop_flag.clear()
                        self._tts_play_queue.put(data)

                except websockets.exceptions.ConnectionClosedOK:
                    self._print("TTS WebSocket connection closed normally.")
                    time.sleep(3)                
                    continue
                except websockets.exceptions.ConnectionClosedError as e:
                    self._print(f"TTS WebSocket connection closed with error: {e}")
                    time.sleep(3)                
                    continue
                except Exception as e:
                    self._print(f"An error occurred while receiving TTS data: {e}")
                    time.sleep(3)                                    
                    continue
                except Exception as e:
                    self._print(f"Failed to connect to TTS WebSocket: {e}")
                    time.sleep(3)
                    continue

            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

        self._tts_thread = threading.Thread(target=handle_tts_msgs_loop, daemon=True)
        self._tts_thread.start()

        def playback_loop():
            # check for sample rate and if it's supported by our device
            while self.tts_samplerate is None:
                # we need to get it from the server
                self.tts_websocket_send("get_samplerate")
                time.sleep(1)


            # some parameters for pyaudio
            channels = 1
            format = pyaudio.paFloat32 #p.get_format_from_width(2),
                
            # ok we have the samplerate that the server wants. do we support it?
            if is_output_format_supported(self.tts_samplerate, channels=channels, format=format, pyaudio_object=p):
                resampling = False
                supported_samplerate = self.tts_samplerate
                self._print(f"Using device native {self.tts_samplerate} herz sample rate for tts output.")                
            else:
                resampling = True
                supported_samplerate = get_default_output_sample_rate(p)
                self._print(f"Sample rate of {self.tts_samplerate} not supported by device. Resampling to {supported_samplerate} for tts output.")
                
            nonlocal stream
            if not stream:
                stream = p.open(
                    #format=pyaudio.paFloat32,,
                    format=format,
                                channels=channels,
                                rate=int(supported_samplerate),
                    frames_per_buffer=self.tts_chunk_size,
                                output=True)
                stream.start_stream()

            while self.running:
                try:
                    data = self._tts_play_queue.get(timeout=0.01)
                except Empty:
                    if not(self._sent_done):
                        self._print("Done playing.")
                        self.tts_websocket_send("done")
                    continue

                self._sent_done = False
                self._tts_stop_flag.clear()
                # FIXME: right now the server always sends int16, so we at least always convert to float32, but this will change in the future
                np_data = convert_int16_to_float(data)
                if resampling:
                    np_data = librosa.resample(np_data, self.tts_samplerate, supported_samplerate)
                    
                stream.write(np_data.tobytes())

        self._tts_playback_thread = threading.Thread(target=playback_loop, daemon=True)
        self._tts_playback_thread.start()

        
    def _start_audio_client(self) -> None:
        """Starts a websock client that connects to the remote audio transcription service and launches a handler loop that records from the local microphone and sends audio data to the remote endpoint."""
        info = self._remote_info
        if not info or not info.audio_websock:
            self._print("Audio transcription service not available.")
            return

        uri = f"ws://{self.pick_host(info.audio_websock_host)}:{info.audio_websock_port}"
        self._print(f"Connecting to Audio WebSocket at {uri}.")

        def record_audio_loop():
            import pyaudio
            import wave
            # FIXME: so on some systems opening pyaudio will vomit a bunch of irrelevant error messages into sdterr
            # we can turn this off but that will sometimes supress other stderr messages
            # I don't wanna do that since the client is still under heavy development, we're just going to eat the errors for now
            p = pyaudio.PyAudio()
            sample_rate = rate if (rate := get_default_microphone_sample_rate(p)) is not None else 16000
            chunk_size = 1024

            def send_audio_data(data):
                try:
                    websocket.send(data)
                except websockets.exceptions.ConnectionClosedError as e:
                    self._print(f"Audio WebSocket connection closed with error: {e}")
                except Exception as e:
                    self._print(f"An error occurred while sending audio data: {e}\n" + traceback.format_exc())

            try:
                stream = None
                with connect(uri) as websocket:
                    # Send sample rate to the server
                    websocket.send(f"samplerate:{sample_rate}")

                    stream = p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=sample_rate,
                                      input=True,
                                      frames_per_buffer=chunk_size)

                    while self.running:
                        try:
                            data = stream.read(chunk_size)
                            send_audio_data(data)
                        except websockets.exceptions.ConnectionClosedOK:
                            self._print("Audio WebSocket connection closed normally.")
                            break
                        except websockets.exceptions.ConnectionClosedError as e:
                            self._print(f"Audio WebSocket connection closed with error: {e}")
                            break
                        except Exception as e:
                            self._print(f"An error occurred while recording audio: {e}")
                            break
            except Exception as e:
                self._print(f"Failed to connect to Audio WebSocket: {e}\n" + traceback.format_exc())
            finally:
                if stream:
                    stream.stop_stream()
                    stream.close()
                p.terminate()

        self._audio_thread = threading.Thread(target=record_audio_loop, daemon=True)
        self._audio_thread.start()


    def input_loop(self) -> None:
        """Read-eval-print loop for the client."""
        while self.running:
            try:
                w = input()
                if w == "/quit":
                    self.shutdown()
                else:
                    self.write_line(w)
            except EOFError:
                self.shutdown()
        return
        
