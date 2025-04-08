from typing import *
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from queue import Queue, Empty
import websockets
from websockets.sync.client import connect, ClientConnection
import threading, time, sys, json, traceback
from ghostbox.util import printerr, get_default_microphone_sample_rate

@dataclass
class RemoteMsg:
    text: str
    is_stderr: bool = False

_remote_info_token = "RemoteInfo: "

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

    def _start_tts_client(self) -> None:
        """Starts a websock client that connects to the remote TTS host and launches a handler loop that plays audio on the local machine."""
        import pyaudio
        import wave
        import numpy as np

        p = pyaudio.PyAudio()
        stream = None
        buffer = bytearray()

        info = self._remote_info
        if not info or not info.tts_websock:
            self._print("TTS service not available.")
            return

        uri = f"ws://{info.tts_websock_host}:{info.tts_websock_port}"
        self._print(f"Connecting to TTS WebSocket at {uri}.")

        def handle_tts_msgs_loop():
            while self.running:
                try:
                    if self._tts_websocket is None:
                        self._tts_websocket = connect(uri)
                        
                    data = self._tts_websocket.recv()
                    if data == "stop":
                        self._print("Received stop signal.")
                        self._tts_stop_flag.set()
                    else:
                        self._print("Queueing data.")
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
            chunk_size = 1024            
            nonlocal stream
            if not stream:
                stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=24000,
                                output=True)
            #stream.write(np.frombuffer(buffer, dtype=np.int16))
                
            while self.running:
                try:
                    data = self._tts_play_queue.get(timeout=1)
                except Empty:
                    continue

                stream.start_stream()
                for i in range(0, len(data), chunk_size):
                    if self._tts_stop_flag.is_set():
                        break
                    chunk = data[i:i + chunk_size]
                    stream.write(chunk)
                self._tts_stop_flag.clear()
                stream.stop_stream()
                self._tts_websocket.send("done")

        self._tts_playback_thread = threading.Thread(target=playback_loop, daemon=True)
        self._tts_playback_thread.start()

        
    def _start_audio_client(self) -> None:
        """Starts a websock client that connects to the remote audio transcription service and launches a handler loop that records from the local microphone and sends audio data to the remote endpoint."""
        info = self._remote_info
        if not info or not info.audio_websock:
            self._print("Audio transcription service not available.")
            return

        uri = f"ws://{info.audio_websock_host}:{info.audio_websock_port}"
        self._print(f"Connecting to Audio WebSocket at {uri}.")

        def record_audio_loop():
            import pyaudio
            import wave
            # FIXME: so on some systems opening pyaudio will vomit a bunch of irrelevant error messages into sdterr
            # we can turn this off but that will sometimes supress other stderr messages
            # I don't wanna do that since the client is still under heavy development, we're just going to eat the errors for now
            p = pyaudio.PyAudio()
            sample_rate = int(rate) if (rate := get_default_microphone_sample_rate(p)) is not None else 16000
            chunk_size = 1024

            def send_audio_data(data):
                try:
                    websocket.send(data)
                except websockets.exceptions.ConnectionClosedError as e:
                    self._print(f"Audio WebSocket connection closed with error: {e}")
                except Exception as e:
                    self._print(f"An error occurred while sending audio data: {e}")

            try:
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
                self._print(f"Failed to connect to Audio WebSocket: {e}")
            finally:
                if stream:
                    stream.stop_stream()
                    stream.close()
                p.terminate()

        self._audio_thread = threading.Thread(target=record_audio_loop, daemon=True)
        self._audio_thread.start()

