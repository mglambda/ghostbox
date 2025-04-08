from typing import *
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from queue import Queue, Empty
import websockets
from websockets.sync.client import connect, ClientConnection
import threading, time, sys, json, traceback
from ghostbox.util import printerr

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
    stdout_callback: Callable[[str], None] = lambda w: print(w, end="")
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
    
    _websocket: Optional[ClientConnection] = None
    

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
            

        if info.audio and info.audio_websock:
            self._print("Audio transcription service active.")
            no_services = False

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
        except websockets.exceptions.ConnectionRefusedError as e:
            self._print(f"Connection refused: {e}")
            time.sleep(3)
        except Exception as e:
            self._print(f"An error occurred while connecting: {e}")
            time.sleep(3)
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
        
        
