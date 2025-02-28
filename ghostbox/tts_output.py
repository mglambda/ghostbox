import traceback, threading, wave
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

    @abstractmethod
    def shutdown(self) -> None:
        """Shuts down the output module, allowing for any necessary cleanup. Calling any of the methods after this is undefined behaviour."""
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
        import pyaudio
        
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

    def shutdown(self) -> None:
        super().shutdown()
        
class WebsockTTSOutput(TTSOutput):
    def __init__(self, host="localhost", port=5052, **kwargs):
        # fail early        
        import websockets.sync.server
        
        super().__init__(**kwargs)
        self.clients = []
        self.stop_flag = threading.Event()
        self.go_flag = threading.Event()
        self.go_flag.set()
        self.server_running = threading.Event()
        self.host = host
        self.port = port
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        printerr("WebSocket TTS output initialized.")

    def _run_server(self):
        import websockets
        import websockets.sync.server as WS

        def handler(websocket):
            remote_address = websocket.remote_address
            printerr("[WEBSOCK] Got connection from " + str(remote_address))
            self.clients.append(websocket)
            try:
                while self.server_running.isSet():
                    msg = websocket.recv()
                    #print(msg)
                    if msg == "done":
                        # current sound has finished playing
                        self.go_flag.set()
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.clients.remove(websocket)
                # FIXME: this doesn't work with multiple clients. see play()
                self.go_flag.set()
                printerr("[WEBSOCK] Closed connection with " + str(remote_address))                

        self.server_running.set() 
        self.server = WS.serve(handler, self.host, self.port)
        printerr("WebSocket server running on ws://" + self.host + ":" + str(self.port))
        self.server.serve_forever()

    def stop_server(self) -> None:
        self.server_running.clear()
        printerr("Halting websocket server.")
        


    def play(self, filename: str, volume: float = 1.0) -> None:
        from websockets import ConnectionClosedError
        printerr("[WEBSOCK] Playing with " + str(len(self.clients)) + " clients.")

        with open(filename, 'rb') as wf:
            # Read the entire file content
            #data = wf.readframes(wf.getnframes())
            data = wf.read()

            for client in self.clients:
                printerr("[WEBSOCK] Playing audio to " + str(client.remote_address) + " with file " + filename)
                try:
                    client.send(data, text=False)
                except ConnectionClosedError:
                        print("[WEBSOCK] error: Unable to send data to " + str(client.remote_address) + ": Connection closed.")
        # we block now, just as if we were playing the sound waiting for it to finish
        self.go_flag.clear()
        while True:
            # FIXME: this doesn't work 100% correctly with multiple clients (it might be ok though), but that is not the intended use case
            if self.go_flag.isSet():
                break
            time.sleep(0.1)

    def stop(self) -> None:
        self.stop_flag.set()
        self.go_flag.clear()
        for client in self.clients:
            client.close()
        printerr("[WEBSOCK] TTS output stopped.")        

    def shutdown(self) -> None:
        self.stop_server()
        super().shutdown()
