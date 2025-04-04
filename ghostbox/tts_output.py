import traceback, threading, wave, time
import numpy
from abc import ABC, abstractmethod
from functools import *
from typing import *
from queue import Queue, Empty
from ghostbox.util import *
from ghostbox.definitions import *


class TTSOutput(ABC):
    """Manages output of TTS sound."""

    @abstractmethod
    def __init__(self, volume: float= 1.0, **kwargs):
        self.volume = volume

    @abstractmethod
    def enqueue(self, payload: str | Iterator[bytes]) -> None:
        """Enqeueu a wave file or generator for playback. Start playback immediately if nothing is playing.
        This function is non-blocking.
        :param payload: Either a string denoting the filename of a wave file, or a generator yielding wave audio data."""
        pass


    def set_volume(factor: float = 1.0) -> None:
        """Set the volume for the output module.
        :param factor: A multiplier that will be applied to the base volume. 1.0 is no change to the base."""
        self.volume = factor
        
    
    def stop(self) -> None:
        """Instantly interrups and stops any ongoing playback. This method is thread safe."""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """Returns true if the output module is currently playing sound."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully shut down output module, finishing playback of all enqueued files.
        Calling any of the methods after this one is undefined behaviour."""
        pass


class DefaultTTSOutput(TTSOutput):
    """Local TTS sound output using pyaudio."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # fail early if pyaudio isn't available
        import pyaudio

        printerr("Using pyaudio for local playback.")
        self.stop_flag = threading.Event()
        self._queue = Queue()
        self.pyaudio = pyaudio.PyAudio()
        self.running = True
        ### streaming stuff
        # used for streaming only, playing files opens its own pyaudio stream using the wav files parameters

        # we use yet another queue and buffer
        # that will contain the actual raw audio for streaming mode
        self._audio_queue = Queue()
        
        # this is the actual stream object we will write to
        # init_stream will start a worker
        # and set self._stream
        self._init_stream()
        
        def play_worker():
            while self.running or not (self._queue.empty()):
                # don't remove the loop or timeout or else thread will not be terminated through signals
                try:
                    payload = self._queue.get(timeout=1)
                except Empty:
                    continue

                # _play will block until stop is called or playback finishes
                self.stop_flag.clear()                
                self._play(payload)

        self.worker = threading.Thread(target=play_worker)
        self.worker.start()


    def _init_stream(self):
        """Initializes a stream for streaming playback."""
        chunk = 512
        # FIXME: hardcoded some stuff as we are trying out streaming with orpheus first
        p = self.pyaudio
        self._stream = p.open(
            format=p.get_format_from_width(2),
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=chunk,
        )
        self._stream.start_stream()
        # so atm we sometimes get buffer underruns
        # this is on a rtx 3090 with let's say 150is t/s
        # it tends to happen at the start of generation, so this is an experimental fix for that particular situation
        # we don't start streaming until we have n number of audio chunks prebuffered
        n_prebuffer = 3
        
        def stream_worker():
            skip_prebuffer = False
            while self.running:
                if len(self._audio_queue.queue) < n_prebuffer and not(skip_prebuffer):
                    time.sleep(0.1)
                    continue
                else:
                    skip_prebuffer = True                    
                    
                try:
                    chunk = self._audio_queue.get(timeout=1)
                except Empty:
                    skip_prebuffer = False
                    continue


                # FIXME: this doesn't work
                #if self.volume != 1.0:
                    #chunk_np = numpy.fromstring(chunk, numpy.int16) * self.volume
                    #chunk = chunk_np.astype(numpy.int16)

                self._stream.write(chunk)

        self._stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self._stream_thread.start()

    def _stream_write(self, audio_chunk) -> None:
        """Small helper to play audio chunks in streaming mode."""
        self._audio_queue.put(audio_chunk)
        
    def enqueue(self, payload) -> None:
        self._queue.put(payload)

    def _play(self, payload: str | Iterator[bytes], **kwargs) -> None:
        if type(payload) == str:
            self._play_file(payload, **kwargs)
        else:
            self._play_stream(payload, **kwargs)
            
    def _play_file(self, filename: str) -> None:
        import pyaudio
        wf = wave.open(filename, "rb")
        chunk = 1024
        p = self.pyaudio
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            frames_per_buffer=chunk,
        )

        # Play the audio data
        while data := wf.readframes(chunk):
            if self.stop_flag.isSet():
                break

            stream.write(data)

        stream.stop_stream()
        stream.close()
        self.stop_flag.set()



    def _play_stream(self, audio_stream: Iterator[bytes]) -> None:
        # Play the audio data
        for data in audio_stream:
            if self.stop_flag.isSet():
                break
            self._stream_write(data)

        #self._stream.stop_stream()
        self.stop_flag.set()
        
    def stop(self) -> None:
        """Instantly interrupts and stops all playback."""
        self.stop_flag.set()
        self._queue.queue.clear()
        self._audio_queue.queue.clear()        

    def is_speaking(self) -> bool:
        #print(f"{self._queue.empty()=}")
        #print(f"{self.stop_flag.is_set()=}")
        return not(self._queue.empty()) or not(self._audio_queue.empty()) or not(self.stop_flag.is_set())

    def shutdown(self) -> None:
        """Gracefully shut down output module, finishing playback of all enqueued files."""
        self.running = False
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
        self._queue = Queue()
        self.host = host
        self.port = port

        # start websock server
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        printerr("WebSocket TTS output initialized.")

        def play_worker():
            while self.server_running.isSet() or not (self._queue.empty()):
                # don't remove the loop or timeout or else thread will not be terminated through signals
                try:
                    filename = self._queue.get(timeout=1)
                except Empty:
                    continue

                # _play will block until stop is called or playback finishes
                self._play(filename)

        # start queue worker
        self.worker_thread = threading.Thread(target=play_worker)
        # deliberately not a daemon so we deal with EOF properly
        self.worker_thread.start()

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
                    # print(msg)
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

    def enqueue(self, filename: str) -> None:
        """Send a wave file   over the network, or enqueue it to be sent if busy.
        This method is non-blocking."""
        self._queue.put(filename)

    def _play(self, filename: str) -> None:
        """Sends a wave file ofer the network to all connected sockets."""
        from websockets import ConnectionClosedError

        printerr("[WEBSOCK] Playing with " + str(len(self.clients)) + " clients.")

        with open(filename, "rb") as wf:
            # Read the entire file content
            # data = wf.readframes(wf.getnframes())
            data = wf.read()

            for client in self.clients:
                printerr(
                    "[WEBSOCK] Playing audio to "
                    + str(client.remote_address)
                    + " with file "
                    + filename
                )
                try:
                    client.send(data, text=False)
                except ConnectionClosedError:
                    print(
                        "[WEBSOCK] error: Unable to send data to "
                        + str(client.remote_address)
                        + ": Connection closed."
                    )
        # we block now, just as if we were playing the sound waiting for it to finish
        self.go_flag.clear()
        while True:
            # FIXME: this doesn't work 100% correctly with multiple clients (it might be ok though), but that is not the intended use case
            if self.go_flag.isSet():
                break
            time.sleep(0.1)

    def stop(self) -> None:
        """Instantly stop playback and clears the queue."""
        self._queue.queue.clear()
        self.stop_flag.set()
        self.go_flag.clear()
        for client in self.clients:
            client.close()
        printerr("[WEBSOCK] TTS output stopped.")

    def is_speaking(self) -> bool:
        return self.go_flag.is_set()
        
    def shutdown(self) -> None:
        """Shuts down the output module gracefully, waiting for playback/sending of all enqueue files to be finished."""
        while not (self._queue.empty()):
            time.sleep(0.1)
        self.stop()
        self._stop_server()
        super().shutdown()
