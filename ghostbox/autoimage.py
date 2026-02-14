import os, appdirs, threading, time, sys
from .util import *
import operator
from stat import ST_MTIME
from typing import Tuple, Dict, List, Callable, Any

def mostRecentFile(path: str) -> Tuple[str, float]:
    """Returns pair of most recent file in directory PATH, as well as its modification date. Returns ("", 0) if path is not a directory."""
    if not os.path.isdir(path):
        return ("", 0.0)
    
    all_files: List[str] = os.listdir(path)
    file_mtimes: Dict[str, float] = {}
    for file in all_files:
        e: str = os.path.join(path, file)
        file_mtimes[e] = time.time() - os.stat(e).st_mtime
    if not file_mtimes:
        return ("", 0.0)
    winner: str = sorted(file_mtimes.items(), key=operator.itemgetter(1))[0][0]
    return (winner, os.stat(winner).st_mtime)




class AutoImageProvider(object):
    """When instanciated, takes a directory and scans it periodically for a new image file. Once a new file is detected, the AutoImageProvider executes a callback with the image data as argument.
This allows people to take screenshots and have them automatically be described, without having to further input anything."""
    def __init__(self, watch_dir: str, on_new_image_func: Callable[[str, int], Any], update_period: float = 0.25, image_id: int = 0) -> None:
        """watch_dir - Directory that will be periodically checked for new images.
on_new_image_func - Function that takes one argument, a dictionary with keys 'data' and 'id'. This is in the format expected by Llamacpp.
        update_period - How often to check the directory.
image_id - The id of the image. This is relevant as it will inform the tokens used to invoke the image, e.g. with id=0 it would be '[img-0]'."""
        self.watch_dir: str = watch_dir
        self.update_period: float = update_period
        self.image_id: int = image_id
        self.callback: Callable[[str, int], Any] = on_new_image_func
        self.latestFile: Tuple[str, float] = mostRecentFile(self.watch_dir) # this is pair (filename, modtime)
        self.running: bool = False
        self._initWatchLoop()

    def _initWatchLoop(self) -> None:
        self.running = True
        t: threading.Thread = threading.Thread(target=self._watchLoop, args=[])
        t.start()

    def stop(self) -> None:
        self.running = False
            
    def _watchLoop(self) -> None:
        while self.running:
            if not os.path.isdir(self.watch_dir):
                printerr("warning: AutoImageProvider cannot watch directory '" + self.watch_dir + "': Not a directory. Halting watch loop.")
                self.stop()
            else:
                (file, modtime) = mostRecentFile(self.watch_dir)
                if modtime > self.latestFile[1] and isImageFile(file):
                    self.latestFile = (file, modtime)
                    self.invokeCallback()
            time.sleep(self.update_period)
            
    def invokeCallback(self) -> None:
        (file, _time) = self.latestFile
        self.callback(file, self.image_id)
