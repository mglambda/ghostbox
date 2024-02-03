import os, appdirs, threading, time, sys
from ghostbox.util import *
import operator
from stat import ST_MTIME

def mostRecentFile(path):
    """Returns pair of most recent file in directory PATH, as well as its modification date. Returns ("", 0) if path is not a directory."""
    if not(os.path.isdir(path)):
        return ("", 0)
    
    
    all_files = os.listdir(path);
    file_mtimes = dict();
    for file in all_files:
        e = path + "/" + file
        file_mtimes[e] = time.time() - os.stat(e).st_mtime;
    winner =  sorted(file_mtimes.items(), key=operator.itemgetter(1))[0][0]
    return (winner, os.stat(winner).st_mtime)




class AutoImageProvider(object):
    """When instanciated, takes a directory and scans it periodically for a new image file. Once a new file is detected, the AutoImageProvider executes a callback with the image data as argument.
This allows people to take screenshots and have them automatically be described, without having to further input anything."""
    def __init__(self, watch_dir, on_new_image_func, update_period=0.25, image_id=0):
        """watch_dir - Directory that will be periodically checked for new images.
on_new_image_func - Function that takes one argument, a dictionary with keys 'data' and 'id'. This is in the format expected by Llamacpp.
        update_period - How often to check the directory.
image_id - The id of the image. This is relevant as it will inform the tokens used to invoke the image, e.g. with id=0 it would be '[img-0]'."""
        self.watch_dir = watch_dir
        self.update_period = update_period
        self.image_id = image_id
        self.callback = on_new_image_func
        self.latestFile = mostRecentFile(self.watch_dir) # this is pair (filename, modtime)
        self.running = False
        self._initWatchLoop()

    def _initWatchLoop(self):
        self.running = True
        t = threading.Thread(target=self._watchLoop, args=[])
        t.start()

    def stop(self):
        self.running = False
            
    def _watchLoop(self):
        while self.running:
            if not(os.path.isdir(self.watch_dir)):
                printerr("warning: AutoImageProvider cannot watch directory '" + self.watch_dir + "': Not a directory. Halting watch loop.")
                self.stop()
            else:
                (file, modtime) = mostRecentFile(self.watch_dir)
                if modtime > self.latestFile[1] and isImageFile(file):
                    self.latestFile = (file, modtime)
                    self.invokeCallback()
            time.sleep(self.update_period)
            
    def invokeCallback(self):
        (file, time) = self.latestFile
        self.callback(file, self.image_id)
                
