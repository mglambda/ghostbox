#!/usr/bin/env python
import unittest
from collections import deque
import threading, time, json
from ghostbox.commands import *

import ghostbox

class TTSTest(unittest.TestCase):

    def test_speak(self):
        time.sleep(20)
        box = ghostbox.from_generic(character_folder="test_dolphin",
                                     tts = True,
                                     verbose=True)
        
        w = box.text("Hi how are you?")
        print("You should hear:" + w)
        time.sleep(30)
        print(ttsDebug(box._plumbing, []))
        box._plumbing.tts.close()
            
    def test_speak_async(self):
        # we need to wait for the gpu to release memory
        time.sleep(10)
        box = ghostbox.from_generic(character_folder="test_dolphin",
                                     tts = True,
                                     verbose=True)
                                    
        def f(w):
            print("You should hear:" + w)
        
        box.text_async("Hi how are you?",
                       callback=f)
        time.sleep(30)
        print(ttsDebug(box._plumbing, []))        
        box._plumbing.tts.close()


    def test_speak_stream(self):
        # we need to wait for the gpu to release memory
        time.sleep(10)
        box = ghostbox.from_generic(character_folder="test_dolphin",
                                     tts = True,
                                     verbose=True)
                                    
        def f(w):
            print("You should hear:" + w)

        def g(token):
            print(token, end="", flush=True)

        box.text_stream("Hi how are you?",
                       chunk_callback=g,
                       generation_callback=f)
        time.sleep(30)
        print(ttsDebug(box._plumbing, []))        
        box._plumbing.tts.close()
        
    def test_interrupt(self):
        time.sleep(10)
        box = ghostbox.from_generic(character_folder="test_dolphin",
                                     tts = True,
                                     verbose=True)

        w = "This is an extremely long sentence that should be interrupted after about 5 seconds. It was handcrafted to be extremely long and obnoxious, and it will be repeated 3 times unless I get interrupted."
        done = threading.Event()
        done.clear()
        def stop_this(w):
            return w


            
        box.text_async(w, callback=stop_this)

        time.sleep(20)
        box.tts_stop()
        box._plumbing.tts.close()


    def test_orpheus_stream(self):
        time.sleep(10)
        box = ghostbox.from_generic(character_folder="test_dolphin",
                                     tts = True,
                                    tts_model="orpheus",
                                     verbose=True)

        time.sleep(5)
        w = "Hey, can you give an example sentence saying that you are demonstrating the orpheus TTS model."
        box.text_stream(w,
                        chunk_callback= lambda w: None,
                        generation_callback= lambda w: None)

        box.tts_wait()
        box._plumbing.tts.close()



def main():
    unittest.main()


        
if __name__=="__main__":
    print("Make sure you have headphones for these tests. These kind of require a human listener. Also, grab a coffee, they take a while.") 
    main()
    
        
        
