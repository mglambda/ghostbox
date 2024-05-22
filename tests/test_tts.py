import unittest
from collections import deque
import threading, time, json
from ghostbox.commands import *

import ghostbox

class TTSTest(unittest.TestCase):

    def test_speak(self):
        box = ghostbox.from_llamacpp(character_folder="test_dolphin",
                                     tts = True,
                                     tts_voice_dir="voices",
                                     tts_program="/home/marius/prog/ai/ghostbox/tts.sh",
                                     verbose=True,
                                     include=["/home/marius/prog/ai/ghostbox"])
        w = box.text("Hi how are you?")
        print("You should hear:" + w)
        time.sleep(30)
        print(ttsDebug(box._plumbing, []))
        box._plumbing.tts.close()
            
    def test_speak_async(self):
        # we need to wait for the gpu to release memory
        time.sleep(10)
        box = ghostbox.from_llamacpp(character_folder="test_dolphin",
                                     tts = True,
                                     tts_voice_dir="voices",
                                     tts_program="/home/marius/prog/ai/ghostbox/tts.sh",
                                     verbose=True,
                                     include=["/home/marius/prog/ai/ghostbox"])
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
        box = ghostbox.from_llamacpp(character_folder="test_dolphin",
                                     tts = True,
                                     tts_voice_dir="voices",
                                     tts_program="/home/marius/prog/ai/ghostbox/tts.sh",
                                     verbose=True,
                                     include=["/home/marius/prog/ai/ghostbox"])        
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
        
        
        
def main():
    unittest.main()

if __name__=="__main__":
    main()
    
