#!/usr/bin/env python
import unittest
from collections import deque
import threading, time, json
from ghostbox.commands import *

import ghostbox

class ToolTest(unittest.TestCase):

    def test_tools_llama(self):
        box = ghostbox.from_llamacpp(character_folder="test_butterscotch",
                                     prompt_format="auto")

        time.sleep(3)
        w = box.text("Can you look up the latest news about berlin?")
        self.assertGreater(len(w), 0)
        # not to hammer the search API
        time.sleep(5)        

    def test_tools_llama_streaming(self):
        box = ghostbox.from_llamacpp(character_folder="test_butterscotch",
                                     prompt_format="auto")

        time.sleep(3)        
        done = threading.Event()
        result = ""
        
        def f(w):
            result = w
            done.set()
            
        box.text_stream("Can you look up the latest news about berlin?",
                        chunk_callback=lambda w: w,
                        generation_callback=f)
        done.wait()
        assertGreater(len(result), 0)
        # not to hammer the search API
        time.sleep(5)        


    def test_tools_generic(self):
        box = ghostbox.from_generic(character_folder="test_butterscotch",
                                    force_params=True)
        w = box.text("Can you look up the latest news about berlin?")
        self.assertGreater(len(w), 0)
        # not to hammer the search API
        time.sleep(5)        

    def test_tools_generic_streaming(self):
        box = ghostbox.from_generic(character_folder="test_butterscotch",
                                    force_params=True)
        done = threading.Event()
        result = ""
        
        def f(w):
            result = w
            done.set()
            
        box.text_stream("Can you look up the latest news about berlin?",
                        chunk_callback=lambda w: w,
                        generation_callback=f)
        done.wait()
        assertGreater(len(result), 0)
        # not to hammer the search API
        time.sleep(5)        
        
        
def main():
    unittest.main()

if __name__=="__main__":
    main()
    
