#!/usr/bin/env python
import unittest, os
from collections import deque
import threading, time, json
from ghostbox.commands import *

import ghostbox


common = {"quiet":True}
envvar_prompt = "TEST_TOOL_PROMPT"
# prompt will determine the tool used
if (prompt := os.environ.get(envvar_prompt, None)) is None:
    prompt = "Can you look up the latest news about berlin?"
    print("You can supply a prompt to this test by doing:\n`export " + envvar_prompt + "=...`\n. It will determine the tool used.\nUsing default prompt: " + prompt)
else:
    print("Using prompt: " + prompt)

def debug_info(box):
    import json
    print("request info:" + json.dumps(box._plumbing.getBackend().getLastRequest())        )
    print("result info:" + json.dumps(box._plumbing.getBackend().getLastJSON()))    

class ToolTest(unittest.TestCase):


    def test_tools_llama_streaming(self):
        box = ghostbox.from_llamacpp(character_folder="test_butterscotch",
                                     prompt_format="auto",
                                     **common)

        done = threading.Event()
        result = ""
        
        def f(w):
            nonlocal result
            nonlocal done
            print("DEBUG: "+ str(len(w)))            
            result = w
            done.set()
            
        box.text_stream(prompt,
                        chunk_callback=lambda w: w,
                        generation_callback=f)
        done.wait()

        try:
            self.assertGreater(len(result), 0)
        except AssertionError as e:
            debug_info(box)
            raise e

    def test_tools_llama(self):
        box = ghostbox.from_llamacpp(character_folder="test_butterscotch",
                                     prompt_format="auto",
                                     **common)

        print("DEBUG: " + box._plumbing.getBackend().getName())
        result = box.text(prompt)

        try:
            self.assertGreater(len(result), 0)
        except AssertionError as e:
            debug_info(box)
            raise e
        



    def test_tools_generic(self):
        box = ghostbox.from_generic(character_folder="test_butterscotch",
                                    force_params=True,
                                    **common)
        result = box.text(prompt)

        try:
            self.assertGreater(len(result), 0)
        except AssertionError as e:
            debug_info(box)
            raise e

    def test_tools_generic_streaming(self):
        box = ghostbox.from_generic(character_folder="test_butterscotch",
                                    force_params=True,
                                    **common)
        done = threading.Event()
        result = ""
        
        def f(w):
            nonlocal result
            nonlocal done
            print("DEBUG: " + str(len(w)))
            result = w
            done.set()
            
        box.text_stream(prompt,
                        chunk_callback=lambda w: w,
                        generation_callback=f)
        done.wait()
        try:
            self.assertGreater(len(result), 0)
        except AssertionError as e:
            debug_info(box)
            raise e

def main():
    unittest.main()

if __name__=="__main__":
    main()
    
