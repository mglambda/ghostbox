import unittest
from collections import deque
import threading, os

import ghostbox

class LlamacppTest(unittest.TestCase):
    box = ghostbox.from_llamacpp()

    def test_box_optionsrom(self):
        # make extra box and see if setting options through **kwargs works
        x = ghostbox.from_llamacpp(specific_option="test")
        
        self.assertEqual(x.specific_option, "test")
        self.assertEqual(x._plumbing.options["specific_option"], "test")
        self.assertEqual(x._plumbing.getOption("specific_option"), "test")

        new = "another test"
        x.specific_option = new
        self.assertEqual(x.specific_option, new)
        self.assertEqual(x._plumbing.options["specific_option"], new)
        self.assertEqual(x._plumbing.getOption("specific_option"), new)


    def test_start_session(self,):
        bx = ghostbox.from_llamacpp(character_folder="test_dolphin")
        w = bx.text("Hi dolphin!")
        dolphin = "Dolphin"
        self.assertEqual(bx._plumbing.getOption("chat_ai"), dolphin)
        self.assertEqual(bx.chat_ai, dolphin)
        self.assertEqual(bx.get("chat_ai"), dolphin)
        self.assertEqual(type(w), str)
        
        bx.start_session("test_joshu")
        w = bx.text("Hi Joshu!")
        joshu = "Joshu"
        self.assertEqual(bx._plumbing.getOption("chat_ai"), joshu)
        self.assertEqual(bx.chat_ai, joshu)
        self.assertEqual(bx.get("chat_ai"), joshu)
        self.assertEqual(type(w), str)        
    
    def test_text_basic(self):
        # this has no character set so it doesn't need to return anything sensible. however it does need to return
        w = self.box.text("Hello, how are you!")
        self.assertEqual(type(w), str)
        return



    def test_text_async_basic(self):
        tmp = None
        done_flag = threading.Event()
        
        def f(v):
            nonlocal tmp
            nonlocal done_flag
            tmp = v
            done_flag.set()
        
        self.box.text_async("Hello, how are you!",
                      callback=f)

        done_flag.wait()
        self.assertEqual(type(tmp), str)
        return


    def test_text_stream_basic(self):
        chunks = deque()
        tmp = ""
        done_flag = threading.Event()
        
        def process_chunk(v):
            nonlocal chunks
            chunks.append(v)


        def process_generation(v):
            nonlocal tmp
            nonlocal done_flag
            tmp = v
            done_flag.set()
            
        self.box.text_stream("Hello, how are you!",
                             chunk_callback=process_chunk,
                             generation_callback=process_generation)
        done_flag.wait()
        self.assertGreater(len(chunks), 0)
        self.assertEqual(type(tmp), str)
        self.assertEqual(tmp, "".join(chunks))
        return

    def test_dependency_injection(self):
        bx = ghostbox.from_llamacpp(character_folder="test_scribe_dependency_injection")
        # scribe doesn't have 'file' available, we will tell him what his note file is
        test_file = "this_is_a_test_file_for_dependency_injection.txt"
        bx.tools_inject_dependency("file", test_file)
        bx.text("Can you take down a note that reminds me to work out next week?")
        self.assertTrue(os.path.isfile(test_file))
        os.remove(test_file)
        
def main():
    unittest.main()

if __name__=="__main__":
    main()
    
