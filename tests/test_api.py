import unittest
from collections import deque
import threading

import ghostbox

class LlamacppTest(unittest.TestCase):
    box = ghostbox.from_llamacpp()    
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
    
def main():
    unittest.main()

if __name__=="__main__":
    main()
    
