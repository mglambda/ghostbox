import unittest
from collections import deque
import threading, os, time
from pydantic import BaseModel
import json
from typing import *
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

    def test_start_session(
        self,
    ):
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

        self.box.text_async("Hello, how are you!", callback=f)

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

        self.box.text_stream(
            "Hello, how are you!",
            chunk_callback=process_chunk,
            generation_callback=process_generation,
        )
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

    def test_tts(self):
        box = ghostbox.from_llamacpp(
            include=["/home/marius/prog/ai/ghostbox"],
            character_folder="test_dolphin",
            tts=True,
            quiet=False,
            tts_model="kokoro",
            tts_voice="af_sky"
        )
        box.tts_say("This is an API use of the TTS.")
        # give it time to actually talk
        time.sleep(10)

    def test_set_char(self):
        history = [
            {
                "role": "user",
                "content": "Hey I just bought a flamingo at the bird store.",
            },
            {"role": "assistant", "content": "Really? What color is it?"},
            {"role": "user", "content": "Believe it or not, it is a green flamingo."},
            {
                "role": "assistant",
                "content": "Fascinating. It must really stand out among the other flamingos.",
            },
        ]
        self.box.set_char("joshu", chat_history=history)
        w = self.box.text(
            "Have you been paying attention? Please repeat to me what color my flamingo is."
        )
        self.assertEqual(type(w), str)
        print("The following response should talk about a flamingo being green:\n" + w+ "\n")
        return

    def test_json(self):
        box = ghostbox.from_llamacpp(character_folder="test_dolphin")
        import json
        noises = json.loads(box.json("Generate a dictionary of animals and their typical noises."))
        try:
            self.assertTrue(type(noises) == type({}))
        except AssertionError as e:
            print("NOISES: " + json.dumps(noises, indent=4))
            raise e

    def test_json_async(self):
        box = ghostbox.from_llamacpp(character_folder="test_dolphin")
        import json
        done = threading.Event()
        noises = None
        def f(w):
            nonlocal noises
            nonlocal done
            noises = json.loads(w)
            done.set()
            
        box.json_async("Generate a dictionary of animals and their typical noises.",
                       callback=f)
        done.wait()
        try:
            self.assertTrue(type(noises) == type({}))
        except AssertionError as e:
            print("NOISES: " + json.dumps(noises, indent=4))
            raise e

        
    def test_json_schema(self):
        box = ghostbox.from_llamacpp(character_folder="test_dolphin")

        class Animal(BaseModel):
            """A member of the animal kingdom."""
            animal: str
            cute_name: str
            number_of_legs: int
            typical_noise: str
            favorite_foods: List[str]
            
        cat = json.loads(box.json("Please provide data describing the typical cat.",
                                     schema=Animal.model_json_schema()))
        print(json.dumps(cat, indent=4))        
        self.assertTrue(cat["animal"].lower() == "cat")

def main():
    unittest.main()


if __name__ == "__main__":
    main()
