#!/usr/bin/env python
import unittest
from typing import *
import pstats
import feedwater
import os, time

def prof_file():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghostbox-tts.prof")
    return

def stats_file(prefix):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ghostbox-tts-{prefix}.stats")

def log_file(prefix):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tts_{prefix}_times.log")

def get_previous_times(model: str) -> List[float]:
    try:
        with open(log_file(model), "r") as f:
            lines = f.read().split("\n")
    except FileNotFoundError:
        return []
    return [float(w) for w in lines if w != ""]
    
samples ="""Hey, what's up!
Hi, I'm commander sheperd, and this is my favorite store in the citadel!
Row, row, row your boat, gently down the stream.
Merrily, merrily merrily! Life is but a dream.
Woah.
The transcendental unity of apperception is the condition for the possibility of objective experience.
yo mamma so fat she got her own zip code""".split("\n")


def profile_model(ut, model):
    print(f"* {model}\n** execution times") 
    # don't mix up stats from different calls
    if os.path.isfile(prof_file()):
        os.remove(prof_file())
    
    cmd = f"ghostbox-tts -m {model} --profiling"
    print("running " + cmd)
    p = feedwater.run(cmd)
    # give it time to load the model
    time.sleep(10)
    print("Ok. Feeding samples to stdin.")
    t1 = time.time()
    p.write_line("<start_profiling>")
    for sample in samples:
        p.write_line(sample)
    p.write_line("<simulate_eof>")
    while p.is_running():
        time.sleep(0.01)
    # done
    t2 = time.time()
    delta = round(t2 - t1, 2)
    records = get_previous_times(model)
    print(f"## Previous times for {model}")
    for prev_time in records:
        print(str(prev_time))
    if records != []:
        print("--------")
        average = sum(records) / len(records)
        print(f"{average} seconds average.")        
    print(f"{delta} seconds to complete.")

    with open(log_file(model), "a") as f:
        f.write(f"{delta}\n")

    print("** stats")
    s = pstats.Stats('ghostbox-tts.prof')
    s.sort_stats('cumulative').print_stats(0.1)

    if records == []:
        return
        
    # make sure we complain if we regress on time by a lot
    # but leave some wiggle room
    lower_bound = 0.9 * average
    ut.assertGreater(delta, lower_bound)
    
    
    
    
class TTSTest(unittest.TestCase):
    def test_tts_orpheus(self):
        profile_model(self, "orpheus")
        
def main():
    unittest.main()

if __name__=="__main__":
    main()
    
        
        
