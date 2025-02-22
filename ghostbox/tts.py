#!/usr/bin/env python
import argparse, traceback

parser = argparse.ArgumentParser(description="tts.py - TTS program to consume text from stdin and speak it out/ save it as wav file.")
parser.add_argument("-f", '--filepath', type=str, default="", help="Filename to save accumulated spoken lines in. Output is in wav format.")
parser.add_argument("-q", "--quiet", action=argparse.BooleanOptionalAction, default=False, help="Do not play any audio.")
parser.add_argument("-p", "--pause_duration", type=int, default=1, help="Duration of pauses after newlines. A value of 0 means no or minimal-duration pause.")
parser.add_argument("-y", "--voice_sample", type=str, default="cloning.wav", help="Path to wav file used as a voice sample to clone.")
parser.add_argument("-i", "--volume", type=float, default=1.0, help="Volume for the voice playback.")
parser.add_argument("--sound_dir", type=str, default="sounds", help="Directory where sound files are located to be played with #sound <SNDNAME>")
parser.add_argument("-m", "--model", type=str, default="xtts", help="Text-to-speech model to use.")
args = parser.parse_args()

from ghostbox.tts_util import *
from ghostbox.tts_state import *
from ghostbox.tts_backends import *
import time, threading, os
prog = TTSState(args)
tts = initTTS(prog.args.model)

# do pygame here because of wonderful hello from pygame message
import pygame
pygame.mixer.init()

# ok we are hacking this to allow stopping of all sounds
from queue import Queue, Empty
msg_queue = Queue()
done = threading.Event()
def input_loop():
    global done
    while True:
        try:
            w = input()
            if w == "<clear>":
                pygame.mixer.stop()
                with msg_queue.mutex:
                    msg_queue.queue.clear()
                prog.clearRetries()
                continue
            msg_queue.put(w)
        except:# EOFError as e:
            print("Exception caught while blocking. Shutting down gracefully. Below is the full exception.")
            print(traceback.format_exc())                    
            prog.handleMixins()
            #print("EOF encountered. Closing up.")
            time.sleep(3)
            #os._exit(1)
            done.set()

t = threading.Thread(target=input_loop)
t.daemon = True
t.start()

while True:
    if done.is_set():
        break
    
    try:
        if prog.isRetrying():
            rawmsg = prog.popRetryMSG()
        else:
            # so fun fact
            # Queue.get blocks. you knew that, ofc
            # but did you know that it super blocks? that's right - it refuses to handle any signals send to the application, including sigint and sigkill
            # so we have to sporadically use a timeout and loop around. btw all of this is undocumented.
            # Thanks, Guido!
            rawmsg = msg_queue.get(timeout=1)
    except Empty:
        # timeout was hit
        continue
    except: #EOFError as e:
        print("Exception caught while blocking. Shutting down gracefully. Below is the full exception.")
        print(traceback.format_exc())        
        prog.handleMixins()
        #print("EOF encountered. Closing up.")
        time.sleep(3)
        os._exit(1)

    (msg, cont, err) = prog.processMsg(rawmsg)
    print(err)
    if cont:
        continue
#    xs = tts.split_into_sentences(msg)
#    print(str(len(xs)))
    try:
        tts.tts_to_file(text=msg, speaker_wav=prog.getVoiceSampleFile(), language="en", file_path="output.wav")
    except ZeroDivisionError:
        print("Caught zero division error. Ignoring.")
        # this happens when the tts is asked to process whitespace and produces a wav file in 0 seconds :) nothing to worry about
        continue
    except AssertionError as e:
        print(str(e) + "\nwarning: Caught assertion error on msg: " + msg)
        prog.retry(msg)
        continue # we retry the msg that was too long
        
        
    prog.accumulateSound("output.wav")
    prog.addPause()
    if prog.args.quiet:
        continue
    snd = pygame.mixer.Sound("output.wav")
    while pygame.mixer.get_busy():
        pygame.time.delay(10) #ms

    snd.set_volume(prog.args.volume)
    snd.play()

    
