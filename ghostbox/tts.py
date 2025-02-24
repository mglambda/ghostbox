#!/usr/bin/env python
import argparse, traceback, sys, tempfile, ast

program_name = sys.argv[0]
parser = argparse.ArgumentParser(description= program_name + " - TTS program to consume text from stdin and speak it out/ save it as wav file.")
parser.add_argument("-f", '--filepath', type=str, default="", help="Filename to save accumulated spoken lines in. Output is in wav format.")
parser.add_argument("--voices", action=argparse.BooleanOptionalAction, default=False, help="List all available voices for chosen model, then exit the program.")
parser.add_argument("-q", "--quiet", action=argparse.BooleanOptionalAction, default=False, help="Do not play any audio.")
parser.add_argument("-l", "--language", type=str, default="en", help="Language for the TTS output. Not all TTS models support all language, and many don't need this option.")
parser.add_argument("-p", "--pause_duration", type=int, default=1, help="Duration of pauses after newlines. A value of 0 means no or minimal-duration pause.")
parser.add_argument("-y", "--voice_sample", type=str, default="cloning.wav", help="Path to wav file used as a voice sample to clone.")
parser.add_argument("-i", "--volume", type=float, default=1.0, help="Volume for the voice playback.")
parser.add_argument("-s", "--seed", type=int, default=420, help="Random seed for voice models that use it.")
parser.add_argument("--sound_dir", type=str, default="sounds", help="Directory where sound files are located to be played with #sound <SNDNAME>")
parser.add_argument("-m", "--model", type=str, default="zonos", help="Text-to-speech model to use.")

# zonos specific
parser.add_argument("--zonos_model", type=str, default="hybrid", help="The pretrained checkpoint to use with the Zonos TTS engine. Try picking 'transformer' or 'hybrid' for good defaults, otherwise consult the zonos project for more checkpoints. Tip: Hybrid seems to give better results than transformer, but requires the mamba-ssm and flash-attn pip packages and doesn't work on all GPUs.")
args = parser.parse_args()

from ghostbox.tts_util import *
from ghostbox.tts_state import *
from ghostbox.tts_backends import *
import time, threading, os
prog = TTSState(args)
tts = initTTS(prog.args.model, config=vars(prog.args))

# list voices if requested
if args.voices:
    for voice_name in tts.get_voices():
        print(voice_name)
    sys.exit()


config_options = dump_config(tts)
if config_options != []:
    printerr("Dumping TTS config options. Set them with '/<OPTION> <VALUE>'. /ls to list again.")
    for w in config_options:
        printerr(w)


if args.filepath == "":
    output_file = tempfile.NamedTemporaryFile(suffix=".wav")
else:
    output_file = open(args.filepath, "w")

output_file.close()    


# do pygame here because of wonderful hello from pygame message
import pygame
pygame.mixer.init()

# ok we are hacking this to allow stopping of all sounds
from queue import Queue, Empty
msg_queue = Queue()
done = threading.Event()
snd_stop_flag = threading.Event()
def input_loop():
    global done
    global prog
    global tts
    
    while True:
        try:
            w = input()
            if w == "<clear>":
                pygame.mixer.stop() 
                with msg_queue.mutex:
                    snd_stop_flag.set()                                                        
                    msg_queue.queue.clear()
                prog.clearRetries()
                continue
            elif w.startswith("/"):
                vs = w[1:].split(" ")
                option = vs[0]
                if option == "ls":
                    for u in dump_config(tts):
                        printerr(u)
                    continue
                elif option in tts.config:
                    try:
                        value = ast.literal_eval(" ".join(vs[1:]))
                    except:
                        printerr("Couldn't set config option '" + vs[0] + "'. Error in value literal?")
                        continue
                    tts.configure(**{option:value})
                    continue

            # main event -> speak input msg w
            snd_stop_flag.clear()
            ws = tts.split_into_sentences(w)
            for chunk in ws:
                msg_queue.put(chunk)
        except EOFError as e:
            prog.handleMixins()
            time.sleep(3)
            done.set()
            break
        except:
            print("Exception caught while blocking. Shutting down gracefully. Below is the full exception.")
            print(traceback.format_exc())                    
            prog.handleMixins()
            #print("EOF encountered. Closing up.")
            time.sleep(3)
            done.set()
            break
        

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
    try:
        tts.tts_to_file(text=msg, speaker_file=prog.getVoiceSampleFile(), file_path=output_file.name)
    except ZeroDivisionError:
        print("Caught zero division error. Ignoring.")
        # this happens when the tts is asked to process whitespace and produces a wav file in 0 seconds :) nothing to worry about
        continue
    except AssertionError as e:
        print(str(e) + "\nwarning: Caught assertion error on msg: " + msg)
        prog.retry(msg)
        continue # we retry the msg that was too long
        
        
    prog.accumulateSound(output_file.name)
    prog.addPause()
    if prog.args.quiet:
        continue
    snd = pygame.mixer.Sound(output_file.name)
    while pygame.mixer.get_busy():
        if snd_stop_flag.isSet():
            snd.stop()
            break
        pygame.time.delay(10) #ms

    if snd_stop_flag.isSet():
        snd_stop_flag.clear()            
        continue
    snd.set_volume(prog.args.volume)
    snd.play()

prog.cleanup()
os.remove(output_file.name)

