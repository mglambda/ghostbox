#!/usr/bin/env python
import argparse, traceback, sys, tempfile, ast, shutil, json
from ghostbox.definitions import TTSOutputMethod, TTSModel
from ghostbox.tts_util import *
from ghostbox.tts_state import *
from ghostbox.tts_backends import *
from ghostbox.tts_backends_orpheus import OrpheusBackend
from ghostbox.tts_output import *
import cProfile

def main():
    program_name = sys.argv[0]
    parser = argparse.ArgumentParser(description= program_name + " - TTS program to consume text from stdin and speak it out/ save it as wav file.")
    #parser.add_argument("-f", '--filepath', type=str, default="", help="Filename to save accumulated spoken lines in. Output is in wav format.")
    parser.add_argument("--voices", action=argparse.BooleanOptionalAction, default=False, help="List all available voices for chosen model, then exit the program.")
    parser.add_argument("-q", "--quiet", action=argparse.BooleanOptionalAction, default=False, help="Do not play any audio.")
    parser.add_argument("-l", "--language", type=str, default="en", help="Language for the TTS output. Not all TTS models support all languages, and many don't need this option.")
    parser.add_argument("-p", "--pause_duration", type=int, default=1, help="Duration of pauses after newlines. A value of 0 means no or minimal-duration pause.")
    parser.add_argument("-y", "--voice", type=str, default="", help="Name of the voice to use. What voices are supported differs with each model. For a list, start the program with --voices.")
    parser.add_argument("--clone", type=str, default="", help="Path to a .wav file of a voice to clone, for models that support cloning. This will usually override the --voice option.")
    parser.add_argument("--clone_dir", type=str, default="", help="Directory in which to search for wave files to clone with --clone.")
    parser.add_argument("-i", "--volume", type=float, default=1.0, help="Volume for the voice playback. Not supported by all models in all modes (streaming vs file playback).")
    parser.add_argument("-s", "--seed", type=int, default=420, help="Random seed for voice models that use it.")
    parser.add_argument("--sound_dir", type=str, default="sounds", help="Directory where sound files are located to be played with #sound <SNDNAME>")
    parser.add_argument("-m", "--model", type=str, choices=[tm.name for tm in TTSModel], default=TTSModel.zonos.name, help="Text-to-speech model to use.")
    parser.add_argument("-o", "--output-method", type=str, choices=[om.name for om in TTSOutputMethod], default=TTSOutputMethod.default.name, help="How to play the generated speech.")
    parser.add_argument("--websock-host", type=str, default="localhost", help="The hostname to bind to when using websock as output method.")
    parser.add_argument("--websock-port", type=int, default=5052, help="The port to listen on for connections when using websock as output method.")
    # zonos specific
    parser.add_argument("--zonos_model", type=str, default=ZonosTTSModel.hybrid.name, help="The pretrained checkpoint to use with the Zonos TTS engine. Hybrid seems to get the best results. This argument is ignored unless you use the Zonos model. Options: " + ", ".join([m.name for m in ZonosTTSModel]))
    parser.add_argument("--orpheus_quantization", type=str, default="Q4_K_M", help="Quantization method to use for the orpheus model. Options currently supported are 'Q4_M_K', 'Q8_0'. Using lower quants may degrade performance, but will lower vram requirements significantly. This option makes ghostbox figure out the exact model to use. If you want a quant that's not supported, set bta custom model or huggingface repo with the --orpheus_model option, which will cause this option to be ignored.")
    parser.add_argument("--orpheus_model", type=str, default="", help="The exact orpheus model to use. By default, ghostbox-tts will figure this out on its own based on the value of orpheus_quantization. Setting this option will override orpheus_quantization. You may set this option to either a filepath pointing to a model (e.g. a gguf file), or to a huggingface repo like 'https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf'.")
    parser.add_argument("--llm_server", type=str, default="", help="Hostname and port of LLM server to query for models that need it. Currently this is used only by Orpheus. Any OpenAI compatible backend can be used. If this is set, ghostbox-tts will not spawn its own server and options like orpheus_model or orpheus_quantization are ignored.")
    parser.add_argument("--llm_server_executable", type=str, default="llama-server", help="Path to an executable of a server capable of loading LLM models. This is only relevant when ghostbox-tts attempts to spawn its own llm server. Note: only tested with llama.cpp.")
    parser.add_argument("--profiling", action=argparse.BooleanOptionalAction, default=False, help="Enable profiling. This produces a .prof file that you can analyze for performance statistics.")
    args = parser.parse_args()


    import time, threading, os

    def initTTS(model: str, config: Dict[str, Any] = {}) -> TTSBackend:
        if model == TTSModel.xtts.name:
            return XTTSBackend()
        elif model == TTSModel.zonos.name:
            return ZonosBackend(config=config)
        elif model == TTSModel.kokoro.name:
            return KokoroBackend(config=config)
        elif model == TTSModel.orpheus.name:
            return OrpheusBackend(config=config)        
        raise ValueError("Not a valid TTS model: " + model + ". Valid choices are " + "\n  ".join([tm.name for tm in TTSModel]))

    def initOutputMethod(method: str, args) -> TTSOutput:
        if method == TTSOutputMethod.default.name:
            return DefaultTTSOutput(volume=args.volume)
        elif method == TTSOutputMethod.websock.name:
            return WebsockTTSOutput(host=args.websock_host, port=args.websock_port, volume=args.volume)
        raise ValueError("Not a valid output method: " + method + ". Valid choices are " + "\n  ".join([om.name for om in TTSOutputMethod]))

    # initialization happens here
    prog = TTSState(args)
    output_module = initOutputMethod(prog.args.output_method, prog.args)
    tts = initTTS(prog.args.model, config=vars(prog.args))
    # we have to put something on the message queue that signals EOF but isn't actually EOF
    # the tokens are only for the msg_queue, and can on principle never be a user request, since we strip leading and trailing whitespace
    eof_token = "  <EOF>  "
    silence_token = "  <silence>  "
    # thisis different from e.g. <clear>, which is currently the only special string that might actullay be user input. oh well

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

    from queue import Queue, Empty
    msg_queue = Queue()
    done = threading.Event()

    if args.profiling:
        # we start profiling as we don't care about how long initialization takes
        profiler = cProfile.Profile()
    
    def input_loop():
        nonlocal done
        nonlocal prog
        nonlocal tts
        nonlocal output_module

        while True:
            try:
                w = input()
                if w == "<clear>":
                    with msg_queue.mutex:
                        output_module.stop()                                    
                        msg_queue.queue.clear()
                    prog.clearRetries()
                    continue
                elif w.strip() == "":
                    msg_queue.put(silence_token)
                elif w == "<is_speaking>":
                    print("is_speaking: " + str(output_module.is_speaking()), flush=True)
                    continue
                elif args.profiling and w == "<simulate_eof>":
                    # this is just convenient for testing, since it can be nontrivial to send EOF ina way that respects the msg queue
                    msg_queue.put(eof_token)
                    continue
                elif args.profiling and w == "<start_profiling>":
                    profiler.enable()
                    continue
                elif w == "<dump_config>":
                    print("config: " + json.dumps(tts.config))
                    continue
                elif w.startswith("/"):
                    vs = w[1:].split(" ")
                    option = vs[0]
                    if option == "ls":
                        for u in dump_config(tts):
                            printerr(u)
                        continue
                    elif option in tts.get_config().keys():
                        try:
                            value = ast.literal_eval(" ".join(vs[1:]))
                        except:
                            printerr("Couldn't set config option '" + vs[0] + "'. Error in value literal?")
                            continue
                        tts.configure(**{option:value})
                        continue

                # main event -> speak input msg w
                ws = tts.split_into_sentences(w)
                for chunk in ws:
                    msg_queue.put(chunk)
            except EOFError as e:
                printerr("EOF")
                msg_queue.put(eof_token)
                break
            except:
                printerr("Exception caught while blocking. Shutting down gracefully. Below is the full exception.")
                printerr(traceback.format_exc())                    
                time.sleep(3)
                done.set()
                break


    t = threading.Thread(target=input_loop)
    t.daemon = True
    t.start()


    # this is so muggels know to type stuff when they accidentally run ghostbox-tts standalone
    printerr("Good to go. Reading messages from standard input. Hint: Type stuff and it will be spoken.")
    while True:
        # here we handle text chunks that were placed on the msg_queue
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
                if rawmsg == eof_token:
                    done.set()
                    continue
                elif rawmsg == silence_token:
                    if not(prog.args.quiet):
                        output_module.enqueue(prog.silence_filename())
                    continue
        except Empty:
            # timeout was hit
            continue
        except: #EOFError as e:
            printerr("Exception caught while blocking. Shutting down gracefully. Below is the full exception.")
            printerr(traceback.format_exc())        
            time.sleep(3)
            os._exit(1)

        (msg, cont, err) = prog.processMsg(rawmsg)
        if err:
            printerr(err)
        if cont:
            continue


        output_file = prog.temp_wav_file()
        try:
            if tts.can_stream():
                payload = tts.tts_to_generator(msg)
            else:
                tts.tts_to_file(text=msg, file_path=output_file.name)
                payload = output_file
        except IgnoreValueError as e:
            # this happens on some bad values that are hard to filter but harmless.
            # e.g. "---" for text in kokoro
            printerr("Ignored `" + e.ignored_value + "`.")
            continue
        except ZeroDivisionError:
            printerr("Caught zero division error. Ignoring.")
            # this happens when the tts is asked to process whitespace and produces a wav file in 0 seconds :) nothing to worry about
            continue
        except AssertionError as e:
            printerr(str(e) + "\nwarning: Caught assertion error on msg: " + msg)
            prog.retry(msg)
            continue # we retry the msg that was too long

        if prog.args.quiet:
            continue


        if not(tts.can_stream()):
            # this filecopy is horrible but it is necessary because
            # even with ful program synchronization, the filesystem might not play ball
            # in any case without this, the wavefile created on this thread wouldn't show up on the other one
            newfilename = tempfile.mkstemp(suffix=".wav")[1]
            shutil.copy(output_file.name, newfilename)
            payload = newfilename

        # queue and play
        output_module.enqueue(payload)


    # this will let all enqueued files finish playing
    output_module.shutdown()
    prog.cleanup()
    if args.profiling:
        profiler.disable()
        profiler.dump_stats("ghostbox-tts.prof")
    
if __name__ == "__main__":
    main()
    
