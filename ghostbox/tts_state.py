import os, subprocess, tempfile
from moviepy.editor import *
import ghostbox
from ghostbox.tts_util import *
from queue import Queue

class TTSState(object):
    def __init__(self, args):
        self.args = args
        if args.model == "xtts":
            self._default_samplerate = "24000"
        else:
            self._default_samplerate = "44100"
            # FIXME: accumulation temporarily disabled
            args.filepath = ""
        if args.filepath != "":
            # user wants to keep acc file
            self._keep_acc = True
        else:
            self._keep_acc = False
        self.accfile = getAccumulatorFile(args.filepath)
        self.accfile.close()
        self._empty_filename = ghostbox.get_ghostbox_data("empty." + self._default_samplerate + ".wav")
        self._silence_filename = ghostbox.get_ghostbox_data("silence.1." + self._default_samplerate + ".wav")
        subprocess.run(["cp", self._empty_filename, self.accfile.name])        
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.tmpfile.close()
        self.mixins = [] # list of (musicfilename, timestampe)
        self.retry_queue = Queue()
        self.temp_wav_files = []
        
        self.tagFunctions = {
            "sound" : self._soundTag,
            "mixin" : self._mixinTag}

    def temp_wav_file(self):
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        # without fsync the file isn't guaranteed to be available to other threads
        # even if you .close() it
        # python is so great it just works /s
        os.fsync(f)
        f.close()
        self.temp_wav_files.append(f)
        return f

    def silence_filename(self) -> str:
        return self._silence_filename
    
    def cleanup(self):
        for tf in self.temp_wav_files:
            os.remove(tf.name)
        
        os.remove(self.tmpfile.name)
        if not(self._keep_acc):
            os.remove(self.accfile.name)
            
        
        
    def getVoiceSampleFile(self):
        return self.args.voice_sample
        
    def processMsg(self, msg):
        # returns (newMsg, continue_bool, errormsg)

        if msg == "":
            return ("", True, "")

        msg = fixDivZero(msg)
        
        (tag, tagArgs) = maybeGetTag(msg)
        if tag in self.tagFunctions:
            return self.tagFunctions[tag](tagArgs)

        return (msg, False, "")

    def _soundTag(self, argv):
        if argv == []:
            return ("", True, "")

        sndFile = self.args.sound_dir + "/" + " ".join(argv)
        if not(os.path.isfile(sndFile)):
            return ("", False, "Warning: Could not find sound file: " + sndFile)

        self.accumulateSound(sndFile)
        return ("", True, "")

    def _mixinTag(self, argv):
        if argv == []:
            return ("", True, "")        

        mixfile = argv[0]
        clip = AudioFileClip(self.accfile)
        self.mixins.append((mixfile, clip.duration))
        return ("", True, "")

    def handleMixins(self):
        # adds background music etc. to the accumulated file
        clips = [AudioFileClip(self.accfile.name)]
        for (mixfile, timestamp) in self.mixins:
            clips.append(AudioFileClip(mixfile).with_start(timestamp))

        outclip = CompositeAudioClip(clips)
        # this is due to a bug (I think) in moviepy with fps not being defined
        if "fps" not in outclip.__dict__:
            outclip.fps = 44100
        outclip.write_audiofile(self.tmpfile.name)
        subprocess.run(["cp", self.tmpfile.name, self.accfile.name])        
        
    def accumulateSound(self, sndFile):
        subprocess.run(["sox", self.accfile.name, sndFile, self.tmpfile.name])
        subprocess.run(["cp", self.tmpfile.name, self.accfile.name])
        
    def addPause(self):
        if self.args.pause_duration == 0:
            return

        for n in range(0, self.args.pause_duration):
            self.accumulateSound(self._silence_filename)

    def isRetrying(self):
        return not(self.retry_queue.empty())
                
    def retry(self, msg):
        # called when e.g. a messsage is too long, which we can only know after the fact (thanks api designers)
        # first attempt of fixing is by removing all quotation marks. This seems to confuse the tokenizer, when quotation marks aren'tr balanced
        w = msg.replace('"', "")
        if w != msg:
            self.retry_queue.put(w)
            return
        # most really long run-on sentences are due to many commas. We find a comma in the middle and split the string in two
        ws = msg.split(",")
        if len(ws) > 1:
            i = (len(ws) // 2) - 1
            v1 = ",".join(ws[0:i])
            v2 = ws[i] + "." + ",".join(ws[i:])
            self.retry_queue.put(v1)
            self.retry_queue.put(v2)
            return

        # now the gloves are off. Brutally split the string in half.
        i = len(msg) // 2
        self.retry_queue.put(msg[:i])
        self.retry_queue.put(msg[i:])
        return

    def popRetryMSG(self):
        return self.retry_queue.get()
    
    def clearRetries(self):
        with self.retry_queue.mutex:
            self.retry_queue.queue.clear()
            
        
        
