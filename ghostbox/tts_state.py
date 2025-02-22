import os, subprocess
from moviepy.editor import *
from ghostbox.tts_util import *
from queue import Queue

class TTSState(object):
    def __init__(self, args):
        self.args = args
        self.accfile = getAccumulatorFilename(args.filepath)
        self.tmpfile = "tmp.wav"
        self.mixins = [] # list of (musicfilename, timestampe)
        self.retry_queue = Queue()
        
        
        self.tagFunctions = {
            "sound" : self._soundTag,
            "mixin" : self._mixinTag}

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
        clips = [AudioFileClip(self.accfile)]
        for (mixfile, timestamp) in self.mixins:
            clips.append(AudioFileClip(mixfile).with_start(timestamp))

        outclip = CompositeAudioClip(clips)
        outclip.write_audiofile(self.tmpfile)
        subprocess.run(["cp", self.tmpfile, self.accfile])        
        
    def accumulateSound(self, sndFile):
        subprocess.run(["sox", self.accfile, sndFile, self.tmpfile])
        subprocess.run(["cp", self.tmpfile, self.accfile])
        
    def addPause(self):
        if self.args.pause_duration == 0:
            return

        for n in range(0, self.args.pause_duration):
            self.accumulateSound(self.args.sound_dir + "/silence.1.wav")

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
            
        
        
