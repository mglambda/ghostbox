import jsonpickle, copy
from ghostbox.Story import *

class StoryFolder(object):
    """Thin wrapper around a list of Story objects."""
    def __init__(self, json_data=None):
        self.stories = [Story()]
        self.index = 0 # points to where to append next
        if json_data:
            self.stories = jsonpickle.loads(json_data) # throw if illegal json
            # FIXME: this will crash and burn if json is bogus, but oh well

    def empty(self):
        return self.stories[self.index] == []
            
    def get(self):
        return self.stories[self.index]
    
    def newStory(self):
        self.stories.append(Story())
        self.index = len(self.stories) - 1

    def reset(self) -> None:
        """Reset storyfolder to an empty state."""
        self.stories = [Story()]
        self.index = 0
        
    def cloneStory(self, index=-1):
        if index == -1:
            # -1  means currrent story
            index = self.index

        l = len(self.stories)
        if index >= 0 and index < l:
            self.stories.append(copy.deepcopy(self.stories[index]))
            self.index = l

    def copyFolder(self, only_active=False):
        sf = StoryFolder()
        if only_active:
            sf.stories = copy.deepcopy(self.stories[self.index:self.index+1])
        else:
            sf.stories = copy.deepcopy(self.stories)
            sf.index = self.index
        return sf
        
    def _shiftStory(self, i):
        l = len(self.stories)
        newIndex = self.index + i
        if newIndex >= l:
            return 1

        if newIndex < 0:
            return -1

        self.index = newIndex
        return 0

    def nextStory(self):
        return self._shiftStory(1)

    def previousStory(self):
        return self._shiftStory(-1)
    
    def toJSON(self):
        return jsonpickle.dumps(self.stories)
    
    def shiftTo(self, i):
        l = len(self.stories)
        if i >= l or i < 0:
            return "Index out of range."
        self.index = i
        return ""
    
