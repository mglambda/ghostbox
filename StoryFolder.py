import json


class StoryFolder(object):
    def __init__(self, json_data=None):
        self.stories = [[]]
        self.index = 0 # points to where to append next
        if json_data:
            self.stories = json.loads(json_data) # throw if illegal json
            # FIXME: this will crash and burn if json is bogus, but oh well

            


    def getStory(self):
        return self.stories[self.index]

    def showStory(self):
        return "".join(self.getStory())
    
    def newStory(self):
        self.stories.append([])
        self.index = len(self.stories) - 1

    def cloneStory(self, index=-1):
        if index == -1:
            index = self.index

        l = len(self.stories)
        if index >= 0 and index < l:
            self.stories.append(self.stories[index])
            self.index = l

    def dropEntry(self):
        self.stories[self.index] = self.stories[self.index][0:-1]
        
    def addText(self, w):
        self.getStory().append(w)
        
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
        return json.dumps(self.stories)
    
    def shiftTo(self, i):
        l = len(self.stories)
        if i >= l or i < 0:
            return "Index out of range."
        self.index = i
        return ""
    
