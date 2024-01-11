

class StoryFolder(object):
    def __init__(self):
        self.stories = [[]]
        self.index = 0 # points to where to append next


    def getStory(self):
        return self.stories[self.index]

    def showStory(self):
        return "".join(self.getStory())
    
    def newStory(self):
        self.stories.append([])
        self.index += 1
        
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
    
        
