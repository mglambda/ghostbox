
class Story(object):
    """A story is a thin wrapper around a list of story-items. A story-item is a simple python dictionary with 'role' and 'content' fields defined, though it may contain others as well."""
    def __init__(self):
        self.data = []

    def addUserText(self, w, *kwargs):
        self.data.append({ "role" : "user", "content" : w} % kwargs)

    def addAssistantText(self, w, *kwargs):
        self.data.append({ "role" : "assistant", "content" : w} % kwargs)

    def getData(self):
        return self.data
        
