
class Story(object):
    """A story is a thin wrapper around a list of story-items. A story-item is a simple python dictionary with 'role' and 'content' fields defined, though it may contain others as well."""
    def __init__(self):
        self.data = []

    def addUserText(self, w, **kwargs):
        self.data.append({ "role" : "user", "content" : w} | kwargs)

    def addAssistantText(self, w, **kwargs):
        self.data.append({ "role" : "assistant", "content" : w} | kwargs)

    def extendAssistantText(self, w):
        if self.data == []:
            self.addAssistantText(w)
        else:
            self.data[-1]["content"] = self.data[-1]["content"] + w
            
    def extendLast(self, w):
        self.data[-1]["content"] = self.data[-1]["content"] + w
            
    def getData(self):
        return self.data
        
    def pop(self, n=-1):
        """Remove and return the last story item. If n is supplied, item at position n is removed and returned."""
        return self.data.pop(n)

    def dropUntil(self, predicate):
        """Drops story items from the back of the list until predicate is True. Predicate takes a story item as argument. Returns true if predicate was true for an item."""
        while self.data != []:
            if predicate(self.data[-1]):
                return True
            self.pop()
        return False
        

