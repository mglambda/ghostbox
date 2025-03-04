from typing import *

class Story(object):
    """A story is a thin wrapper around a list of story-items. A story-item is a simple python dictionary with 'role' and 'content' fields defined, though it may contain others as well."""
    def __init__(self):
        self.data = []

    def addUserText(self, w, image_id=None, **kwargs):
        new_data = { "role" : "user", "content" : w} | kwargs
        if image_id is not None:
            new_data["image_id"] = image_id
        self.data.append(new_data)

    def addAssistantText(self, w, **kwargs):
        self.data.append({ "role" : "assistant", "content" : w} | kwargs)

    def addRawJSONs(self, json_list: List[Dict[str, Any]]) -> None:
        # FIXME: once we redo this with pydantic this will be validated
        self.data.extend(json_list)
        
    def addSystemText(self, w, **kwargs):
        self.data.append({ "role" : "system", "content" : w} | kwargs)
        

    def extendAssistantText(self, w):
        if self.data == []:
            self.addAssistantText(w)
        else:
            self.data[-1]["content"] = self.data[-1]["content"] + w
            
    def extendLast(self, w):
        self.data[-1]["content"] = self.data[-1]["content"] + w
            
    def getData(self):
        return self.data

    def drop(self, n=-1):
        """Safely remove the nth story item from the story. Defaults to the last item. If there are no elements, of if n is out of range, this has no effect. Returns True if an item was removed."""
        if self.data == []:
            return False

        if n == -1:
            n = len(self.data) - 1
        
        if not(n in range(0, len(self.data))):
            return False
        self.pop(n)
        return True
    
    
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
        

