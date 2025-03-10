from typing import *
from pydantic import BaseModel
from ghostbox.definitions import *

class Story(object):
    """A story is a thin wrapper around a list of ChatMessages."""
    
    data: List[ChatMessage] = [] 

    def addUserText(self, w: str, image_id:Optional[int]=None, **kwargs) -> None:
        new_data = ChatMessage(role = "user", content = w, image_id=image_id, **kwargs)
        self.data.append(new_data)

    def addAssistantText(self, w: str, **kwargs):
        self.data.append(ChatMessage(role="assistant", content= w, **kwargs))

    def addRawJSONs(self, json_list: List[Dict[str, Any]]) -> None:
        """Add one or more python dictionaries that will be interpreted as ChatMessages and appended to the story.
                         If any of the passed dictionaries don't conform to the ChatMessage schema, you will get a pydantic ValidationError."""
        self.data.extend([ChatMessage(**item)
                          for item in json_list])

    def addMessages(self, msgs: List[ChatMessage]) -> None:
        """Appends ChatMessages to the story."""
        self.data.extend(msgs)
        
    def addSystemText(self, w: str, **kwargs) -> None:
        self.data.append(ChatMessage(role="system", content= w, **kwargs))

    def extendAssistantText(self, w: str) -> None:
        """Alters the latest found message in the story that is by the assistant, and extends it with w. If no such message exists, it adds w as an assistant message to the story."""
        for i in range(-1, -1*(len(self.data)+1), -1):
            # go through msgs from back to front
            msg = self.data[i]
            if msg.role == "assistant":
                if msg.content is None:
                    # this case is too weird, we just skip empty content
                    continue
                elif type(msg) == str:
                    # easy case
                    msg.content += w
                    return
                else:
                    # the content is complex -> a list of images + text or smth
                    for cmsg in msg.content:
                        if cmsg.type == "text":
                            cmsg.content += w
                            return

        # if we reached this point, no assistant was found
        # we just append a new message
        self.addAssistantText(w)
                

                
                

            
    def extendLast(self, w:str) -> None:
        """Appends w to the last message. Does nothing if there are no messages."""
        if self.data == []:
            return
        
        msg = self.data[-1]
        if msg.content is None:
            # tricky case, I say we do nothing
            return
        elif type(msg.content) == str:
            msg.content += w
        else:
            # the message is complex
            for cmsg in msg.content:
                # this is a truly weird case, I guess we mirror the behaviour of extend assistant and just extend the first text field
                if cmsg.type == "text":
                    cmsg.content += w
                    return
                
            
            
    def getData(self) -> List[ChatMessage]:
        return self.data

    def drop(self, n:int=-1) -> bool:
        """Safely remove the nth story item from the story. Defaults to the last item. If there are no elements, of if n is out of range, this has no effect. Returns True if an item was removed."""
        if self.data == []:
            return False

        if n == -1:
            n = len(self.data) - 1
        
        if not(n in range(0, len(self.data))):
            return False
        self.pop(n)
        return True
    
    
    def pop(self, n:int=-1) -> ChatMessage:
        """Remove and return the last story item. If n is supplied, item at position n is removed and returned."""
        return self.data.pop(n)
            
    def dropUntil(self, predicate: Callable[[ChatMessage], bool]) -> bool:
        """Drops chat messages from the back of the list until predicate is True. Predicate takes a ChatMessage as argument. Returns true if predicate was true for an item."""
        while self.data != []:
            if predicate(self.data[-1]):
                return True
            self.pop()
        return False
        
    def to_json(self) -> List[Dict[str, Any]]:
        """Returns internal data as json models.
        Shorthand for mapping model_dump over a getData() call."""
        return [msg.model_dump() for msg in self.data]
