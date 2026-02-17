from typing import Callable, Any, Dict, Optional, Mapping
import requests, json
#from requests_html import HTMLSession
from time import sleep
from threading import Thread, Event
from .util import printerr


# FIXME:   # poor man's closure; somehow this isn't enough yet to warrant making a class
stop_streaming = Event()

def connect_to_endpoint(url: str, prompt: Any, headers: Optional[Mapping[str, str]] = None) -> requests.Response | None:
    try:
        #session = HTMLSession()
        session = requests.Session()
        r = session.post(url, json=prompt, stream=True, headers=headers)
        return r
    except Exception as e:
        printerr(f"Error connecting to {url}: {e}")
        return None


  
    
def process_sse_streaming_events(callback: Callable[[Dict[str, Any]], None], done_flag: Event, r: requests.Response) -> None:
    global stop_streaming
    for event in r.iter_lines():
        if stop_streaming.is_set():
            # FIXME: this being global will be an issue :/
            break
            
        if event:
            w = event.decode()
            if w == "data: [DONE]":
                # openai do this
                #break
                pass
            elif w.startswith("data: "):
                d = json.loads(w[6:])            
                callback(d)
            elif w.startswith("data:"):
                d = json.loads(w[5:])            
                callback(d)                
            else:
                # this works usually if people aren't actually streaming, but maybe we should just crash
                printerr("warning: Malformed data in process_sse_streaming_events. Are you actually streaming?")
                printerr(f"dump: {w}")
                d = json.loads(w)            
                callback(d)                

                
    done_flag.set()


def streamPrompt(callback: Callable[[Dict[str, Any]], None], done_flag: Event, url: str, json: Any = "", headers: Optional[Mapping[str, str]] = None) -> requests.Response | None:
    global stop_streaming
    stop_streaming.clear()
    
    response = connect_to_endpoint(url, json, headers=headers)
    if response:
        thread = Thread(target=process_sse_streaming_events, args=(callback, done_flag, response))
        thread.start()
    return response
