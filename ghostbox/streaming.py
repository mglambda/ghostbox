import requests, json
#from requests_html import HTMLSession
from time import sleep
from threading import Thread, Event
from ghostbox.util import printerr

# FIXME:   # poor man's closure; somehow this isn't enough yet to warrant making a class
stop_streaming = Event()

def connect_to_endpoint(url, prompt, headers=""):
    try:
        #session = HTMLSession()
        session = requests.Session()
        r = session.post(url, json=prompt, stream=True, headers=headers)
        return r
    except Exception as e:
        printerr(f"Error connecting to {url}: {e}")
        return None



  
    
def process_sse_streaming_events(callback, done_flag, r):
    global stop_streaming
    for event in r.iter_lines():
        if stop_streaming.isSet():
            break
            
        if event:
            w = event.decode()
            if w == "data: [DONE]":
                # openai do this
                break
            elif w.startswith("data: "):
                d = json.loads(w[6:])            
                callback(d)
    done_flag.set()


def streamPrompt(callback, done_flag, url, json="", headers=""):
    global stop_streaming
    stop_streaming.clear()
    
    response = connect_to_endpoint(url, json, headers=headers)
    if response:
        thread = Thread(target=process_sse_streaming_events, args=(callback, done_flag, response))
        thread.start()
    return response
