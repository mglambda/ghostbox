import requests, json
#from requests_html import HTMLSession
from time import sleep
from threading import Thread
from ghostbox.util import printerr

def connect_to_endpoint(url, prompt, headers=""):
    try:
        #session = HTMLSession()
        session = requests.Session()
        r = session.post(url, json=prompt, stream=True, headers=headers)
        return r
    except Exception as e:
        printerr(f"Error connecting to {url}: {e}")
        return None

def process_sse_streaming_events(callback, flag, r):
    for event in r.iter_lines():
        if event:
            w = event.decode()
            if w == "data: [DONE]":
                # openai do this
                break
            elif w.startswith("data: "):
                d = json.loads(w[6:])            
                callback(d)
    flag.set()


def streamPrompt(callback, flag, url, json="", headers=""):
    response = connect_to_endpoint(url, json, headers=headers)
    if response:
        thread = Thread(target=process_sse_streaming_events, args=(callback, flag, response))
        thread.start()
    return response
