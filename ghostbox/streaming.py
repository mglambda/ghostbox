import requests, json
from requests_html import HTMLSession
from time import sleep
from threading import Thread
def connect_to_endpoint(url, prompt):
    try:
        session = HTMLSession()
        r = session.post(url, json=prompt, stream=True)
        return r
    except Exception as e:
        print(f"Error connecting to {url}: {e}")
        return None

def process_sse_streaming_events(callback, flag, r):
    print("thread started")
    #    for event in r.iter_lines():
    for event in r.iter_content(decode_unicode=True):
        print(event)
        if event:
            w = event
        callback(w)
    flag.set()


def streamPrompt(callback, flag, url, json=""):
    response = connect_to_endpoint(url, json)
    if response:
        thread = Thread(target=process_sse_streaming_events, args=(callback, flag, response))
        thread.start()
    return response
