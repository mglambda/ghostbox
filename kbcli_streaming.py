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

def process_sse_streaming_events(prog, r):
    for event in r.iter_lines():
        if event:
            w = event.decode()
            if w.startswith("data: "):
                d = json.loads(w[6:])
                v = d["token"]
                print(v, end="", flush=prog.getOption("streaming_flush"))
                prog.stream_queue.append(v)
    prog.streaming_done.set()


def streamPrompt(prog, url, json=""):
    response = connect_to_endpoint(url, json)
    if response:
        thread = Thread(target=process_sse_streaming_events, args=(prog, response))
        thread.start()
    return response
