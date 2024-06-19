import json
import urllib3
import httpx

http = urllib3.PoolManager()

url = "http://0.0.0.0:8000/chat"

data = json.dumps({"query": "Give me a short introduction to large language model.", "max_new_tokens": 1024, "temperature": 0.8, "top_p": 0.9})

with httpx.stream('POST', url, data=data) as r:
    response = ''
    for chunk in r.iter_lines():
        new_text = chunk.split('data: ')[-1]
        response += new_text
        print(response)
