import asyncio
from threading import Thread
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)


class Item(BaseModel):
    query: str
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.9


def start_generation(query, max_new_tokens, temperature, top_p):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': query}]
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text
        asyncio.sleep(0.1)


@app.post('/chat')
async def stream(params: Item):
    query = params.query
    max_new_tokens = params.max_new_tokens
    temperature = params.temperature
    top_p = params.top_p
    print(f'query receieved: {query}')
    return EventSourceResponse(start_generation(query, max_new_tokens, temperature, top_p), media_type='text/event-stream')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
