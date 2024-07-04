import json
import torch
import asyncio
import torch_npu
from threading import Thread
from fastapi import FastAPI
from pydantic import BaseModel
from torch_npu.contrib import transfer_to_npu
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

device_id = torch.npu.current_device()
npu_device = f"npu:{device_id}"
torch.npu.set_device(torch.device(npu_device))
torch.npu.set_compile_mode(jit_compile=False)

option = dict()
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
torch.npu.set_option(option)

app = FastAPI()

history = []

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map=npu_device,
    trust_remote_code=True
).half().npu()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)


def generate_with_npu_device(**generation_kwargs):
    torch.npu.set_device(torch.device(npu_device))
    generation_kwargs["input_ids"] = generation_kwargs["input_ids"].to(npu_device)
    model.generate(**generation_kwargs)


class Item(BaseModel):
    query: str
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.9
    clear: bool = False


def start_generation(query, max_new_tokens, temperature, top_p):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': query}]
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    total_prompt_tokens = inputs.size()[1]
    total_output_tokens = 0
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    thread = Thread(target=generate_with_npu_device, kwargs=generation_kwargs)
    thread.start()

    partial_text = ''
    for new_text in streamer:
        total_output_tokens += len(tokenizer.encode(new_text))
        partial_text += new_text
        yield json.dumps({'code': 200, 'success': True, 'data': new_text}, ensure_ascii=False)
        asyncio.sleep(0.1)

    response = partial_text
    history.append((query, response))
    yield json.dumps({'code': 200, 'success': True, 'end': {'input_tokens': total_prompt_tokens,
                                                            'total_tokens': total_prompt_tokens + total_output_tokens}}, ensure_ascii=False)


@app.post('/chat')
async def stream(params: Item):
    global history
    query = params.query
    max_new_tokens = params.max_new_tokens
    temperature = params.temperature
    top_p = params.top_p
    clear = params.clear

    if clear:
        history = []
    print(f'Query receieved: {query}')
    return EventSourceResponse(start_generation(query, max_new_tokens, temperature, top_p), media_type='text/event-stream')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
