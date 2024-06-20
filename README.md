# LLM-Stream-Chat

I give an example of streaming chat of a large language model that is simple and can be used quickly. It is worth noting that it is based on the HTTP interface form of FastAPI.

If your model only supports single round chat and requires a prompt template, you can follow the following:
```python
ROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)
query = PROMPT_TEMPLATE.format_map({'instruction': instruction})
inputs = tokenizer(query, return_tensors="pt")["input_ids"].to(model.device)
```
