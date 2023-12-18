import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 128
model_name = "./opt-1.3b"

text = 'Hamburg is in which country?\n'
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids

model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map='auto',
  load_in_8bit=True,
  torch_dtype=torch.bfloat16
)
print('[info] model dtype', model.dtype)
with torch.no_grad():
  generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("output:", output, " (type =", type(output), ", len =", len(output), ")")
