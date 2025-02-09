from transformers import AutoTokenizer, ReformerModel
import torch

tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
model = ReformerModel.from_pretrained("google/reformer-crime-and-punishment")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)