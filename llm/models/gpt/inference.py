import tiktoken
import torch
from GPT2 import GPT2
from init import init

conf = init(seed=42)

enc = tiktoken.get_encoding("gpt2")

batch_size = 5
max_length = 30

prompt = "Hello, I'm a language model,"
tokens = enc.encode(prompt)  # (sequence_length,)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, sequence_length)

input_ids = tokens.to(conf.device)

model = GPT2.from_pretrained_direct("gpt2")

output_ids = model.generate(
    input_ids=input_ids,
    max_length=30,
    num_return_sequences=1,
    do_sample=True,
    # do_sample=False,
    top_k=50,
)

for i in range(batch_size):
    tokens = output_ids[i, :].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
