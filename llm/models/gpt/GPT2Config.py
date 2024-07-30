"""
I am learning Transformers from huggungface's transformers library and Andrej Karpathy's nanoGPT + 3b1b's YT series.
They all have different names for the same thing. To make sense of it all, here there are different names for the same thing.

GPT2 Config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py


There are 4 types of GPT2 models:
- GPT2: default (this file)
- GPT2 Medium: n_layer=24, n_head=16, d_model=1024
- GPT2 Large: n_layer=36, n_head=20, d_model=1280
- GPT2 XL: n_layer=48, n_head=25, d_model=1600
"""

from dataclasses import dataclass


@dataclass
class GPT2Config:

    # Vocabulary size = Num of tokens
    # This is used to build a lookup table for embeddings. Each token is a row in the table pointing to a the corresponding embedding vector.
    vocab_size: int = 50257  # hf
    n_vocab: int = vocab_size  # 3b1b

    # Word embedding dimension
    n_embd: int = 768  # Andrej, hf
    embed_dim: int = n_embd  # hf
    d_embed: int = n_embd  # 3b1b
    # Hidden layer dimension
    # First, we have tokens (integers) as the input of the model. Then after the embedding layer, we have embeddings (vectors) which can be seen as hidden states.
    # Because of that, it is making sense to call the embedding dimension as the hidden size.
    hidden_size = embed_dim  # hf transformers uses both names,

    # Number of positional embeddings = Max number of tokens in a sequence
    # GPT2 uses an absolute positional embedding. The positional embeddings are added to the token embeddings.
    n_positions: int = 1024  # hf
    # This should be maximum. GPT2 doesn't use KV cache. So, the inference process starts by the input tokens
    # and adds new tokens to the sequence until the max number of tokens.
    sequence_len: int = n_positions
    max_seq_len: int = n_positions
    max_position_embeddings: int = n_positions  # hf, transformers lib,
    block_size: int = n_positions  # Andrej

    # Number of context tokens = Attention window size. There is no actual windowing in GPT2, so this is the max number of tokens in a sequence.
    n_ctx: int = 1024  # hf
    ctx_len: int = n_ctx
    ctx_size: int = n_ctx

    # Number of layers
    # Number of GPT2Blocks (in transformers lib)
    # These layer are used sequentially. Each layer has a self-attention mechanism and a feedforward neural network.
    # In each layer (iteration), there are multiple attention heads. Each head has its own query, key, value matrices.
    n_layer: int = 12  # Andrej, hf
    num_hidden_layers: int = n_layer  # hf

    # Number of attention heads
    # They run in parallel. Each head learns different features.
    # Dimension of each each can be calculated as d_model / n_head.
    n_head: int = 12  # Andrej, hf
    num_attention_heads: int = n_head  # hf

    # Head size = head dimension
    head_size: int = n_embd // n_head

    # Query space dimension
    query_dim: int = 64
    d_query: int = query_dim  # 3b1b

    # Value space dimension
    value_dim: int = 64
    d_value: int = value_dim  # 3b1b

    # Key space dimension
    key_dim: int = 64
    d_key: int = key_dim  # 3b1b

    # Dropout and layer norm
    attn_pdrop: float = 0.1  # hf
    embd_pdrop: float = 0.1  # hf
    layer_norm_epsilon: float = 1e-5  # hf
    resid_pdrop: float = 0.1  # hf

    def __post_init__(self):
        assert (
            self.n_embd % self.n_head == 0
        ), "Embedding dimension must be divisible by the number of heads"


if __name__ == "__main__":
    from transformers import AutoConfig

    models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    for model in models:
        config = AutoConfig.from_pretrained(model)
        print(f"Model: {model}\n")
        print(config)


# if __name__ == "__main__":
#     models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

#     for model in models:
#         print(
#             {
#                 "gpt2": GPT2Config(),
#                 "gpt2-medium": GPT2Config(n_embd=1024, n_head=16, n_layer=24),
#                 "gpt2-large": GPT2Config(n_embd=1280, n_head=20, n_layer=36),
#                 "gpt2-xl": GPT2Config(n_embd=1600, n_head=25, n_layer=48),
#             }[model]
#         )
