import math
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from GPT2Config import GPT2Config
from init import init
from transformers import GPT2LMHeadModel

conf = init()


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd
        self.head_size = config.head_size
        self.n_head = config.n_head

        self.c_attn = nn.Linear(
            in_features=config.n_embd, out_features=3 * config.n_embd
        )

        self.c_proj = nn.Linear(
            in_features=config.n_embd, out_features=config.n_embd, bias=True
        )

        # self.register_buffer(
        #     "mask", torch.tril(torch.ones(config.sequence_len, config.sequence_len))
        # )

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))

    def _attention(self, Q, K, V, batch_size, seq_len, n_embd):
        """
        Compute the attention output.

        Args:
            Q (torch.Tensor): The query tensor of shape (batch_size, n_head, seq_len, head_size).
            K (torch.Tensor): The key tensor of shape (batch_size, n_head, seq_len, head_size).
            V (torch.Tensor): The value tensor of shape (batch_size, n_head, seq_len, head_size).
            batch_size (int): The batch size.
            seq_len (int): The sequence length.
            n_embd (int): The embedding dimension.

        """

        K_tr = K.transpose(-2, -1)  # (batch_size, n_head, head_size, seq_len)
        # -2 and -1 are the last two dimensions, don't touch the batch_size dimension

        attention_scores = (
            Q @ K_tr
        )  # (batch_size, n_head, seq_len, seq_len) or (B, H, T, T)

        attention_scores_normalized = attention_scores / (
            self.head_size**0.5
        )  # Normalization by square root of key dimension

        T = seq_len
        masked_attention_scores_normalized = attention_scores_normalized.masked_fill(
            self.bias[:, :, :T, :T] == 0, float("-inf")
        )

        attention_weights = F.softmax(masked_attention_scores_normalized, dim=-1)
        # it calculates the softmax for each row in the last dimension

        attention = attention_weights @ V  # (batch_size, n_head, seq_len, head_size)

        attention_output = (
            attention.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        )

        return attention_output

    def forward(self, x):
        batch_size, seq_len, n_embd = x.size()

        QKV = self.c_attn(x)
        # print("QKV shape:", QKV.shape)
        Q, K, V = QKV.split(self.n_embd, dim=2)
        # print("Q shape:", Q.shape)
        # print("K shape:", K.shape)
        # print("V shape:", V.shape)

        Q = Q.view(
            batch_size, seq_len, self.n_head, n_embd // self.n_head
        )  # (batch_size, seq_len, n_head, head_size)
        Q = Q.transpose(1, 2)  # (batch_size, n_head, seq_len, head_size)

        K = K.view(
            batch_size, seq_len, self.n_head, n_embd // self.n_head
        )  # (batch_size, seq_len, n_head, head_size)
        K = K.transpose(1, 2)

        V = V.view(
            batch_size, seq_len, self.n_head, n_embd // self.n_head
        )  # (batch_size, seq_len, n_head, head_size)
        V = V.transpose(1, 2)

        attention_output = self._attention(Q, K, V, batch_size, seq_len, n_embd)

        output = self.c_proj(attention_output)

        return output


class GPT2MLP(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx=None):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(normalized_shape=config.n_embd)
        self.mlp = GPT2MLP(config, layer_idx=layer_idx)

    def forward(self, hidden_states):
        # input dimension: (batch_size, sequence_length, n_emd)
        residual = hidden_states
        hidden_states = self.ln_1(
            hidden_states
        )  # This is the input to the attention layer
        attn_output = self.attn(hidden_states)  # the size is (B, T, n_emd)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)  # This is the input to the MLP layer
        mlp_output = self.mlp(
            hidden_states
        )  # or feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states


class GPT2(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # word token embeddings
                wte=nn.Embedding(
                    num_embeddings=config.vocab_size, embedding_dim=config.n_embd
                ),
                # word position embeddings
                wpe=nn.Embedding(
                    num_embeddings=config.n_positions, embedding_dim=config.n_embd
                ),
                h=nn.ModuleList(
                    [GPT2Block(config, layer_idx=i) for i in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(
            in_features=config.n_embd, out_features=config.vocab_size, bias=False
        )

    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass of the GPT2 model. The forward pass of the GPT2 model consists of the following steps:
        1. Token Embeddings: The input sequence of tokens is passed through the token embeddings layer to get the token embeddings.
        2. Position Embeddings: The position embeddings are added to the token embeddings to get the input embeddings.
        3. GPT2 Block: The input embeddings are passed through the GPT2 block, which consists of a multi-head self-attention layer and a feed-forward neural network.
            3.1. Layer Normalization: The input embeddings are passed through a layer normalization layer.
            3.2. Multi-Head Self-Attention: The output of the layer normalization layer is passed through the multi-head self-attention layer to get the attention output.
            3.3. Residual Connection: The attention output is added to the input embeddings to get the residual output.
            3.4. Layer Normalization: The residual output is passed through a layer normalization layer.
            3.5. Feed-Forward Neural Network: The output of the layer normalization layer is passed through a feed-forward neural network to get the feed-forward output.
            3.6. Residual Connection: The feed-forward output is added to the residual output to get the output of the GPT2 block.
        4. Language Model Head: The output of the GPT2 block is passed through the language model head to get the logits for the next token.

        Args:
            input_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length) and dtype torch.int64 (LongTensor).

        """
        batch_size, sequence_length = input_ids.size()  # (B, T)

        _, sequence_length = input_ids.size()

        assert (
            sequence_length <= self.config.n_positions
        ), "Sequence length is longer than the maximum position"

        input_embeds = self.transformer.wte(
            input_ids
        )  # (batch_size, sequence_length, n_emd)

        # First this will be tested.
        # position_ids = torch.arange(start = 0, end = sequence_length, device=input_ids.device) # (sequence_length)
        position_ids = torch.arange(
            start=0, end=sequence_length, dtype=torch.long, device=input_ids.device
        )  # (sequence_length)

        # Another implementation
        # position_ids = torch.arange(start = 0, end = sequence_length, dtype=  torch.long,device=input_ids.device) # (sequence_length)
        # position_ids = position_ids.expand(batch_size, sequence_length) # (batch_size, sequence_length

        position_embeds = self.transformer.wpe(
            position_ids
        )  # (batch_size, sequence_length, n_emd)

        hidden_states = (
            input_embeds + position_embeds
        )  # (batch_size, sequence_length, n_emd)

        x = hidden_states  # (batch_size, sequence_length, n_emd) this is the input to the GPT Block

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_type: Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"] = "gpt2",
    ):

        config = {
            "gpt2": GPT2Config(),
            "gpt2-medium": GPT2Config(n_embd=1024, n_head=16, n_layer=24),
            "gpt2-large": GPT2Config(n_embd=1280, n_head=20, n_layer=36),
            "gpt2-xl": GPT2Config(n_embd=1600, n_head=25, n_layer=48),
        }[model_type]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        model = cls(config)

        sd = model.state_dict()
        sd_hf = model_hf.state_dict()

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        with torch.no_grad():
            for key, value in sd.items():
                if "attn.bias" in key:
                    if key.endswith("attn.c_attn.bias"):
                        value.copy_(sd_hf[key])
                else:
                    value_to_copy = (
                        sd_hf[key].t()
                        if any(key.endswith(suffix) for suffix in transposed)
                        else sd_hf[key]
                    )
                    value.copy_(value_to_copy)

        return model

    # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1577
    @classmethod
    @torch.inference_mode()
    def generate(
        cls,
        input_ids,
        max_length=30,
        max_new_tokens=None,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
    ):
        """
            Generate a sequence of tokens using the model.
            1. Initial Input: The process begins with an initial sequence of tokens represented by input_ids, which typically has a shape (batch_size, sequence_length).
            2. Token-by-Token Generation: The model generates new tokens one at a time. After generating each token, it appends the token to the input sequence and uses the updated sequence to generate the next token.
            3. Sequence Continuation: This process continues until the sequence reaches a specified maximum length, a stop token is generated, or another stopping criterion is met.

        Args:
            input_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length) and dtype torch.int64 (LongTensor).
            max_length (int): The maximum length of the sequence to be generated.
            num_return_sequences (int): The number of independently computed returned sequences for each element in the batch.
            do_sample (bool): If set to False greedy decoding is used. Otherwise, sampling is used.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filter

        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_length) and dtype torch.int64 (LongTensor).

        """
        # max_new_token = max_new_token or max_length # refactor this later
        # s.t.
        # max_new_tokens + input_ids.shape[1] = max_length

        input_len = input_ids.shape[1]  # (batch_size, sequence_length)

        model = cls.from_pretrained("gpt2")
        model.eval()

        device = input_ids.device
        model.to(device)

        x = input_ids
        while input_ids.shape[1] < max_length:
            logits = model(input_ids)  # (batch_size, sequence_length, vocab_size)
            next_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_probs = F.softmax(next_logits, dim=-1)  # (batch_size, vocab_size)

            if do_sample:
                top_k_probs, top_k_ids = torch.topk(
                    input=next_probs, k=top_k, dim=-1
                )  # (batch_size, top_k)

                idx = torch.multinomial(input=top_k_probs, num_samples=1)
                next_token = torch.gather(
                    input=top_k_ids, dim=-1, index=idx
                )  # (batch_size, 1)

            else:
                next_token = torch.argmax(next_probs, dim=-1)  # (batch_size,)
                next_token = next_token.unsqueeze(-1)  # (batch_size, 1)

            input_ids = torch.cat(
                (input_ids, next_token), dim=-1
            )  # (batch_size, sequence_length + 1)

        return input_ids
