import json
from pathlib import Path

import torch
from Config import ModelArgs
from init import init
from model import Transformer
from tokenizer import load_tokenizer

config = init()

tokenizer, vocab_size = load_tokenizer()


def load_model(llama_path=Path(f"{config.base_dir}/.cache/meta_llama2/llama-2-7b/")):

    max_seq_len = 20 # To be fixed
    max_batch_size = 5 # To be fixed

    (
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        if config.device.type == "cuda"
        else torch.set_default_tensor_type(torch.BFloat16Tensor)
    )

    checkpoints_path = sorted(
        (llama_path).glob("*.pth")
    )  # For llama-2-7b, there is no need to sort the checkpoints since there is only one checkpoint.

    assert len(checkpoints_path) > 0, f"No checkpoints found in {checkpoints_path}"

    checkpoint = torch.load(
        checkpoints_path[0], map_location="cpu"
    )  # Load the checkpoint on CPU, [0] since there is only one checkpoint
    # Comment from Meta repo: The only unmatched key in the checkpoint is rope.freqs. Remove it
    del checkpoint["rope.freqs"]  # Remove the unmatched key

    with open(llama_path / "params.json", "r") as f:  # Load the params
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=config.device,
        **params,
    )
    model_args.vocab_size = vocab_size

    model = Transformer(model_args).to(config.device)

    model.load_state_dict(checkpoint, strict=True)

    print("Checkpoint loaded successfully")

    return model


if __name__ == "__main__":
    model = load_model()
    print(model)
