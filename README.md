# LLM Exploration

## GSM8K Evaluation

script:`llm/evaluate/gsm8k.py`



## PyTroch

- using **in-place** methods performs better than using **out-of-place** methods. `output/experiments/{inplace, outplace}`.

- **Contiguous** tensors performs better than non-contiguous. `output/experiments/{contiguity}`. (using `torch.contiguous()` and `torch.view()`)


## KV cache

| Model                    | 16FP                | KIVI                |
| ------------------------ | ------------------- | ------------------- |
| Meta-Llama-3-8B          | 0.49683544303797467 |                     |
| Meta-Llama-3-8B-Instruct | 0.7554179566563467  |                     |
| Llama-2-7b-hf            | 0.1342925659472422  | 0.10454908220271349 |
| Llama-2-7b-chat-hf       | 0.21674418604651163 | 0.1759927797833935  |
| Mistral-7B-v0.1          | 0.43967611336032386 | 0.4080971659919028  |
| Mistral-7B-Instruct-v0.2 | 0.45616883116883117 | 0.41804635761589404 |
| OLMo-1.7-7B-hf           | 0.2793950075512405  |                     |






## Llama models

- Original implementation: `llm/models/llama/meta/model.py`
- Single node implementation: `llm/models/llama/meta/model_single_node.py`







## References
Template has been taken from [here](https://github.com/Guangxuan-Xiao/GSM8K-eval).