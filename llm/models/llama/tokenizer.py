from sentencepiece import SentencePieceProcessor
from init import init

config = init()


def load_tokenizer(
    tokenizer_path=f"{config.base_dir}/.cache/meta_llama2/tokenizer.model",
):

    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size()

    return tokenizer, vocab_size


if __name__ == "__main__":
    _, vocab_size = load_tokenizer()  # 32000 in case of llama-2-7b
    print(vocab_size)
