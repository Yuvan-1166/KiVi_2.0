import sentencepiece as spm

class Tokenizer:
    def __init__(self, model: str):
        self.sp = spm.SentencePieceProcessor(model_file=model)
    def encode(self, text: str) -> str:
        return self.sp.encode(text)
    def decode(self, tokens: list) -> str:
        return self.sp.decode(tokens)