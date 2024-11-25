import os
from typing import List, Union
from sentencepiece import SentencePieceProcessor

class Tokenizer:
    """Tokenizer for Chinese-English translation using SentencePiece."""
    def __init__(self, src_model_path: str, tgt_model_path: str):
        assert os.path.isfile(src_model_path), f"src_model_path: {src_model_path} is not a file"
        assert os.path.isfile(tgt_model_path), f"tgt_model_path: {tgt_model_path} is not a file"
        self.src_sp_model = SentencePieceProcessor(model_file=src_model_path)
        self.tgt_sp_model = SentencePieceProcessor(model_file=tgt_model_path)
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"  # Beginning of sentence
        self.eos_token = "[EOS]"  # End of sentence
        
        # Special token IDs for source language
        self.src_pad_id = self.src_sp_model.piece_to_id(self.pad_token)
        self.src_unk_id = self.src_sp_model.piece_to_id(self.unk_token)
        self.src_bos_id = self.src_sp_model.piece_to_id(self.bos_token)
        self.src_eos_id = self.src_sp_model.piece_to_id(self.eos_token)
        
        # Special token IDs for target language
        self.tgt_pad_id = self.tgt_sp_model.piece_to_id(self.pad_token)
        self.tgt_unk_id = self.tgt_sp_model.piece_to_id(self.unk_token)
        self.tgt_bos_id = self.tgt_sp_model.piece_to_id(self.bos_token)
        self.tgt_eos_id = self.tgt_sp_model.piece_to_id(self.eos_token)
        
        self.src_vocab_size = self.src_sp_model.get_piece_size()
        self.tgt_vocab_size = self.tgt_sp_model.get_piece_size()

    def encode_src(self, s: str, bos: bool, eos: bool) -> List[int]:
        """Convert source text to token IDs."""
        assert type(s) is str
        t = self.src_sp_model.encode(s)
        if bos:
            t = [self.src_bos_id] + t
        if eos:
            t = t + [self.src_eos_id]
        return t
    
    def encode_tgt(self, s: str, bos: bool, eos: bool) -> List[int]:
        """Convert target text to token IDs."""
        assert type(s) is str
        t = self.tgt_sp_model.encode(s)
        if bos:
            t = [self.tgt_bos_id] + t
        if eos:
            t = t + [self.tgt_eos_id]
        return t
   
    def decode_src(self, t: List[int]) -> str:
        """Convert source token IDs back to text."""
        return self.src_sp_model.decode_ids(t)
    
    def decode_tgt(self, t: List[int]) -> str:
        """Convert target token IDs back to text."""
        return self.tgt_sp_model.decode_ids(t)

    def tokenize_src(self, text: str) -> List[str]:
        """Convert source text to tokens (subwords)."""
        assert type(text) is str
        return self.src_sp_model.encode_as_pieces(text)
        
    def tokenize_tgt(self, text: str) -> List[str]:
        """Convert target text to tokens (subwords)."""
        assert type(text) is str
        return self.tgt_sp_model.encode_as_pieces(text)
    def switchToTensor():
        source_Token=SentencePieceProcessor(model_file="source_sp.model")
        target_tokenizer = SentencePieceProcessor(model_file="target_sp.model")

if __name__ == "__main__":
    tokenizer = Tokenizer(src_model_path='./data/sp/src_sp.model', tgt_model_path='./data/sp/tgt_sp.model')
    t = tokenizer.encode_src("The tendency is either", bos=True, eos=True)
    print(t)
    print(tokenizer.decode_src(t))
    t = tokenizer.encode_tgt("我 的 世 界", True, True)
    print(t)
    print(tokenizer.decode_tgt(t))
    # print(tokenizer.tokenize_tgt("你好，世界！"))
