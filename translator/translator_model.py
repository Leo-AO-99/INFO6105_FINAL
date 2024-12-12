from typing import Literal
import uuid
import torch
from tqdm import tqdm
from tokenizer import Tokenizer
from transformer_model import Transformer
from config import ModelConfig
from torch import nn
import numpy as np

from data import get_dataloaderV2


class Translator:
    def __init__(self, norm_way: Literal["LN", "RMS"], cpt_name: str = None):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = Tokenizer("./data/sp/src_sp.model", "./data/sp/tgt_sp.model")
        self.model_config = ModelConfig(
            src_vocab_size=self.tokenizer.src_vocab_size,
            tgt_vocab_size=self.tokenizer.tgt_vocab_size,
            norm_way=norm_way,
        )
        self.transformer = Transformer(
            self.model_config,
            self.tokenizer.src_pad_id,
            self.tokenizer.tgt_pad_id,
            self.device,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.transformer.parameters(), lr=self.model_config.optim_lr)
        if cpt_name:
            checkpoint = torch.load(cpt_name, weights_only=False, map_location=self.device)
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, num_epochs=10):
        train_loader = get_dataloaderV2("./data/src.txt", "./data/tgt.txt", self.model_config.max_batch_size, True)
        criterion = nn.NLLLoss()
        session_id = uuid.uuid4()
        for epoch in range(1, num_epochs + 1):
            self.transformer.train()
            train_losses = []
            total_batches = len(train_loader)
            for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", unit="batch", total=total_batches):
                src_batch = batch['src']
                tgt_batch = batch['tgt']
                
                src_tokens = [self.tokenizer.encode_src(s, False, True) for s in src_batch] # for transformer input
                train_src = torch.tensor([self._preprocess_sequence(s, "src") for s in src_tokens]).to(self.device)
                
                train_tgt_tokens = [self.tokenizer.encode_tgt(t, True, False) for t in tgt_batch] # for transformer output
                valid_tgt_tokens = [self.tokenizer.encode_tgt(t, False, True) for t in tgt_batch] # for loss calculation
                train_tgt = torch.tensor([self._preprocess_sequence(t, "tgt") for t in train_tgt_tokens]).to(self.device)
                valid_tgt = torch.tensor([self._preprocess_sequence(t, "tgt") for t in valid_tgt_tokens]).to(self.device)

                output_logits = self.transformer(train_src, train_tgt)

                self.optim.zero_grad()
                loss = criterion(
                    output_logits.view(-1, self.tokenizer.tgt_vocab_size),
                    valid_tgt.contiguous().view(valid_tgt.size(0) * valid_tgt.size(1)),
                )
                
                loss.backward()
                self.optim.step()
                
                train_losses.append(loss.item())
                del src_tokens, train_src, train_tgt_tokens, train_tgt, valid_tgt_tokens, valid_tgt, output_logits
                torch.cuda.empty_cache()

            avg_loss = np.mean(train_losses)
            print(f"Epoch {epoch}, Average Loss: {avg_loss}")
            torch.save({
                'model_state_dict': self.transformer.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': avg_loss
            }, f"checkpoint_{session_id}_{epoch}_.pth")
                
                
    def _preprocess_sequence(self, sequence, sequence_type: Literal["src", "tgt"]):
        if sequence_type == "src":
            pad_id = self.tokenizer.src_pad_id
        elif sequence_type == "tgt":
            pad_id = self.tokenizer.tgt_pad_id
        else:
            raise ValueError(f"Invalid sequence type: {sequence_type}")
        
        if len(sequence) < self.model_config.max_seq_len:
            sequence += [pad_id] * (
                self.model_config.max_seq_len - len(sequence)
            )
        else:
            sequence = sequence[: self.model_config.max_seq_len]
        return sequence

    def inference(self, src_sentence: str) -> str:
        src_tokens = self.tokenizer.encode_src(src_sentence, False, False)
        src_tokens = (
            torch.LongTensor(self._preprocess_sequence(src_tokens, 'src'))
            .unsqueeze(0)
            .to(self.device)
        )
        src_pad_mask = (
            (src_tokens != self.tokenizer.src_pad_id).unsqueeze(1).to(self.device)
        )
        src_tokens = self.transformer.src_embedding(src_tokens)
        src_tokens = self.transformer.positional_encoder(src_tokens)
        e_output = self.transformer.encoder(src_tokens, src_pad_mask)

        result_sequence = torch.LongTensor(
            [self.tokenizer.tgt_pad_id] * self.model_config.max_seq_len
        ).to(self.device)
        result_sequence[0] = self.tokenizer.tgt_bos_id
        cur_len = 1

        for i in range(self.model_config.max_seq_len):
            d_mask = (
                (result_sequence.unsqueeze(0) != self.tokenizer.tgt_pad_id)
                .unsqueeze(1)
                .to(self.device)
            )
            lookahead_mask = torch.ones(
                [1, self.model_config.max_seq_len, self.model_config.max_seq_len],
                dtype=torch.bool,
            ).to(self.device)
            lookahead_mask = torch.tril(lookahead_mask)
            d_mask = d_mask & lookahead_mask

            tgt_tokens = self.transformer.tgt_embedding(result_sequence.unsqueeze(0))
            tgt_tokens = self.transformer.positional_encoder(tgt_tokens)
            decoder_output = self.transformer.decoder(
                tgt_tokens, e_output, d_mask, src_pad_mask
            )

            output = self.transformer.softmax(
                self.transformer.output_layer(decoder_output)
            )

            output = torch.argmax(output, dim=-1)
            latest_word_id = output[0][i].item()

            if i < self.model_config.max_seq_len - 1:
                result_sequence[i + 1] = latest_word_id
                cur_len += 1
            if latest_word_id == self.tokenizer.tgt_eos_id:
                break
        if result_sequence[-1].item() == self.tokenizer.tgt_pad_id:
            result_sequence = result_sequence[1:cur_len].tolist()
        else:
            result_sequence = result_sequence[1:].tolist()
        return self.tokenizer.decode_tgt(result_sequence)