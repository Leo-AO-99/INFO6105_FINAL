import torch
from tqdm import tqdm
from tokenizer import Tokenizer
from transformer_model import Transformer
from config import ModelConfig
from torch import nn
import numpy as np
import uuid


from data import TranslationDatasetV2, get_dataloader, get_dataloader_v2



class Translator:
    def __init__(self, cpt_name: str = None):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = Tokenizer("./new_data/sp/src_sp.model", "./new_data/sp/tgt_sp.model")
        self.model_config = ModelConfig(
            src_vocab_size=self.tokenizer.src_vocab_size,
            tgt_vocab_size=self.tokenizer.tgt_vocab_size,
        )
        self.transformer = Transformer(
            self.model_config,
            self.tokenizer.src_pad_id,
            self.tokenizer.tgt_pad_id,
            self.device,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.transformer.parameters(), lr=1e-4)
        if cpt_name:
            checkpoint = torch.load(cpt_name)
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, num_epochs=10):
        # train_loader = get_dataloader("train", self.model_config.max_batch_size, False)
        train_loader = get_dataloader_v2("./new_data/sentence.txt", self.model_config.max_batch_size, True)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.tgt_pad_id)

        # Generate a unique identifier for the training session
        session_id = str(uuid.uuid4())
        for epoch in range(1, num_epochs + 1):
            self.transformer.train()
            train_losses = []
            total = len(train_loader)
            for i, batch in tqdm(enumerate(train_loader), total=total, desc="Processing", unit="item"):
                src_batch = batch['src']
                tgt_batch = batch['tgt']
                
                src_tokens = [self.tokenizer.encode_src(s, False, True) for s in src_batch]
                tgt_tokens = [self.tokenizer.encode_tgt(t, True, True) for t in tgt_batch]
                
                # Prepare input and target sequences
                tgt_input = [
                    t[:-1] for t in tgt_tokens
                ]  # Exclude the last token of each sequence
                tgt_target = [
                    t[1:] for t in tgt_tokens
                ]  # Exclude the first token of each sequence

                src_tokens = torch.tensor(
                    [self._preprocess_sequence(s) for s in src_tokens]
                ).to(self.device)
                tgt_input = torch.tensor(
                    [self._preprocess_sequence(t) for t in tgt_input]
                ).to(self.device)
                tgt_target = torch.tensor(
                    [self._preprocess_sequence(t) for t in tgt_target]
                ).to(self.device)
                
                output_logits = self.transformer(src_tokens, tgt_input)

                self.optim.zero_grad()
                
                loss = criterion(
                    output_logits.view(-1, self.tokenizer.tgt_vocab_size),
                    tgt_target.contiguous().view(-1),
                )
                
                loss.backward()
                self.optim.step()
                
                train_losses.append(loss.item())
                del src_tokens, tgt_input, tgt_target, output_logits
                torch.cuda.empty_cache()
                
            torch.save({
                'model_state_dict': self.transformer.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': np.mean(train_losses)
            }, f"checkpoint_{session_id}_{epoch}.pth")
                
                
    def _preprocess_sequence(self, sequence):
        if len(sequence) < self.model_config.max_seq_len:
            sequence += [self.tokenizer.src_pad_id] * (
                self.model_config.max_seq_len - len(sequence)
            )
        else:
            sequence = sequence[: self.model_config.max_seq_len]
        return sequence

    def translate(self, src_sentence: str) -> str:
        src_tokens = self.tokenizer.encode_src(src_sentence, False, False)
        src_tokens = (
            torch.LongTensor(self._preprocess_sequence(src_tokens))
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
            nopeak_mask = torch.ones(
                [1, self.model_config.max_seq_len, self.model_config.max_seq_len],
                dtype=torch.bool,
            ).to(self.device)
            nopeak_mask = torch.tril(nopeak_mask)
            d_mask = d_mask & nopeak_mask

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


if __name__ == "__main__":
    translator = Translator()
    translator.train(10)
    input_sentence = "I am a student"

    # translated_sentence = translator.translate(input_sentence)
    # print(translated_sentence)
