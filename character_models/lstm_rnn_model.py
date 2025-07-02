"""
Implements a Long-Term Short Term (LSTM) Recurrent Nueral Network in accordance with
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from .abstract_classes import (
    ModelConfig,
    CharacterDatasetLoader,
    CharacterModel,
    TokenWithPosterior,
)
import os
from typing import Optional, List, Dict, Tuple


class LSTMDataset(Dataset):
    def __init__(self, lines: List[List[str]], stoi: Dict[str, int]) -> None:
        self.lines = lines
        self.stoi = stoi

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> torch.LongTensor:
        line = self.lines[idx]
        chars = torch.LongTensor([self.stoi[char] for char in line])
        input_seq = chars[:-1]
        target_seq = chars[1:]
        return input_seq, target_seq


def collate_fn(batch: List[torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
    # Sort sequences by length in descending order.
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # Separate input and target sequences.
    input_seqs, target_seqs = zip(*batch)

    # Pad input and target sequences to the maximum sequence length.
    # pad the sequence with the <PAD> character at the end of the seqence
    pad_input_seqs = torch.nn.utils.rnn.pad_sequence(
        input_seqs, batch_first=True, padding_value=2
    )
    pad_target_seqs = torch.nn.utils.rnn.pad_sequence(
        target_seqs, batch_first=True, padding_value=2
    )

    return pad_input_seqs, pad_target_seqs


class CharTextGenerationLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        drop_prob: float,
        device: str,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        ## Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

    def forward(self, x: torch.tensor, hidden=None) -> torch.tensor:
        if hidden is None:
            hidden = self.init_hidden(x.shape[0])
        x = self.embedding(x)
        out, (h_n, c_n) = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        ## Pass x through the dropout layer
        x = self.dropout(x)
        out = self.fc(out)
        return out, (h_n, c_n)

    def init_hidden(self, batch_size: int) -> Tuple[torch.tensor, torch.tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h0, c0


class LSTMDatasetLoader(CharacterDatasetLoader):
    def __init__(
        self, file_name: str, batch_size: int, eos: str, sos: str, pad: str
    ) -> None:
        self.sos = sos
        self.pad = pad
        self.eos = eos

        self.batch_size = batch_size
        lines = open(file_name).read().splitlines()
        self.chars = sorted(list(set("".join(lines))))
        self.define_codec()

        self.lines = [[self.sos] + list(line) + [self.eos] for line in lines]
        self.max_word_length = max(len(line) for line in self.lines)

    def define_codec(
        self,
    ) -> None:
        self.stoi = {ch: i + 3 for i, ch in enumerate(self.chars)}  # encode
        self.stoi[self.sos] = 0  # start token
        self.stoi[self.eos] = 1  # end token
        self.stoi[self.pad] = 2  # padding token
        self.vocab_size = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}  # decode

    def __call__(self) -> DataLoader:
        dataset = LSTMDataset(self.lines, self.stoi)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn
        )
        return dataloader


class LSTMModel(CharacterModel):
    def __init__(self, *, config: ModelConfig, dataloader: LSTMDatasetLoader) -> None:
        super().__init__(config=config, dataloader=dataloader)
        self.model = CharTextGenerationLSTM(
            self.dataloader.vocab_size,
            self.config.embedding_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.drop_prob,
            self.config.device,
        ).to(self.config.device)
        self.model_name = "lstm"
        self.model_binary_path = os.path.join(
            self.config.model_folder, "tmp", self.model_name
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.itos = self.dataloader.itos
        self.stoi = self.dataloader.stoi
        self.sos = self.dataloader.sos
        self.eos = self.dataloader.eos
        self.pad = self.dataloader.pad

    def train(self) -> None:
        self.datasets = self.dataloader()

        lossi = []
        best_loss = None
        self.model.train()
        for epoch in range(self.config.epochs):
            running_loss = 0
            for input_seq, target_seq in self.datasets:
                input_seq, target_seq = (
                    input_seq.to(self.config.device),
                    target_seq.to(self.config.device),
                )
                outputs, _ = self.model(input_seq)
                loss = F.cross_entropy(outputs, target_seq.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.detach().cpu().numpy()
            epoch_loss = running_loss / len(self.datasets)
            print(f"Epoch {epoch} loss: {epoch_loss:.3f}")
            if best_loss is None or loss < best_loss:
                print(
                    f"test loss {loss:.3f} is the best so far, saving model to {self.model_binary_path}"
                )
                self.save_model_binary()
                best_loss = loss
            lossi.append(loss.log10().item())

    def save_model_binary(self) -> None:
        os.makedirs(os.path.dirname(self.model_binary_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_binary_path)

    def load_model_binary(self) -> None:
        self.model.load_state_dict(torch.load(self.model_binary_path))

    def sample_sequence_with_posteriors(
        self, *, max_len: Optional[int] = None
    ) -> List[TokenWithPosterior]:
        prefix = [0]  # <START>
        sequence = []
        while True:
            token_posterior = self.get_posterior(prefix=prefix)
            idx = torch.multinomial(
                token_posterior, replacement=True, num_samples=1
            ).item()
            token_x = self.dataloader.itos[idx]
            if token_x == self.dataloader.eos:
                break
            if max_len:
                if max_len == len(sequence):
                    break
            # guard against infinite sequences caused by repeatedly predicting the <PAD> token
            # only occurs in models which haven't been trained for more than a handful of epochs
            if token_x == self.dataloader.pad:
                break
            prefix += [idx]
            token_and_posterior = TokenWithPosterior(
                token=token_x, posterior=token_posterior
            )
            sequence.append(token_and_posterior)
        return sequence

    def get_posterior(self, *, prefix: List[int]) -> torch.tensor:
        self.model.eval()
        h, c = self.model.init_hidden(1)
        input_seq = torch.LongTensor(prefix).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            output, (h, c) = self.model(input_seq, (h, c))
        return F.softmax(output, dim=-1)[-1, :]  # only the final prediction
