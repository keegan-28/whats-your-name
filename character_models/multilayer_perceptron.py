from random import seed
import pickle
from .abstract_classes import (
    ModelConfig,
    CharacterDatasetLoader,
    CharacterModel,
    TokenWithPosterior,
)
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from random import shuffle
import os

"""
Implement a simple multilayer perceptron nueral network according to the following reference
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""

"""
Implements a Markov chain style ngram model for predicting next character in a sequence
Usese the abstract base classes which define the interface (in abstract_classes.py)
"""

seed(42)


class MLPDatasetLoader(CharacterDatasetLoader):
    def __init__(self, file_name: str, eos: str, sos: str, pad: str) -> None:
        self.sos = sos
        self.pad = pad
        self.eos = eos

        self.lines = open(file_name).read().splitlines()
        self.chars = sorted(list(set("".join(self.lines))))
        self.define_codec()

    def define_codec(
        self,
    ) -> None:
        self.stoi = {ch: i + 3 for i, ch in enumerate(self.chars)}  # encode
        self.stoi[self.sos] = 0  # start token
        self.stoi[self.eos] = 1  # end token
        self.stoi[self.pad] = 2  # padding token
        self.vocab_size = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}  # decode

    def __call__(self, order: str) -> Dict[str, Tuple[torch.tensor, torch.tensor]]:
        shuffle(self.lines)
        n1 = int(0.8 * len(self.lines))
        n2 = int(0.9 * len(self.lines))

        Xtr, Ytr = self.build_dataset(self.lines[:n1], order)
        Xdev, Ydev = self.build_dataset(self.lines[n1:n2], order)
        Xte, Yte = self.build_dataset(self.lines[n2:], order)

        return {"train": (Xtr, Ytr), "test": (Xdev, Ydev), "val": (Xte, Yte)}

    def build_dataset(
        self, words: List[str], order: int
    ) -> Tuple[torch.tensor, torch.tensor]:
        X, Y = [], []
        for word in words:
            context = [0] * order
            for ch in list(word) + [self.eos]:
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  # crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y


class Tanh:
    def __call__(self, x: torch.tensor) -> torch.tensor:
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> List[Any]:
        return []


class MLPModel(CharacterModel):
    def __init__(self, *, config: ModelConfig, dataloader: MLPDatasetLoader) -> None:
        super().__init__(config=config, dataloader=dataloader)
        self.model = None
        self.model_name = "mlp"
        self.model_binary_path = os.path.join(
            self.config.model_folder, "tmp", self.model_name + f"_{self.config.order}"
        )

    def train(self) -> None:
        self.datasets = self.dataloader(self.config.order)

        Xtr, Ytr = self.datasets["train"]

        g = torch.Generator().manual_seed(self.config.seed)
        C = torch.randn(
            (self.dataloader.vocab_size, self.config.embedding_dim), generator=g
        )

        layers = [
            nn.Linear(
                self.config.embedding_dim * self.config.order,
                self.config.n_hidden,
                bias=False,
            ),
            nn.BatchNorm1d(self.config.n_hidden),
            Tanh(),
            nn.Linear(self.config.n_hidden, self.config.n_hidden, bias=False),
            nn.BatchNorm1d(self.config.n_hidden),
            Tanh(),
            nn.Linear(self.config.n_hidden, self.config.n_hidden, bias=False),
            nn.BatchNorm1d(self.config.n_hidden),
            Tanh(),
            nn.Linear(self.config.n_hidden, self.config.n_hidden, bias=False),
            nn.BatchNorm1d(self.config.n_hidden),
            Tanh(),
            nn.Linear(self.config.n_hidden, self.config.n_hidden, bias=False),
            nn.BatchNorm1d(self.config.n_hidden),
            Tanh(),
            nn.Linear(self.config.n_hidden, self.dataloader.vocab_size, bias=False),
            nn.BatchNorm1d(self.dataloader.vocab_size),
        ]

        self.model = {"C": C, "layers": layers}

        parameters = [C] + [p for layer in layers for p in layer.parameters()]
        print(sum(p.nelement() for p in parameters))  # number of parameters in total
        for p in parameters:
            p.requires_grad = True

        lossi = []
        ud = []
        best_loss = None

        for i in range(self.config.max_steps):
            # minibatch construct
            ix = torch.randint(
                0, Xtr.shape[0], (self.config.minibatch_size,), generator=g
            )
            Xb, Yb = Xtr[ix], Ytr[ix]  # batch X,Y

            # forward pass
            emb = C[Xb]  # embed the characters into vectors
            x = emb.view(emb.shape[0], -1)  # concatenate the vectors
            for layer in layers:
                x = layer(x)
            loss = nn.functional.cross_entropy(x, Yb)  # loss function

            for p in parameters:
                p.grad = None
            loss.backward()

            # update
            lr = 0.1 if i < 150000 else 0.01  # step learning rate decay
            for p in parameters:
                p.data += -lr * p.grad

            # track stats
            if i % 10000 == 0:  # print every once in a while
                print(f"{i:7d}/{self.config.max_steps:7d}: {loss.item():.4f}")

            test_loss = self.score_model("test")
            if i % 500 == 0:
                if best_loss is None or test_loss < best_loss:
                    print(
                        f"test loss {test_loss:.3f} is the best so far, saving model to {self.model_binary_path}"
                    )
                    self.save_model_binary()
                    best_loss = test_loss
            lossi.append(loss.log10().item())
            with torch.no_grad():
                ud.append(
                    [
                        ((lr * p.grad).std() / p.data.std()).log10().item()
                        for p in parameters
                    ]
                )

        print(f"{self.model_name} has finished its training")
        print("train loss: ", self.score_model("train"))
        print("test loss: ", self.score_model("test"))
        print("val loss: ", self.score_model("val"))

    def score_model(self, split: str) -> float:
        x, y = self.datasets[split]
        C = self.model["C"]
        layers = self.model["layers"]

        for layer in layers:
            layer.training = False

        emb = C[x]  # (N, block_size, self.config.embedding_dim)
        x = emb.view(
            emb.shape[0], -1
        )  # concat into (N, block_size * self.config.embedding_dim)
        for layer in layers:
            x = layer(x)
        loss = nn.functional.cross_entropy(x, y)
        return loss.item()

    def save_model_binary(
        self,
    ) -> None:
        if not self.model:
            raise Exception("No model has been trained")
        os.makedirs(os.path.dirname(self.model_binary_path), exist_ok=True)
        with open(self.model_binary_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"{self.model_name} saved")

    def load_model_binary(
        self,
    ) -> None:
        with open(self.model_binary_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"{self.model_name} loaded")

    def sample_sequence(self, *, max_len: Optional[int] = None) -> List[str]:
        sequence_with_posteriors = self.sample_sequence_with_posteriors(max_len=max_len)
        return [elem.token for elem in sequence_with_posteriors]

    def sample_sequence_with_posteriors(
        self, *, max_len: Optional[int] = None
    ) -> List[TokenWithPosterior]:
        prefix = [0] * self.config.order  # precondition on the <.> tokens index
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
            prefix = prefix[-self.config.order :] + [
                idx
            ]  # crop sequence to the order of the model
            token_and_posterior = TokenWithPosterior(
                token=token_x, posterior=token_posterior
            )
            sequence.append(token_and_posterior)
        return sequence

    def get_posterior(self, *, prefix: List[int]) -> torch.tensor:
        prefix = prefix[-self.config.order :]

        C = self.model["C"]
        layers = self.model["layers"]

        for layer in layers:
            layer.training = False

        # forward pass the neural net
        emb = C[torch.tensor([prefix])]  # (1,block_size,self.config.embedding_dim)
        x = emb.view(emb.shape[0], -1)  # concatenate the vectors
        for layer in layers:
            x = layer(x)
        logits = x
        posterior = nn.functional.softmax(logits, dim=1)

        return posterior
