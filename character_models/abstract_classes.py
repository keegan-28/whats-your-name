from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import List, Optional
import pickle


@dataclass
class TokenWithPosterior:
    token: str
    posterior: torch.tensor


class ModelConfig:
    """
    Class the keeps track of the relevant paremeter for a given model
    Eg.
        # model order
        # max_sequence length
        # dropout/layers etc
    """

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class CharacterDatasetLoader(ABC):
    def __init__(self, file_name: str, eos: str) -> None:
        self.lines = open(file_name).read().splitlines()
        self.chars = sorted(list(set("".join(self.lines))))
        self.eos = eos
        self.max_word_length = max(len(line) for line in self.lines)
        self.define_codec()

    def define_codec(
        self,
    ) -> None:
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.chars)}  # encode
        self.stoi[self.eos] = 0  # start token
        self.vocab_size = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}  # decode

    @abstractmethod
    def __call__(
        self,
    ) -> List[torch.tensor]:
        raise NotImplementedError


class CharacterModel(ABC):
    def __init__(
        self, *, config: ModelConfig, dataloader: CharacterDatasetLoader
    ) -> None:
        self.config = config
        self.dataloader = dataloader

    def sample_sequence(self, *, max_len: Optional[int] = None) -> List[str]:
        sequence_with_posteriors = self.sample_sequence_with_posteriors(max_len=max_len)
        return [elem.token for elem in sequence_with_posteriors]

    @abstractmethod
    def sample_sequence_with_posteriors(
        self, *, max_len: Optional[int] = None
    ) -> List[TokenWithPosterior]:
        # get posteriour
        # for a multinomial sample from it
        raise NotImplementedError()

    @abstractmethod
    def get_posterior(self, *, prefix: List[int]) -> torch.tensor:
        raise NotImplementedError()

    @abstractmethod
    def train(
        self,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_model_binary(
        self,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model_binary(
        self,
    ) -> None:
        raise NotImplementedError


def save_model(model: CharacterModel, file_name: str) -> None:
    model.load_model_binary()
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name: str) -> CharacterModel:
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model
