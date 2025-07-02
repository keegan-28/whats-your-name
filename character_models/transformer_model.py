from .abstract_classes import (
    ModelConfig,
    CharacterDatasetLoader,
    CharacterModel,
    TokenWithPosterior,
)
from typing import List, Tuple, Optional
import os

import math
import inspect
import time

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


class CharacterDataset(Dataset):
    """
    Class for preparing, loading, wrangling, en/decoding text datasets at a character token level.
    """

    def __init__(self, words, chars, max_word_length, sos, eos, pad) -> None:
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 3 for i, ch in enumerate(chars)}  # encode
        self.stoi[sos] = 0
        self.stoi[eos] = 1
        self.stoi[pad] = 2
        self.itos = {i: s for s, i in self.stoi.items()}  # decode

    def __len__(self) -> int:
        return len(self.words)

    def contains(self, word) -> bool:
        return word in self.words

    def get_vocab_size(self) -> int:
        return len(self.chars) + 3  # all the possible characters and special 0 token

    def get_output_length(self) -> int:
        return self.max_word_length + 3  # <START> + word + <END> + <PAD>

    def encode(self, word) -> torch.tensor:
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix) -> str:
        word = "".join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns a sample from the dataset at the given index idx
        Handles all required padding with 0 and 1
        """
        word = self.words[idx]

        ix = self.encode(word)
        x = torch.zeros(
            self.max_word_length + 3, dtype=torch.long
        )  # add room for a <start> word <end> <pad>
        y = torch.zeros(
            self.max_word_length + 3, dtype=torch.long
        )  # add room for a <start> word <end> <pad>

        x[1 : len(ix) + 1] = ix  # x is <.> token followed by characters
        x[len(ix) + 1] = 1  # eos
        x[len(ix) + 2 :] = 2  # will mask the loss at the inactive locations
        y[: len(ix)] = ix  # y is characters followed by padding
        y[len(ix)] = 1  # eos
        y[len(ix) + 1 :] = 2  # mask the loss at the inactive locations
        return x, y


def create_datasets(
    input_file, test_prop, sos, eos, pad
) -> Tuple[CharacterDataset, CharacterDataset]:
    """
    Clean and curate training and testing datasets
    """
    # preprocessing of the input text file
    with open(input_file, "r") as f:
        data = f.read()

    words = data.splitlines()
    # remove whitespace and empty strings
    words = [w.strip() for w in words]
    words = [w for w in words if w]
    chars = sorted(list(set("".join(words))))  # all the possible characters
    max_word_length = max(len(w) for w in words)

    # report key dataset statistics
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print(f"vocabulary: {''.join(chars)}")

    test_set_size = int(len(words) * test_prop)
    rp = torch.randperm(len(words)).tolist()  # shuffle the dataset
    # train and test split
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(
        f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples"
    )

    # wrap in dataset objects
    train_dataset = CharacterDataset(train_words, chars, max_word_length, sos, eos, pad)
    test_dataset = CharacterDataset(test_words, chars, max_word_length, sos, eos, pad)

    return train_dataset, test_dataset


class PerpetualData:
    """
    Class which supports (infinite) repeated sampling of the dataset
    """

    def __init__(self, dataset, **kwargs) -> None:
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self) -> List[torch.tensor]:
        # returns [x: torch.tensor, y: torch.tensor]
        try:
            batch = next(self.data_iter)
        except StopIteration:  # handle the rare case of num_samples being exceeded ... by starting from the beginning
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


class TransformerDatasetLoader(CharacterDatasetLoader):
    def __init__(
        self,
        input_file: str,
        sos: str,
        eos: str,
        pad: str,
        test_prop: float,
        batch_size: int,
        num_workers: int,
        device: str,
    ) -> None:
        """
        Clean and curate training and testing datasets
        """
        self.lines = open(input_file).read().splitlines()
        self.chars = sorted(list(set("".join(self.lines))))
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.max_word_length = max(len(line) for line in self.lines)

        self.stoi = {ch: i + 3 for i, ch in enumerate(self.chars)}  # encode
        self.stoi[self.sos] = 0
        self.stoi[self.eos] = 1
        self.stoi[self.pad] = 2
        self.vocab_size = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}  # decode
        print(self.itos)

        # report key dataset statistics
        print(f"number of examples in the dataset: {len(self.lines)}")
        print(f"max word length: {self.max_word_length}")
        print(f"number of unique characters in the vocabulary: {len(self.chars)}")
        print(f"vocabulary: {''.join(self.chars)}")

        test_set_size = int(len(self.lines) * test_prop)
        rp = torch.randperm(len(self.lines)).tolist()  # shuffle the dataset
        # train and test split
        train_words = [self.lines[i] for i in rp[:-test_set_size]]
        test_words = [self.lines[i] for i in rp[-test_set_size:]]
        print(
            f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples"
        )

        # wrap in dataset objects
        self.train_dataset = CharacterDataset(
            train_words, self.chars, self.max_word_length, sos, eos, pad
        )
        self.test_dataset = CharacterDataset(
            test_words, self.chars, self.max_word_length, sos, eos, pad
        )
        self.perpeptual_data = PerpetualData(
            self.train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
        )
        self.device = device

    def __call__(
        self,
    ) -> List[torch.tensor]:
        batch = self.perpeptual_data.next()
        # prepare the data for the specific hardware it is running on
        batch = [t.to(self.device) for t in batch]
        return batch


class TransformerModel(CharacterModel):
    def __init__(
        self,
        *,
        config: ModelConfig,
        dataloader: TransformerDatasetLoader,
        writer: SummaryWriter,
    ) -> None:
        super().__init__(config=config, dataloader=dataloader)
        self.model = CharacterTransformer(config)
        self.model_name = "transformer"
        self.model_binary_path = os.path.join(
            self.config.model_folder, "tmp", self.model_name
        )
        self.optimizer = None
        self.writer = writer
        self.itos = self.dataloader.itos
        self.stoi = self.dataloader.stoi
        self.sos = self.dataloader.sos
        self.eos = self.dataloader.eos
        self.pad = self.dataloader.pad

    def configure_optimiser(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device: str,
    ) -> None:
        self.optimizer = self.model.configure_optimizers(
            weight_decay, learning_rate, (betas[0], betas[1]), device
        )

    def train(self) -> None:
        # training loop
        best_loss = None
        step = 0
        lossi = []
        max_steps = self.config.max_steps
        while True:
            t0 = time.time()

            X, Y = self.dataloader()  # batch

            # feed into the model
            logits, loss = self.model(X, Y)

            # calculate the gradient, update the weights
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            t1 = time.time()

            # wait for all CUDA work on the GPU to finish then calculate iteration time taken
            if self.config.device.startswith("cuda"):
                torch.cuda.synchronize()

            # logging to console
            if step % 500 == 0:
                print(
                    f"step {step} | loss {loss.item():.4f} | step time {(t1 - t0) * 1000:.2f}ms"
                )

            # evaluate the model
            if step > 0 and step % 100 == 0:
                train_loss = evaluate(
                    self.model,
                    self.dataloader.train_dataset,
                    self.config.device,
                    batch_size=100,
                    max_batches=10,
                )
                test_loss = evaluate(
                    self.model,
                    self.dataloader.test_dataset,
                    self.config.device,
                    batch_size=100,
                    max_batches=10,
                )
                # track the results in tensorboard
                self.writer.add_scalar("Loss/train", train_loss, step)
                self.writer.add_scalar("Loss/test", test_loss, step)
                self.writer.add_scalars(
                    "Combined", {"train_loss": train_loss, "test_loss": test_loss}, step
                )
                self.writer.flush()
                print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
                # if the model has improved, dump to disk
                if best_loss is None or test_loss < best_loss:
                    print(
                        f"test loss {loss:.3f} is the best so far, saving model to {self.model_binary_path}"
                    )
                    self.save_model_binary()
                    best_loss = test_loss

            # At intervals sample from the model and log to the console
            if step > 0 and step % 1_000 == 0:
                samples = print_samples(
                    self.model,
                    self.config.device,
                    self.dataloader.train_dataset,
                    self.dataloader.test_dataset,
                    num=50,
                )
                self.writer.add_text("sample_name", samples[0], step)
                self.writer.flush()

            step += 1
            lossi.append(loss.log10().item())
            # termination conditions
            if max_steps >= 0 and step >= max_steps:
                break

    def __getstate__(self) -> Dict[str, Any]:
        # dataloaders can't be picked so need to remove the
        # dataloader from the state of the model when picling
        # and unpickling
        self_dict = self.__dict__.copy()

        del self_dict["dataloader"]
        del self_dict["writer"]
        return self_dict

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def save_model_binary(self) -> None:
        os.makedirs(os.path.dirname(self.model_binary_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_binary_path)

    def load_model_binary(self) -> None:
        self.model.load_state_dict(torch.load(self.model_binary_path))

    def sample_sequence_with_posteriors(
        self, *, max_len: Optional[int] = None
    ) -> List[TokenWithPosterior]:
        num_samples = 1
        idx = torch.zeros(
            num_samples, 1, dtype=torch.long
        ).to(
            self.config.device
        )  # initialiase tensor([[0],[0] ... [0]]) - 1 x num samples tensor of start tokens

        sequence: List[TokenWithPosterior] = []
        while True:
            # if the sequence context is growing too long we must crop it at max_position_embeddings
            idx_cond = (
                idx
                if idx.size(1) <= self.config.max_position_embeddings
                else idx[:, -self.config.max_position_embeddings :]
            )
            posterior = self.get_posterior(
                prefix=idx_cond.tolist()
            )  # forward the model to get the logits for the index in the sequence
            idx_next = torch.multinomial(posterior, num_samples=1)

            # when all of the sequences in the batch have observed eos (<END>) then return
            if idx_next.item() == 1:
                break
            if max_len:
                if max_len == len(sequence):
                    break

            token_and_posterior = TokenWithPosterior(
                token=self.itos[idx_next.item()], posterior=posterior
            )
            sequence.append(token_and_posterior)
            # else append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return sequence

    def get_posterior(self, *, prefix: List[int]) -> torch.tensor:
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(prefix))
        # extract the logits and scale by desired temperature
        logits = logits[:, -1, :] / self.config.temperature
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)

        return probs


# ======================================
#    Character Tranfromer Definition
# ======================================


class LayerNorm(nn.Module):
    """
    Implement a custom LayerNorm which has an optional bias.
    The Pytorch documentation seems to be wrong and doesn't behave as I would expect
    # todo: validate in the PT source code that there is an error with the Pytorch docs and submit PR
    """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.tensor) -> torch.tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class AttentionMechanism(nn.Module):
    """
    Multi-head self-attention layer (masked) and projection.
    Explicity haven't used torch.nn.MultiheadAttention as we may want to play around with some of the internals
    here, different dropout schemes etc.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        assert (
            config.n_embd % config.n_head == 0
        )  # composite embedding is made by concantenating the projections from each attention head

        # batching the key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # linear projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # FLASH ATTENTION
        # PyTorch >= 2.0 can support flash attention which is a significantly more compute
        # efficient way of computing causal self attention
        # https://github.com/Dao-AILab/flash-attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Install PyTorch >= 2.0 in order to use flash attention")
            # add causal masking so that attention is only applied to the LHS of the input sequence
            # tensor manipulate is a bit dense so you have to think a bit
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(
                        config.max_position_embeddings, config.max_position_embeddings
                    )
                ).view(
                    1, 1, config.max_position_embeddings, config.max_position_embeddings
                ),
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and reshape head to be the batch dim
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.attn_pdrop if self.training else 0,
                is_causal=True,
            )
        else:
            # implement our own self attention (slow version)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # reshape the outputs of hte head to be side by side -> (B, T, C)

        # project with and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class CTMLP(nn.Module):
    """
    A simple Multilayer perceptron
        * using Gausian Error Linear Units as the activation
        * Applying dropout
        * Using a fixed (for now) intermediate FF layer size of 4*n_embed
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # default the dimensionality of the inner Feed Forward layers to 4 x n_embed
        # this is what was done in https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py
        # is this appriate for character transformers of this type? who knows ...
        # todo: pull out this factor as a hyperparamter and look at implications of the intermediate embedding size.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CTBlock(nn.Module):
    """
    A transformer layer implimenting
    * Layer Norming (with optional bias)
    * Attention
    * FF NN
    * Residual (skip connections)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = AttentionMechanism(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = CTMLP(config)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # include residual connections to improve the flow of gradients
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CharacterTransformer(nn.Module):
    """
    CharacterTransformer is a simple implementation of a transformer model as seen in GPT-2
    It includes some additions standard improvement which are incorporated in the SOTA
    * Weight tying
    * Flash Attention
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
                wpe=nn.Embedding(
                    self.config.max_position_embeddings, self.config.n_embd
                ),
                drop=nn.Dropout(self.config.embd_pdrop),
                h=nn.ModuleList([CTBlock(config) for _ in range(self.config.n_layer)]),
                ln_f=LayerNorm(self.config.n_embd, bias=self.config.bias),
            )
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # add weight tying between the input embedding projection and the output
        # probability prediction. It is effectively a form of regularisation as described
        # here https://arxiv.org/abs/1608.05859v3
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights, and apply a scaled init to the residual projections,
        self.apply(self._init_weights)
        # follow the GPT-2 paper and apply the scaled initialisation
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )

        # calculate the number of model parameters
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(
            "The number of parameters in the CharacterTransformer is: %.2fM"
            % (n_params / 1e6,)
        )

    def _init_weights(self, module: nn.Module) -> None:
        """ "
        Initialise the linear and embedding layers using a Gaussian distribution
        * Optional bias for the Linear layers
        todo: incorporate SOTA weight initialisation techniques
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        # inspect all model parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # group the parameters for optimisation
        # weight decay all tensor using in matrix multiplication and embeddings
        # all over biases and layernorms don't layer decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # report parameter statistics
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"Total # decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"Total # non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Create AdamW  (weight decaying) optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, eps=1e-8, **extra_args
        )

        return optimizer

    def get_max_position_embeddings(self) -> int:
        return self.config.max_position_embeddings

    def forward(self, idx: torch.tensor, targets=None) -> torch.tensor:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_position_embeddings, (
            "Error: Sequence is greater than maximum sequence length"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the CharacterTransformer model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # scoring a target sequence by calculating the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # only compute the forward -> the lm_head on the very last position (time saver)
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # -----------------------------------------------------------------------------
    # helper functions for evaluating and sampling from the model

    @torch.no_grad()
    def generate(self, idx: torch.tensor, temperature=1.0) -> torch.tensor:
        """
        Take a sequence of (conditional) indices idx (LongTensor of shape (b,t)) and complete
        inference of the next character produced until the end of sequence "<.>" token is predicted

            * The outputs of the sampling are iteratively fed into the model each time.
        """
        i = 0
        sampling_ongoing = torch.ones(
            idx.size(0), 1
        )  # mask indicating that sampling is still ongoing 0 means finished
        while True:
            # if the sequence context is growing too long we must crop it at max_position_embeddings
            idx_cond = (
                idx
                if idx.size(1) <= self.config.max_position_embeddings
                else idx[:, -self.config.max_position_embeddings :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # find all the sequences in the batch that are terminating and record them
            masked = torch.eq(idx_next, torch.ones(idx.size(0), 1))
            sampling_ongoing.masked_fill_(masked, 0)

            # when all of the sequences in the batch have observed eos (<.>) then return
            if sampling_ongoing.sum().item() == 0:
                return idx

            if i > 20:
                return idx
            # else append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            i += 1


def print_samples(
    model: CharacterTransformer,
    device: str,
    train_dataset: CharacterDataset,
    test_dataset: CharacterDataset,
    num: int = 10,
) -> List[str]:
    """
    Function to sample from the model
        * Decodes the samples
        * Counts how many originate in each of the test/train datasets
    """

    X_init = torch.zeros(num, 1, dtype=torch.long).to(
        device
    )  # initialiase tensor([[0],[0] ... [0]]) - 1 x num tensor of start tokens
    X_samp = model.generate(X_init).to("cpu")
    train_samples, test_samples, new_samples = [], [], []
    for i in range(num):
        # get the i'th row of sampled integers, as python list
        row = X_samp[
            i, 1:
        ].tolist()  # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(1) if 1 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)

        # tally samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)

    # report the samples that have been taken
    print("-" * 80)
    for lst, desc in [
        (train_samples, "in train"),
        (test_samples, "in test"),
        (new_samples, "new"),
    ]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print("-" * 80)

    return train_samples + test_samples + new_samples


@torch.inference_mode()
def evaluate(
    model: CharacterTransformer,
    dataset: CharacterDataset,
    device: str,
    batch_size: int = 50,
    max_batches: Optional[int] = None,
) -> float:
    """
    Function to evaluated the average cross entropy loss across a labelled dataset
    """
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()  # reset model back to training mode
    return mean_loss
