import math
from dataclasses import dataclass
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


@dataclass(frozen=True)
class ModelConfig:
    """
    Dataclass to contain what are effectively the model hyperparameters
    Note that the defaults contained here produce a very small <500k paramters
    model and will only be reasonable to very simple (toy) examples.
    """

    # The maximum sequence length that this model might ever be used with.
    # "<.>"" (start) token and 20 characters length of the input sequences of integers
    max_position_embeddings: int = 20 + 1

    # Defines the number of different tokens that can be represented by the `inputs_ids`
    # the input integers are in range [0 .. vocab_size -1]
    vocab_size: int = None

    # Number of hidden layers in the Transformer encoder.
    n_layer: int = 4
    # Number of attention heads for each attention layer in the Transformer encoder.
    n_head: int = 4
    # Dimensionality of the embeddings and hidden states.
    n_embd: int = 64

    embd_pdrop: float = 0.1  # The dropout ratio for the embeddings.
    resid_pdrop: float = (
        0.1  # The dropout probability for all fully connected layers in the encoder.
    )
    attn_pdrop: float = 0.1  # The dropout ratio for the attention.

    # Adds a bias to any of the Linear Layers and LayerNorms as in GPT2
    # If set to False the model is faster
    bias: bool = True


# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)


class LayerNorm(nn.Module):
    """
    Implement a custom LayerNorm which has an optional bias.
    The Pytorch documentation seems to be wrong and doesn't behave as I would expect
    # todo: validate in the PT source code that there is an error with the Pytorch docs and submit PR
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class AttentionMechanism(nn.Module):
    """
    Multi-head self-attention layer (masked) and projection.
    Explicity haven't used torch.nn.MultiheadAttention as we may want to play around with some of the internals
    here, different dropout schemes etc.
    """

    def __init__(self, config):
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

    def forward(self, x):
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

    def __init__(self, config):
        super().__init__()

        # default the dimensionality of the inner Feed Forward layers to 4 x n_embed
        # this is what was done in https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py
        # is this appriate for character transformers of this type? who knows ...
        # todo: pull out this factor as a hyperparamter and look at implications of the intermediate embedding size.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
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

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = AttentionMechanism(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = CTMLP(config)

    def forward(self, x):
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

    def __init__(self, config):
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

    def _init_weights(self, module):
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
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

    def get_max_position_embeddings(self):
        return self.config.max_position_embeddings

    def forward(self, idx, targets=None):
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
    def generate(self, idx, temperature=1.0):
        """
        Take a sequence of (conditional) indices idx (LongTensor of shape (b,t)) and complete
        inference of the next character produced until the end of sequence "<.>" token is predicted

            * The outputs of the sampling are iteratively fed into the model each time.
        """
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
            masked = torch.eq(idx_next, torch.zeros(idx.size(0), 1))
            sampling_ongoing.masked_fill_(masked, 0)

            # when all of the sequences in the batch have observed eos (<.>) then return
            if sampling_ongoing.sum().item() == 0:
                return idx

            # else append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)


def print_samples(model, device, train_dataset, test_dataset, num=10):
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
        crop_index = row.index(0) if 0 in row else len(row)
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
def evaluate(model, dataset, device, batch_size=50, max_batches=None):
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


class CharacterDataset(Dataset):
    """
    Class for preparing, loading, wrangling, en/decoding text datasets at a character token level.
    """

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}  # encode
        self.stoi["<.>"] = 0  # start token
        self.itos = {i: s for s, i in self.stoi.items()}  # decode

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1  # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1  # <.> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = "".join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index idx
        Handles all required padding with 0 and -1
        """
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1 : 1 + len(ix)] = ix  # x is <.> token followed by characters
        y[: len(ix)] = ix  # y is characters followed by padding
        y[len(ix) + 1 :] = -1  # index -1 will mask the loss at the inactive locations
        return x, y


def create_datasets(input_file, test_prop):
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
    train_dataset = CharacterDataset(train_words, chars, max_word_length)
    test_dataset = CharacterDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset


class PerpetualData:
    """
    Class which supports (infinite) repeated sampling of the dataset
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:  # handle the rare case of num_samples being exceeded ... by starting from the beginning
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
