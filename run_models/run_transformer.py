import os
import sys

# add the root directory of the repo to the PYTHONPATH
# so that the scripts will be run as if from the root of the repo
# this means that posterior_models can be imported as per normal
sys.path.append(os.path.abspath(os.path.join("./")))

from character_models.abstract_classes import ModelConfig, save_model, load_model
from character_models.transformer_model import (
    SummaryWriter,
    TransformerDatasetLoader,
    TransformerModel,
    create_datasets,
)

import torch

# input_file='data/names.txt' # newline seperated input file of "words" to replicate
input_file = "calibration_experiments/data/train.txt"
run_type = "transformer-benchmark"
log_folder = f"tensorboard_logs/{run_type}"  # output working directory
os.makedirs(log_folder, exist_ok=True)
# Tensorboard logging of loss
writer = SummaryWriter(log_dir=log_folder)
model_folder = "../model_artifacts"

# system inits
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 3407
torch.cuda.manual_seed_all(seed)

eos = "<END>"  # end of sequence or padding token
sos = "<START>"
pad = "<PAD>"

# model
n_layer = 6  # Number of hidden layers in the Transformer encoder.
n_head = (
    4  # Number of attention heads for each attention layer in the Transformer encoder.
)
n_embd = 128  # Dimensionality of the embeddings and hidden states.
embd_pdrop = 0.1  # The dropout ratio for the embeddings.
resid_pdrop = (
    0.1  # The dropout probability for all fully connected layers in the encoder.
)
attn_pdrop = 0.1  # The dropout ratio for the attention.
bias = True

# sampling
temperature = 1

# optimization
max_steps = 20_000  # 40_000 # maximum number of optimisation steps.
# todo Should refactor this^^ to be interms of epochs
batch_size = 32  # batch size during optimization
learning_rate = 5e-4
weight_decay = 0.01
beta_1 = 0.9
beta_2 = 0.99

# dataset config
test_prop = 0.1
num_workers = 4  # number of data workers for the dataloaderboth train/test
train_dataset, test_dataset = create_datasets(input_file, test_prop, sos, eos, pad)
# set the maximum sequence length to the longest observed sequence
max_position_embeddings = train_dataset.get_output_length()
vocab_size = train_dataset.get_vocab_size()
print(
    f"dataset determined that: vocab_size:{vocab_size}, max_position_embeddings:{max_position_embeddings}"
)

transformer_config = ModelConfig(
    model_folder=model_folder,
    device=device,
    max_steps=max_steps,
    temperature=temperature,
    vocab_size=vocab_size,
    max_position_embeddings=max_position_embeddings,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    embd_pdrop=embd_pdrop,
    resid_pdrop=resid_pdrop,
    attn_pdrop=attn_pdrop,
    bias=True,
)

transformer_datasets = TransformerDatasetLoader(
    input_file, sos, eos, pad, test_prop, batch_size, num_workers, device
)

transformer_model = TransformerModel(
    config=transformer_config, dataloader=transformer_datasets, writer=writer
)

transformer_model.configure_optimiser(
    weight_decay, learning_rate, (beta_1, beta_2), device
)


try:
    transformer_model.train()
except KeyboardInterrupt:
    pass
finally:
    save_model(transformer_model, model_folder + "/transformer")

transformer_model = load_model(model_folder + "/transformer")

for i in range(10):
    print("".join(transformer_model.sample_sequence()))
