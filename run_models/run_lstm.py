import os
import sys

# add the root directory of the repo to the PYTHONPATH
# so that the scripts will be run as if from the root of the repo
# this means that posterior_models can be imported as per normal
sys.path.append(os.path.abspath(os.path.join("./")))

from character_models.abstract_classes import ModelConfig, save_model, load_model
from character_models.lstm_rnn_model import LSTMModel, LSTMDatasetLoader

from random import seed
import torch

# system inits
device = "cuda" if torch.cuda.is_available() else "cpu"
seed(47)  # repeatable experiments

# system/input/output
input_file = "data/names.txt"
run_type = "lstm"  # "lstm-basic"
model_folder = "model_artifacts"

eos = "<END>"  # end of sequence or padding token
sos = "<START>"
pad = "<PAD>"
batch_size = 32
embedding_dim = 32  # set to the vocab size
hidden_size = 256
num_layers = 3
drop_prob = 0.1

# Training Setup
learning_rate = 0.01
epochs = 1_000

lstm_config = ModelConfig(
    model_folder=model_folder,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    drop_prob=drop_prob,
    device=device,
    learning_rate=learning_rate,
    epochs=epochs,
)

lstm_datasets = LSTMDatasetLoader(input_file, batch_size, eos, sos, pad)

lstm_model = LSTMModel(config=lstm_config, dataloader=lstm_datasets)

try:
    lstm_model.train()
except KeyboardInterrupt:
    pass
finally:
    save_model(lstm_model, model_folder + "/lstm")

lstm_model = load_model(model_folder + "/lstm")

for i in range(10):
    print("".join(lstm_model.sample_sequence()))
