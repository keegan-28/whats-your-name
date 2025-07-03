import os
import sys

# add the root directory of the repo to the PYTHONPATH
# so that the scripts will be run as if from the root of the repo
# this means that posterior_models can be imported as per normal
sys.path.append(os.path.abspath(os.path.join("./")))

from character_models.abstract_classes import ModelConfig, save_model, load_model
from character_models.multilayer_perceptron import MLPDatasetLoader, MLPModel

from random import seed

(seed(47),)  # repeatable experiments
fname = "data/names.txt"
model_folder = "model_artifacts"
eos = "<END>"  # end of sequence or padding token
sos = "<START>"
pad = "<PAD>"
order = 6  # block_size
embedding_dim = 10
n_hidden = 100
seed = 47
minibatch_size = 32
max_steps = 200_000  # training minibatch steps

mlp_config = ModelConfig(
    order=order,
    embedding_dim=embedding_dim,
    n_hidden=n_hidden,
    seed=seed,
    minibatch_size=minibatch_size,
    model_folder=model_folder,
    max_steps=max_steps,
)

mlp_datasets = MLPDatasetLoader(fname, eos, sos, pad)

mlp_model = MLPModel(config=mlp_config, dataloader=mlp_datasets)

try:
    mlp_model.train()
except KeyboardInterrupt:
    pass
finally:
    save_model(mlp_model, model_folder + "/mlp")

mlp_model = load_model(model_folder + "/mlp")

for i in range(10):
    print("".join(mlp_model.sample_sequence()))

print(mlp_model.get_posterior(prefix=[0, 0, 0, 19, 8, 5]))  # s=19 h=8, e=5
