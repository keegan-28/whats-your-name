from model import (
    evaluate,
    create_datasets,
    ModelConfig,
    CharacterTransformer,
    PerpetualData,
    print_samples,
)
import sys
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter

# =========== Config parameters

# set the working directory so that this file can be run portably from anywhere
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# system/input/output
input_file = "../data/names.txt"  # newline seperated input file of "words" to replicate
run_type = "dropout_weight_norm"
work_dir = f"out/{run_type}"  # output working directory

os.makedirs(work_dir, exist_ok=True)
# Tensorboard logging of loss
writer = SummaryWriter(log_dir=work_dir)

# system inits
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 3407
torch.cuda.manual_seed_all(seed)

# runtime config
resume = False  # when this flag is used, we will resume optimization from existing model in the workdir
sample_only = False  # just sample from the model and quit, don't train

# model
n_layer = 4  # Number of hidden layers in the Transformer encoder.
n_head = (
    4  # Number of attention heads for each attention layer in the Transformer encoder.
)
n_embd = 64  # Dimensionality of the embeddings and hidden states.
embd_pdrop = 0.1  # The dropout ratio for the embeddings.
resid_pdrop = (
    0.1  # The dropout probability for all fully connected layers in the encoder.
)
attn_pdrop = 0.1  # The dropout ratio for the attention.

# optimizer config
max_steps = (
    -1
)  # 200 # 40_000, max number of optimization steps to run for, or -1 for infinite.

# optimization
batch_size = 32  # batch size during optimization
learning_rate = 5e-4
weight_decay = 0.01
beta_1 = 0.9
beta_2 = 0.99

# dataset config
test_prop = 0.1
num_workers = 4  # number of data workers for the dataloaderboth train/test
train_dataset, test_dataset = create_datasets(input_file, test_prop)
vocab_size = train_dataset.get_vocab_size()
# set the maximum sequence length to the longest observed sequence
max_position_embeddings = train_dataset.get_output_length()
print(f"dataset determined that: {vocab_size=}, {max_position_embeddings=}")

# =========== Define and train the model

config = ModelConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_position_embeddings,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    embd_pdrop=embd_pdrop,
    resid_pdrop=resid_pdrop,
    attn_pdrop=attn_pdrop,
)
model = CharacterTransformer(config)
model.to(device)

if resume or sample_only:
    print("resuming from existing model in the workdir")
    model.load_state_dict(torch.load(os.path.join(work_dir, "model.pt")))
if sample_only:
    print_samples(model, device, train_dataset, test_dataset, num=50)
    sys.exit()

# init optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta_1, beta_2), device
)
# init dataloader
batch_loader = PerpetualData(
    train_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers
)

# load a single batch in order to populate the tensorboard graph
batch = batch_loader.next()
batch = [t.to(device) for t in batch]
writer.add_graph(model, input_to_model=batch, verbose=False, use_strict_trace=True)

# training loop
best_loss = None
step = 0
lossi = []

while True:
    t0 = time.time()
    # get the next batch, ship to device, and unpack it to input and target
    batch = batch_loader.next()
    # prepare the data for the specific hardware it is running on
    batch = [t.to(device) for t in batch]
    X, Y = batch

    # feed into the model
    logits, loss = model(X, Y)

    # calculate the gradient, update the weights
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    t1 = time.time()

    # wait for all CUDA work on the GPU to finish then calculate iteration time taken
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # logging to console
    if step % 500 == 0:
        print(
            f"step {step} | loss {loss.item():.4f} | step time {(t1 - t0) * 1000:.2f}ms"
        )

    # evaluate the model
    if step > 0 and step % 10 == 0:
        train_loss = evaluate(
            model, train_dataset, device, batch_size=100, max_batches=10
        )
        test_loss = evaluate(
            model, test_dataset, device, batch_size=100, max_batches=10
        )
        # track the results in tensorboard
        writer.add_scalar("Loss/train", train_loss, step)
        writer.add_scalar("Loss/test", test_loss, step)
        writer.add_scalars(
            "Combined", {"train_loss": train_loss, "test_loss": test_loss}, step
        )
        writer.flush()
        print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
        # if the model has improved, dump to disk
        if best_loss is None or test_loss < best_loss:
            out_path = os.path.join(work_dir, "model.pt")
            print(
                f"test loss {test_loss} is the best so far, saving model to {out_path}"
            )
            torch.save(model.state_dict(), out_path)
            best_loss = test_loss

    # At intervals sample from the model and log to the console
    if step > 0 and step % 100 == 0:
        samples = print_samples(model, device, train_dataset, test_dataset, num=50)
        writer.add_text("sample_name", samples[0], step)
        writer.flush()

    step += 1
    lossi.append(loss.log10().item())
    # termination conditions
    if max_steps >= 0 and step >= max_steps:
        break
