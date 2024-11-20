import torch
import torch.nn.functional as F
from diloco_sim import DilocoSimulator
from nanogpt import GPTConfig, GPT
from data import TextDataset
import numpy as np
import argparse
from util import arg_combinations, str2bool, generate_text
import random
import torch.autograd.profiler as profiler


def CELoss(inputs, targets):
    return F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))


if __name__ == "__main__":

    gptconf = GPTConfig(block_size=256, vocab_size=50304, n_layer=2, n_head=4, n_embd=128)

    train_dataset = TextDataset(
        "data/owt/openwebtext.bin",
        dtype=np.uint16,
        seq_length=gptconf.block_size,
    )

    simulator = DilocoSimulator(
        model_cls=GPT,
        model_kwargs={
            "config": gptconf,
        },
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.0005},
        train_dataset=train_dataset,
        loss_fn=CELoss,
    )

    simulator.train()
