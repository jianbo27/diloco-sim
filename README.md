# Distributed Low-Communication (DiLoCo) Training Simulator

diloco-sim is a simulator for the DiLoCo algorithm, which is a distributed training algorithm that synchronizes models every n steps instead of every step.

diloco-sim merely simulates this distributed network, but the workers may or may not be running on the same machine, depending on how many devices are available.

Example usage can be found in the `examples` directory.

The minimal arguments for training are shown below:

```python

from diloco_sim import DilocoSimulator
from models import ModelArchitecture
import torch.nn.functional as F
from data import train_dataset, test_dataset

simulator = DilocoSimulator(
    model_cls=CNNModel,
    model_kwargs={"num_classes": 100, ...},
    optimizer_kwargs={"lr": 0.001},
    train_dataset=train_dataset,
    loss_fn=F.cross_entropy,
    num_nodes=4,
    diloco_interval=500,
    batch_size=16,
)

simulator.train()

```