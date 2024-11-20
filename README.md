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

The full list of available arguments is shown below:

| **Argument**         | **Type**                | **Description**                                                                 |
|-----------------------|-------------------------|---------------------------------------------------------------------------------|
| `model_cls`          | `Type[torch.nn.Module]` | The model class to be instantiated and trained. Must be a subclass of `nn.Module`. |
| `model_kwargs`       | `dict`                 | Keyword arguments to initialize the model. Example: `{"num_classes": 100}`.     |
| `num_nodes`          | `int`                  | Number of nodes (simulated workers) in the distributed system.                  |
| `optimizer_kwargs`   | `dict`                 | Keyword arguments for the optimizer. Example: `{"lr": 0.001}`.                 |
| `diloco_interval`    | `int`                  | Number of local steps before synchronizing the models.                          |
| `batch_size`         | `int`                  | Batch size for training and evaluation.                                         |
| `loss_fn`            | `Callable[..., torch.Tensor]` | The loss function used during training. Example: `torch.nn.functional.cross_entropy`. |
| `train_dataset`      | `torch.utils.data.Dataset` | The dataset for training. Should be a subclass of `Dataset`.                   |
| `eval_dataset`       | `Optional[torch.utils.data.Dataset]` | The dataset for evaluation. Optional.                                           |
| `optimizer_cls`      | `Type[torch.optim.Optimizer]` | Optimizer class for local training. Default is `torch.optim.AdamW`.             |
| `ckpt_interval`      | `Optional[int]`        | Number of outer steps between model checkpoints. Default is `None`.             |
| `eval_iters`         | `int`                  | Number of iterations to use for evaluation. Default is `50`.                   |
| `save_dir`           | `Optional[str]`        | Directory to save model checkpoints. Default is `None`.                        |
| `outer_optimizer_cls` | `Type[torch.optim.Optimizer]` | Optimizer class for outer training. Default is `torch.optim.SGD`.               |
| `outer_optimizer_kwargs` | `dict`                 | Keyword arguments for the outer optimizer. Example: `{"lr": 0.7, "nesterov": True, "momentum": 0.9}`. |
| `max_local_step`     | `Optional[int]`        | Maximum number of local steps to train. Default is `None`.                      |
| `num_epochs`         | `int`                  | Total number of training epochs.                                                |
| `cosine_anneal`      | `bool`                 | Whether to use cosine annealing for learning rate scheduling. Default is `False`. |
| `train_loss_hook`    | `Optional[Callable[[TrainStats], None]]` | Function to call after each local step. Default is `None`.                     |
| `eval_loss_hook`     | `Optional[Callable[[EvalStats], None]]` | Function to call after each evaluation. Default is `None`.                     |