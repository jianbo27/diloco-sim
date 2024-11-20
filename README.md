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

| **Argument**         | **Type**                | **Default**                  | **Description**                                                                 |
|-----------------------|-------------------------|------------------------------|---------------------------------------------------------------------------------|
| `model_cls`          | `Type[torch.nn.Module]` | **Required**                | The model class to be instantiated and trained. Must be a subclass of `nn.Module`. |
| `loss_fn`            | `Callable[..., torch.Tensor]` | **Required**                | The loss function used during training. Example: `torch.nn.functional.cross_entropy`. Must be of form (x,y) => loss |
| `train_dataset`      | `torch.utils.data.Dataset` | **Required**                | The dataset for training. Should be a subclass of `Dataset`.                   |
| `model_kwargs`       | `dict`                 | `{}`                        | Keyword arguments to initialize the model. Example: `{"num_classes": 100, ...}`.     |
| `num_nodes`          | `int`                  | `4`                         | Number of nodes (simulated workers) in the distributed system.                  |
| `optimizer_kwargs`   | `dict`                 | `{"lr": 0.001}`             | Keyword arguments for the inner optimizer. Example: `{"lr": 0.001}`.                 |
| `diloco_interval`    | `int`                  | `500`                       | Number of local steps before synchronizing the models.                          |
| `batch_size`         | `int`                  | `16`                        | Batch size for training and evaluation.                                         |
| `eval_dataset`       | `Optional[torch.utils.data.Dataset]` | `None`                       | The dataset for evaluation. Optional.                                           |
| `optimizer_cls`      | `Type[torch.optim.Optimizer]` | `torch.optim.AdamW`         | Inner Optimizer class for training. `AdamW` is default per recommendation of DiLiCo.                                            |
| `ckpt_interval`      | `Optional[int]`        | `None`                      | Number of outer steps between model checkpoints. Default is `None`.             |
| `eval_iters`         | `int`                  | `50`                        | Number of iterations to use for evaluation. Loss is approximated by `eval_iters * batch_size` samples. Default is `50`.                   |
| `save_dir`           | `Optional[str]`        | `None`                      | Directory to save model checkpoints. Default is `None`.                        |
| `outer_optimizer_cls` | `Type[torch.optim.Optimizer]` | `torch.optim.SGD`           | Optimizer class for outer training. Default is `SGD` per recommnedation of DiLoCo.                |
| `outer_optimizer_kwargs` | `dict`                 | `{"lr": 0.7, "nesterov": True, "momentum": 0.9}` | Keyword arguments for the outer optimizer. Nesterov momentum is default per recommendation of DiLoCo.      |
| `max_local_step`     | `Optional[int]`        | `None`                      | Maximum number of local steps to train. Default is `None`. If specified, training will stop after this many local steps if it occurs before the end of `num_epochs` epochs. |
| `num_epochs`         | `int`                  | `1`                        | Total number of training epochs.                                                |
| `cosine_anneal`      | `bool`                 | `False`                     | Whether to use cosine annealing for learning rate scheduling. Default is `False`. |
| `train_loss_hook`    | `Optional[Callable[[TrainStats], None]]` | `None`                      | Function to call after each local step. Default is `None`.                     |
| `eval_loss_hook`     | `Optional[Callable[[EvalStats], None]]` | `None`                      | Function to call after each evaluation. Default is `None`.                     |
