import torch
import os
from typing import Optional, Callable, Type, Iterator, Tuple
from torch.distributed import init_process_group as init_process_group, destroy_process_group
import torch.distributed as dist
from dataclasses import dataclass, field
from copy import deepcopy
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


@dataclass
class DilocoSimulatorConfig:
    model_cls: Type[torch.nn.Module]
    model_kwargs: dict
    loss_fn: Callable[..., torch.Tensor]
    train_dataset: torch.utils.data.Dataset
    optimizer_kwargs: dict
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    batch_size: int = 16
    eval_dataset: Optional[torch.utils.data.Dataset] = None
    ckpt_interval: Optional[int] = None  # num of outersteps to save model
    eval_iters: int = 50
    save_dir: Optional[str] = None
    num_epochs: int = 1
    cosine_anneal: bool = False
    model_path: Optional[str] = None
    num_nodes: int = 4
    diloco_interval: int = 500
    outer_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD
    outer_optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 0.7, "nesterov": True, "momentum": 0.9})


class DilocoSimulator:
    rank: int
    device: torch.device
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
    master_model: torch.nn.Module
    master_optimizer: torch.optim.Optimizer
    train_dataloader: DataLoader
    eval_dataloader: Optional[DataLoader] = None
    train_data_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]
    eval_data_iter: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None
    max_local_step: int
    local_step: int = 0
    epoch: int = 0
    pbar: tqdm

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        self.config = config
        self.max_local_step = (
            self.config.num_epochs * len(self.config.train_dataset) // (self.config.batch_size * self.config.num_nodes)
        )

    def _initialize_distributed(self, rank: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12355"
        self.rank = rank
        init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            # init_method="env://",
            rank=rank,
            world_size=self.config.num_nodes,
        )
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device) if self.device.type == "cuda" else None

    def _cleanup(self):
        if dist.is_initialized():
            destroy_process_group()

    def _setup_master_model(self):
        self.master_model = deepcopy(self.model).to("cpu")
        for param in self.master_model.parameters():
            param.requires_grad = True

    def _setup_master_optimizer(self):
        self.master_optimizer = self.config.outer_optimizer_cls(
            self.master_model.parameters(), **self.config.outer_optimizer_kwargs
        )

    def _outer_step(self):
        for param in self.model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)  # Sum all parameters
            param.data /= self.config.num_nodes

        if self.rank == 0:
            self.master_optimizer.zero_grad()

            for name, param in self.model.named_parameters():
                param.grad = self.master_model.state_dict()[name].data.to(device=param.device) - param.data

            self.master_optimizer.step()

            for name, param in self.master_model.named_parameters():
                param.data = self.model.state_dict()[name].data.to("cpu")

        for name, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

    def _setup_model(self):
        self.model = self.config.model_cls(**self.config.model_kwargs).to(self.device)
        for name, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

    def _setup_optimizer(self):
        self.optimizer = self.config.optimizer_cls(self.model.parameters(), **self.config.optimizer_kwargs)

    def _setup_scheduler(self):
        self.scheduler = (
            CosineAnnealingLR(self.optimizer, T_max=self.max_local_step) if self.config.cosine_anneal else None
        )

    def _setup_train_dataloader(self):
        sampler = DistributedSampler(
            self.config.train_dataset, num_replicas=self.config.num_nodes, rank=self.rank, shuffle=True, drop_last=True
        )  # May want to do different data split between workers when looping over epochs
        self.train_dataloader = DataLoader(
            self.config.train_dataset, batch_size=self.config.batch_size, sampler=sampler, pin_memory=True
        )
        self.train_data_iter = iter(self.train_dataloader)

    def _setup_eval_dataloader(self):
        self.eval_dataloader = DataLoader(
            self.config.eval_dataset, batch_size=self.config.batch_size, pin_memory=True, shuffle=True
        )
        self.eval_data_iter = iter(self.eval_dataloader)

    def _get_batch(self, eval=False):
        if not eval:
            try:
                x, y = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                x, y = next(self.train_data_iter)
        else:
            try:
                x, y = next(self.eval_data_iter)
            except StopIteration:
                self.eval_data_iter = iter(self.eval_dataloader)
                x, y = next(self.eval_data_iter)

        x, y = x.to(self.device), y.to(self.device)

        return x, y

    def _train_loss_hook(self, loss):
        self.pbar.update(1)
        self.pbar.set_postfix({"Loss": f"{loss:.4f}", "Epoch": self.epoch})

    def _eval_loss_hook(self, loss, accuracy):
        print(f"Eval Loss: {loss:.4f}")
        print(f"Eval Accuracy: {accuracy:.4f}")

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f"model_{self.epoch}.pt"))

    def _eval_model(self):
        self.model.eval()

        correct = 0
        losses = []
        with torch.no_grad():
            for _ in range(self.config.eval_iters):
                x, y = self._get_batch(eval=True)
                output = self.model(x)
                loss = self.config.loss_fn(output, y)
                losses.append(loss.item())
                correct += (output.argmax(dim=1) == y).sum().item()

        avg_loss = sum(losses) / len(losses)
        accuracy = correct / (self.config.eval_iters * self.config.batch_size)

        self._eval_loss_hook(avg_loss, accuracy)

        self.model.train()

    def _train_step(self):
        x, y = self._get_batch()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.config.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        self._train_loss_hook(loss)

    def _train_loop(self):

        self.pbar = tqdm(total=self.max_local_step)
        self.model.train()

        while self.local_step < self.max_local_step:

            self._train_step()

            if self.local_step % self.config.diloco_interval == 0:
                self._outer_step()
                if self.rank == 0 and self.config.eval_dataset:
                    self._eval_model()
                dist.barrier()

            self.local_step += 1

        self.pbar.close()

    def _train(self, rank: int):
        try:
            self._initialize_distributed(rank)
            self._setup_model()
            self._setup_optimizer()
            self._setup_scheduler()
            self._setup_train_dataloader()
            if self.rank == 0:
                self._setup_master_model()
                self._setup_master_optimizer()
                if self.config.eval_dataset:
                    self._setup_eval_dataloader()

            self._train_loop()

            if self.rank == 0 and self.config.save_dir:
                self._save_checkpoint()
        finally:
            self._cleanup()

    def train(self):
        torch.multiprocessing.spawn(self._train, args=(), nprocs=self.config.num_nodes, join=True)


class TrainingMixin:
    