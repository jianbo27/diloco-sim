import torch
import os
from typing import Optional, Iterator, Tuple
from torch.distributed import init_process_group as init_process_group, destroy_process_group
import torch.distributed as dist
from copy import deepcopy
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from .config import DilocoSimulatorConfig


class SetupSimulator:
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

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f"model_{self.epoch}.pt"))

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

    def _setup(self, rank: int):
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
