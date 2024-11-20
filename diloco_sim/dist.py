import torch
from copy import deepcopy
from torch.utils.data import DataLoader, DistributedSampler
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from typing import Optional, Callable, Type
import numpy as np
import torch.multiprocessing as mp
from torch.distributed import init_process_group as init_process_group, destroy_process_group
import torch.distributed as dist
import signal
from dataclasses import dataclass


@dataclass
class TrainStats:
    loss: float
    local_step: int
    global_step: int
    epoch: int
    num_diloco_steps: int


@dataclass
class EvalStats:
    loss: float
    accuracy: float
    local_step: int
    global_step: int
    epoch: int
    num_diloco_steps: int


class DilocoSimulator:

    def __init__(
        self,
        model_cls: Type[torch.nn.Module],
        loss_fn: Callable[..., torch.Tensor],
        train_dataset: torch.utils.data.Dataset,
        model_kwargs: dict = {},
        num_nodes: int = 4,
        optimizer_kwargs: dict = {"lr": 0.001},
        diloco_interval: int = 500,
        batch_size: int = 16,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        ckpt_interval: Optional[int] = None,  # num of outersteps to save model
        eval_iters: int = 50,
        save_dir: Optional[str] = None,
        outer_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD,
        outer_optimizer_kwargs: dict = {"lr": 0.7, "nesterov": True, "momentum": 0.9},
        max_local_step: Optional[int] = None,
        num_epochs: int = 1,
        cosine_anneal: bool = False,
        train_loss_hook: Optional[Callable[[TrainStats], None]] = None,
        eval_loss_hook: Optional[Callable[[EvalStats], None]] = None,
    ) -> None:
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.outer_optimizer_cls = outer_optimizer_cls
        self.outer_optimizer_kwargs = outer_optimizer_kwargs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        self.num_nodes = num_nodes
        self.diloco_interval = diloco_interval
        self.ckpt_interval = ckpt_interval
        self.eval_iters = eval_iters
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.local_step: int = 0
        self.epoch: int = 0
        self.max_local_step = num_epochs * len(train_dataset) // (batch_size * num_nodes)
        if max_local_step:
            self.max_local_step = min(self.max_local_step, max_local_step)
        self.num_epochs = num_epochs
        self.cosine_anneal = cosine_anneal
        self.train_loss_hook = train_loss_hook
        self.eval_loss_hook = eval_loss_hook

    def _initialize_distributed(self, rank: int, model_path: Optional[str] = None):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12355"
        self.rank = rank
        init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            # init_method="env://",
            rank=rank,
            world_size=self.num_nodes,
        )
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device) if self.device.type == "cuda" else None

        self.model = self.model_cls(**self.model_kwargs).to(self.device)
        if self.rank == 0 and model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        for name, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        if self.cosine_anneal:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_local_step)

        if rank == 0:
            self.master_model = deepcopy(self.model).to("cpu")
            # Master model lives on CPU because only used for storage + outer opt (no matrix multiplies)
            for param in self.master_model.parameters():
                param.requires_grad = True
            self.master_optimizer = self.outer_optimizer_cls(
                self.master_model.parameters(), **self.outer_optimizer_kwargs
            )

        sampler = DistributedSampler(
            self.train_dataset, num_replicas=self.num_nodes, rank=rank, shuffle=True, drop_last=True
        )  # May want to do different data split between workers when looping over epochs
        self.train_dataloader: DataLoader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True
        )
        self.train_data_iter = iter(self.train_dataloader)

        if self.eval_dataset and rank == 0:
            self.eval_dataloader: DataLoader = DataLoader(
                self.eval_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True
            )
            self.eval_data_iter = iter(self.eval_dataloader)

    def _cleanup(self):
        if dist.is_initialized():
            destroy_process_group()

    def _save_model(self):
        name = f"iter_{self.local_step}"
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.save_dir}/model_{name}.pth")
        # TODO: better checkpointing (save local variables)

    def _outer_step(self):

        for param in self.model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)  # Sum all parameters
            param.data /= self.num_nodes

        if self.rank == 0:
            self.master_optimizer.zero_grad()

            for name, param in self.model.named_parameters():
                param.grad = self.master_model.state_dict()[name].data - param.data

            self.master_optimizer.step()

            for name, param in self.master_model.named_parameters():
                param.data = self.model.state_dict()[name].data.to("cpu")

        for name, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

    def _eval_model(self):

        self.model.eval()

        correct = 0
        losses = []
        with torch.no_grad():
            for i in range(self.eval_iters):
                x, y = self.get_batch(eval=True)

                output = self.model(x)
                loss = self.loss_fn(output, y)
                losses.append(loss.item())
                correct += (output.argmax(dim=1) == y).sum().item()

        avg_loss = sum(losses) / len(losses)
        loss_std = np.std(losses)
        accuracy = correct / (self.eval_iters * self.batch_size)

        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        if self.eval_loss_hook:
            self.eval_loss_hook(
                EvalStats(
                    loss=avg_loss,
                    accuracy=accuracy,
                    local_step=self.local_step,
                    global_step=self.local_step * self.num_nodes,
                    epoch=self.epoch,
                    num_diloco_steps=self.local_step // self.diloco_interval,
                )
            )

    def get_batch(self, eval=False):
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

    def _train_step(self):

        x, y = self.get_batch()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        if self.cosine_anneal:
            self.scheduler.step()

        return loss.item()

    def _train_loop(self):
        if self.rank == 0:
            print(f"Training for {self.max_local_step} steps")
            pbar = tqdm(total=self.max_local_step)

        self.model.train()

        while self.local_step < self.max_local_step:

            loss = self._train_step()

            if self.diloco_interval and self.local_step % self.diloco_interval == 0:
                self._outer_step()
                if self.rank == 0 and self.eval_dataset:
                    self._eval_model()
                dist.barrier()

            if self.rank == 0:

                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Epoch": self.epoch})

                if (
                    self.ckpt_interval
                    and self.save_dir
                    and self.local_step % (self.diloco_interval * self.ckpt_interval) == 0
                    and self.local_step > 0
                ):
                    self._save_model()

                if self.train_loss_hook:
                    self.train_loss_hook(
                        TrainStats(
                            loss=loss,
                            local_step=self.local_step,
                            global_step=self.local_step * self.num_nodes,
                            epoch=self.epoch,
                            num_diloco_steps=self.local_step // self.diloco_interval,
                        )
                    )

            self.local_step += 1

        if self.rank == 0:
            print("Training Complete")
            pbar.close()
            if self.save_dir:
                self._save_model()

    def _train(self, rank: int, model_path: Optional[str] = None):
        # signal.signal(signal.SIGINT, self._cleanup)
        # signal.signal(signal.SIGTERM, self._cleanup)
        try:
            self._initialize_distributed(rank, model_path)
            self._train_loop()
        finally:
            self._cleanup()

    def train(self, model_path: Optional[str] = None):
        torch.multiprocessing.spawn(self._train, args=(model_path,), nprocs=self.num_nodes, join=True)
