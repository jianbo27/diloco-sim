import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from typing import Optional, Callable, Type
from diloco_sim.util import (
    parameter_correlation,
    mean_squared_difference,
    cosine_similarity,
    euclidean_distance,
    time_function,
)
import numpy as np
import torch.multiprocessing as mp
from torch.distributed import init_process_group as init_process_group


class DilocoSimulator:

    def __init__(
        self,
        model_cls: Type[torch.nn.Module],
        model_kwargs: dict,
        loss_fn: Callable[..., torch.Tensor],
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: dict = {"lr": 0.001},
        world_size: int = 4,
        diloco_interval: int = 100,
        ckpt_interval: Optional[int] = None,  # num of outersteps to save model
        eval_iters: int = 50,
        batch_size: int = 16,
        save_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,  # WandB project name, pass None to disable logging
        wandb_config: Optional[dict] = None,
        outer_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD,
        outer_optimizer_kwargs: dict = {"lr": 0.7, "nesterov": True, "momentum": 0.9},
        max_local_step: Optional[int] = None,
        num_epochs: int = 1,
        cosine_anneal: bool = False,
        log_stats_interval: int = 10,
        device: Optional[torch.device | str] = None,
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
        self.world_size = world_size
        self.diloco_interval = diloco_interval
        self.ckpt_interval = ckpt_interval
        self.eval_iters = eval_iters
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config
        self.local_step: int = 0
        self.epoch: int = 0
        self.max_local_step = num_epochs * len(train_dataset) // (batch_size * world_size)
        if max_local_step:
            self.max_local_step = min(self.max_local_step, max_local_step)
        self.num_epochs = num_epochs
        self.cosine_anneal = cosine_anneal
        self.log_stats_interval = log_stats_interval

        # Initialize Master Model

        self.model = self.model_cls(**self.model_kwargs)
        for param in self.model.parameters():
            param.requires_grad = True

        # self.optimizer = None

        # self.train_dataloader: DataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        # self.train_data_iter = iter(self.train_dataloader)

        # if self.eval_dataset:
        #     self.eval_dataloader: DataLoader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=True)
        #     self.eval_data_iter = iter(self.eval_dataloader)

        # self.losses: list[float] = []
        # self.grad_norms: list[float] = []

        wandb_config = {
            "batch_size": self.batch_size,
            "inner_optimizer_kwargs": self.optimizer_kwargs,
            "outer_optimizer_kwargs": self.outer_optimizer_kwargs,
            "world_size": self.world_size,
            "diloco_interval": self.diloco_interval,
            "cosine_anneal": self.cosine_anneal,
            "max_local_step": self.max_local_step,
            "eval_iters": self.eval_iters,
            "save_dir": self.save_dir,
            "model_kwargs": self.model_kwargs,
        }

        if self.wandb_project:
            wandb.init(project=self.wandb_project, config=self.wandb_config)
            wandb.config.update(wandb_config)

    def _initialize_distributed(self, rank: int):
        self.rank = rank
        init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            rank=rank,
            world_size=self.world_size,
        )
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device) if self.device.type == "cuda" else None

    def _save_model(self):
        if not self.save_dir:
            return

        name = f"iter_{self.local_step}"
        self._load_master_model()
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.master_model.state_dict(), f"{self.save_dir}/avg_{name}.pth")
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{self.save_dir}/model_{i}_{name}.pth")

    def _outer_step(self):

        self.master_optimizer.zero_grad()

        delta = {name: torch.zeros_like(param.data) for name, param in self.master_model.named_parameters()}
        for local_model in self.models:
            for name, param in local_model.named_parameters():
                delta[name] += param.data - self.master_model.state_dict()[name].data

        for name, param in self.master_model.named_parameters():
            delta[name] /= self.world_size
            param.grad = -delta[name]

        self.master_optimizer.step()

        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())

    def _log_stats(self):
        if not self.wandb_project:
            return

        # cum_grad_norm_var = np.var(self.grad_norms)
        # sliding_grad_norm_var = np.var(self.grad_norms[-100:])
        # cum_loss_var = np.var(self.losses)
        # sliding_loss_var = np.var(self.losses[-100:])
        # param_correlation = parameter_correlation(self.models)
        # euclidean_dist = euclidean_distance(self.models)
        # print(f"Parameter Correlation: {param_correlation:.4f}")
        # print(f"Euclidean Distance: {euclidean_dist:.4f}")

        wandb.log(
            {
                "global_step": self.local_step * self.world_size,
                "local_step": self.local_step,
                "lr": self.optimizers[0].param_groups[0]["lr"],
                "train_loss": random.choice(self.losses[-self.world_size :]),
                "grad_norm": random.choice(self.grad_norms[-self.world_size :]),
                # "cum_grad_norm_var": cum_grad_norm_var,
                # "sliding_grad_norm_var": sliding_grad_norm_var,
                # "cum_loss_var": cum_loss_var,
                # "sliding_loss_var": sliding_loss_var,
                # "p_shuffle": self.p_shuffle,
                # "param_correlation": param_correlation,
                # "euclidean_dist": euclidean_dist,
            }
        )

    def _eval_model(self):
        if not self.eval_dataset:
            return

        self.master_model.eval()
        for model in self.models:
            model.eval()

        correct = 0
        master_losses = []
        with torch.no_grad():
            for i in range(self.eval_iters):
                x, y = self.get_batch(eval=True)

                master_output = self.master_model(x)
                master_loss = self.loss_fn(master_output, y)
                master_losses.append(master_loss.item())
                correct += (master_output.argmax(dim=1) == y).sum().item()

        avg_master_loss = sum(master_losses) / len(master_losses)
        master_loss_std = np.std(master_losses)
        accuracy = correct / (self.eval_iters * self.batch_size)

        print(f"Avg Loss: {avg_master_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        if self.wandb_project:
            wandb.log(
                {
                    "global_step": self.local_step * self.world_size,
                    "local_step": self.local_step,
                    "eval_accuracy": accuracy,
                    "outer_steps": self.local_step // self.diloco_interval,
                    "eval_loss": avg_master_loss,
                    # "eval_loss_std": master_loss_std,
                }
            )

        for model in self.models:
            model.train()

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

        for model, optimizer, scheduler in zip(self.models, self.optimizers, self.schedulers):
            x, y = self.get_batch()
            optimizer.zero_grad()
            output = model(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            optimizer.step()
            if self.cosine_anneal:
                scheduler.step()

            self.losses.append(loss.item())
            self.grad_norms.append(
                torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad != None])).item()
            )

    def _train_loop(self):

        pbar = tqdm(total=self.max_local_step)

        while self.local_step < self.max_local_step:

            self._train_step()

            loss = random.choice(self.losses[-self.world_size :])

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss:.4f}", "Epoch": self.epoch})

            if self.diloco_interval and self.local_step % self.diloco_interval == 0:
                self._outer_step()
                self._eval_model()

                if (
                    self.ckpt_interval
                    and self.local_step % (self.diloco_interval * self.ckpt_interval) == 0
                    and self.local_step > 0
                ):
                    self._save_model()

            if self.local_step % self.log_stats_interval == 0:
                self._log_stats()

            self.local_step += 1

        pbar.close()

    def train(self):
        for model in self.models:
            model.train()

        self._train_loop()
        self._save_model()

        if self.wandb_project:
            wandb.finish()

    def load_model(self, path):
        self.master_model.load_state_dict(torch.load(path))
        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())
