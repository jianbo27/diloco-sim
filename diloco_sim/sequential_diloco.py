from dataclasses import dataclass, field
from typing import Optional, Callable, Type, Iterator, Tuple
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy
import os
from tqdm import tqdm


@dataclass
class SequentialDilocoConfig:
    model_cls: Type[torch.nn.Module]
    model_kwargs: dict
    loss_fn: Callable[..., torch.Tensor]
    train_dataset: torch.utils.data.Dataset
    optimizer_kwargs: dict
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    batch_size: int = 16
    eval_dataset: Optional[torch.utils.data.Dataset] = None
    ckpt_interval: Optional[int] = None
    eval_iters: int = 50
    save_dir: Optional[str] = None
    num_epochs: int = 1
    cosine_anneal: bool = False
    model_path: Optional[str] = None
    num_nodes: int = 4
    diloco_interval: int = 500
    outer_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD
    outer_optimizer_kwargs: dict = field(
        default_factory=lambda: {"lr": 0.7, "nesterov": True, "momentum": 0.9}
    )


class SetupSequentialSimulator:
    device: torch.device
    models: list[torch.nn.Module]
    optimizers: list[torch.optim.Optimizer]
    schedulers: list[Optional[torch.optim.lr_scheduler.CosineAnnealingLR]]
    master_model: torch.nn.Module
    master_optimizer: torch.optim.Optimizer
    train_dataloader: DataLoader
    eval_dataloader: Optional[DataLoader] = None
    train_data_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]
    eval_data_iter: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None
    max_local_step: int
    local_step: int = 0
    epoch: int = 0

    def __init__(self, config: SequentialDilocoConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_local_step = (
            self.config.num_epochs * len(self.config.train_dataset)
            // (self.config.batch_size * self.config.num_nodes)
        )

    def _setup_models(self):
        # Setup worker models
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        for _ in range(self.config.num_nodes):
            model = self.config.model_cls(**self.config.model_kwargs).to(self.device)
            optimizer = self.config.optimizer_cls(
                model.parameters(), **self.config.optimizer_kwargs
            )
            scheduler = (
                CosineAnnealingLR(optimizer, T_max=self.max_local_step)
                if self.config.cosine_anneal
                else None
            )
            
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

    def _setup_master_model(self):
        self.master_model = deepcopy(self.models[0]).cpu()
        for param in self.master_model.parameters():
            param.requires_grad = True

    def _setup_master_optimizer(self):
        self.master_optimizer = self.config.outer_optimizer_cls(
            self.master_model.parameters(), **self.config.outer_optimizer_kwargs
        )

    def _setup_train_dataloader(self):
        self.train_dataloader = DataLoader(
            self.config.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.train_data_iter = iter(self.train_dataloader)

    def _setup_eval_dataloader(self):
        self.eval_dataloader = DataLoader(
            self.config.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.eval_data_iter = iter(self.eval_dataloader)

    def _save_checkpoint(self):
        # Save the first model's state as they should all be identical after averaging
        torch.save(
            self.models[0].state_dict(),
            os.path.join(self.config.save_dir, f"model_{self.epoch}.pt"),
        )

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

    def _setup(self):
        self._setup_models()
        self._setup_train_dataloader()
        self._setup_master_model()
        self._setup_master_optimizer()
        if self.config.eval_dataset:
            self._setup_eval_dataloader()


class SequentialDilocoSimulator(SetupSequentialSimulator):
    def __init__(self, config: SequentialDilocoConfig) -> None:
        super().__init__(config)

    def _eval_model(self):
        self.models[0].eval()  # Evaluate the first model after averaging

        correct = 0
        losses = []
        with torch.no_grad():
            for _ in range(self.config.eval_iters):
                x, y = self._get_batch(eval=True)
                output = self.models[0](x)
                loss = self.config.loss_fn(output, y)
                losses.append(loss.item())
                correct += (output.argmax(dim=1) == y).sum().item()

        avg_loss = sum(losses) / len(losses)
        accuracy = correct / (self.config.eval_iters * self.config.batch_size)

        print(f"Eval Loss: {avg_loss:.4f}")
        print(f"Eval Accuracy: {accuracy:.4f}")

        self.models[0].train()

    def _average_parameters(self):
        # Average parameters across all models
        with torch.no_grad():
            for name, _ in self.models[0].named_parameters():
                params = torch.stack([
                    model.state_dict()[name].data
                    for model in self.models
                ])
                avg_param = params.mean(dim=0)
                
                # Update all models with averaged parameters
                for model in self.models:
                    model.state_dict()[name].data.copy_(avg_param)

    def _outer_step(self):
        # Average parameters
        self._average_parameters()
        
        # Update master model
        for name, param in self.master_model.named_parameters():
            param.data.copy_(self.models[0].state_dict()[name].data.cpu())

        # Compute outer optimization step
        self.master_optimizer.zero_grad()
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - self.models[0].state_dict()[name].data.cpu()
        self.master_optimizer.step()

        # Update all models with master parameters
        for model in self.models:
            for name, param in model.named_parameters():
                param.data.copy_(
                    self.master_model.state_dict()[name].data.to(self.device)
                )

    def _train_step(self):
        x, y = self._get_batch()
        
        # Update each model sequentially
        for model, optimizer, scheduler in zip(
            self.models, self.optimizers, self.schedulers
        ):
            optimizer.zero_grad()
            output = model(x)
            loss = self.config.loss_fn(output, y)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

    def _train_loop(self):
        pbar = tqdm(total=self.max_local_step)

        for model in self.models:
            model.train()

        while self.local_step < self.max_local_step:
            self._train_step()

            if self.local_step % self.config.diloco_interval == 0:
                self._outer_step()
                if self.config.eval_dataset:
                    self._eval_model()

            self.local_step += 1
            pbar.update(1)

        pbar.close()

    def train(self):
        self._setup()
        self._train_loop()
        
        if self.config.save_dir:
            self._save_checkpoint()