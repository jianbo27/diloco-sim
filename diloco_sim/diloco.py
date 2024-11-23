import torch
import torch.distributed as dist
from tqdm import tqdm
from .config import DilocoSimulatorConfig
from .setup import DilocoSetup
from .comm import CommunicationSimulator
from .eval import Evaluator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DilocoSimulator(CommunicationSimulator, Evaluator):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)

    def _average_models(self) -> None:
        for param in self.model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.config.num_nodes

    def _broadcast_model_params(self) -> None:
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def _set_master_grad(self) -> None:
        for name, param in self.model.named_parameters():
            param.grad = self.master_model.state_dict()[name].data.to(param.device) - param.data

    def _synchronize_master_model(self) -> None:
        for name, param in self.master_model.named_parameters():
            param.data = self.model.state_dict()[name].data.to("cpu")

    def _outer_step(self) -> None:
        super()._outer_step()
        self._average_models()

        if self.rank == 0:
            self.master_optimizer.zero_grad()
            self._set_master_grad()
            self.master_optimizer.step()
            self._synchronize_master_model()

        self._broadcast_model_params()

    def _train_step(self):
        super()._train_step()
        x, y = self._get_batch()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.config.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        return loss.item()

    def _train_loop(self):

        pbar = tqdm(total=self.max_local_step)
        self.model.train()

        while self.local_step < self.max_local_step:

            loss = self._train_step()

            if self.local_step % self.config.diloco_interval == 0:
                self._outer_step()
                if self.rank == 0 and self.config.eval_dataset:
                    self._evaluate()

            self.local_step += 1
            pbar.update(1)
            pbar.set_postfix(
                {
                    "train_time": self.train_time + self.communication_time,
                    "loss": f"{loss:.4f}",
                    "epoch": self.epoch,
                }
            )

        pbar.close()

    def _train(self, rank: int):
        try:
            self._setup(rank)
            self._train_loop()

            if self.rank == 0 and self.config.save_dir:
                self._save_checkpoint()
        finally:
            self._cleanup()

    def train(self):
        torch.multiprocessing.spawn(self._train, args=(), nprocs=self.config.num_nodes, join=True)
