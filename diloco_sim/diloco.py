import torch
import torch.distributed as dist
from tqdm import tqdm
from .config import DilocoSimulatorConfig
from .setup import SetupSimulator


class DilocoSimulator(SetupSimulator):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)

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

        print(f"Eval Loss: {avg_loss:.4f}")
        print(f"Eval Accuracy: {accuracy:.4f}")

        self.model.train()

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

    def _train_step(self):
        x, y = self._get_batch()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.config.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def _train_loop(self):

        pbar = tqdm(total=self.max_local_step)
        self.model.train()

        while self.local_step < self.max_local_step:

            self._train_step()

            if self.local_step % self.config.diloco_interval == 0:
                self._outer_step()
                if self.rank == 0 and self.config.eval_dataset:
                    self._eval_model()
                dist.barrier()

            self.local_step += 1
            pbar.update(1)

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
