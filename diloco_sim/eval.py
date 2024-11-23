import torch
from .config import DilocoSimulatorConfig
from .setup import DilocoSetup
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)  # Handler's level
logger.addHandler(handler)


class Evaluator(DilocoSetup):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)

    def _evaluate(self):
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

        logger.info(f"Eval Loss: {avg_loss:.4f}, Eval Accuracy: {accuracy:.4f}")
        print(f"Eval Loss: {avg_loss:.4f}, Eval Accuracy: {accuracy:.4f}")

        self.model.train()
