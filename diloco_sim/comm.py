from .setup import DilocoSetup
from .config import DilocoSimulatorConfig


class CommunicationSimulator(DilocoSetup):
    train_time: float = 0.0
    communication_time: float = 0.0

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)

    def _train_step(self):
        self.train_time += self.config.batch_size / self.config.flop_per_second_per_node  # TODO (not right)

    def _outer_step(self):
        model_size = sum(p.numel() for p in self.model.parameters())  # TODO definitely not right
        self.communication_time += self.config.network_delay
        self.communication_time += model_size / self.config.network_bandwidth
