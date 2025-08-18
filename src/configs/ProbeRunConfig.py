from dataclasses import dataclass


@dataclass
class RunConfig:
    model_name: str
    device: str = "cpu"
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.1
    epochs: int = 3
    grad_accum_steps: int = 1
    log_interval: int = 50
    mixed_precision: bool = False
