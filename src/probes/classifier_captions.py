
import torch
from torch import nn

class ClassifierCaptions(nn.Module):
    """Simple linear classifier head."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_classifier(emb_dim: int, num_labels: int, device: str = "cpu" if not torch.cuda.is_available() else "cuda",
                     lr: float = 1e-3, weight_decay: float = 0.0, dropout: float = 0.1):
    """Factory that returns (model, criterion, optimizer)."""
    model = ClassifierCaptions(emb_dim, num_labels, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer
