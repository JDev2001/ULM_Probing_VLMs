from src.probes.classifier import build_classifier
from src.probes.trainer import Trainer, RunConfig
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = 3
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dim = 100
    model_head, criterion, optimizer = build_classifier(emb_dim, num_labels, device, lr=1e-3, dropout=0.1)

    config = RunConfig(
        model_name=model_name,
        device=device,
        lr=1e-3,
        dropout=0.1,
        epochs=1,
        log_interval=20,
        mixed_precision=(device.startswith("cuda"))
    )
    embeddings = torch.randn(100, emb_dim)
    labels = torch.randint(0, 3, (100,))

    dataset = TensorDataset(embeddings, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    trainer = Trainer(model_head, criterion, optimizer, config)
    trainer.fit(train_loader, val_loader)
    # Artifacts saved under: f"{model_name}_run/"

if __name__ == "__main__":
    main()
