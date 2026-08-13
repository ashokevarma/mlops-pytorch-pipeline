"""Train a CIFAR-10 classifier. Every setting comes from the config file."""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_dataloaders  # noqa: E402
from model import get_model  # noqa: E402


def log(**payload) -> None:
    print(json.dumps(payload), flush=True)


def load_config() -> dict:
    candidates = [
        os.environ.get("CONFIG_PATH"),
        "/app/configs/training_config.yaml",
        "configs/training_config.yaml",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            with open(candidate) as handle:
                return yaml.safe_load(handle)
    raise FileNotFoundError(
        "No training config found. Set CONFIG_PATH or provide "
        "configs/training_config.yaml."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_train_batches = int(train_cfg.get("max_train_batches", 0))
    max_val_batches = int(train_cfg.get("max_val_batches", 0))

    log(
        event="training_start",
        device=str(device),
        architecture=model_cfg["architecture"],
        dataset=data_cfg.get("dataset", "cifar10"),
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
    )

    model = get_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=train_cfg["batch_size"],
        num_workers=int(data_cfg.get("num_workers", 2)),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = int(train_cfg["early_stopping_patience"])

    checkpoint_dir = Path(output_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / output_cfg["model_name"]

    for epoch in range(int(train_cfg["epochs"])):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_train_batches
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, max_val_batches
        )

        log(
            epoch=epoch + 1,
            train_loss=round(train_loss, 4),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_accuracy=round(val_acc, 4),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "architecture": model_cfg["architecture"],
                    "num_classes": model_cfg["num_classes"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            log(event="checkpoint_saved", path=str(save_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(event="early_stopping", epoch=epoch + 1)
                break

    log(event="training_complete", best_val_loss=round(best_val_loss, 4))


if __name__ == "__main__":
    main()
