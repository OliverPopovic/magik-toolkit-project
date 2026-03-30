import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network_local import NetworkT40Local


def mnist_to_sample_input(x: torch.Tensor) -> torch.Tensor:
    """
    Original sample logic:
    - input starts as [1, 28, 28]
    - pad to [1, 32, 32]
    - repeat channel to [3, 32, 32]
    """
    x_np = x.numpy().astype(np.float32)  # [1, 28, 28]
    temp = np.zeros((1, 32, 32), dtype=np.float32)
    temp[:, 2:30, 2:30] = x_np
    temp = np.repeat(temp, 3, axis=0)    # [3, 32, 32]
    return torch.from_numpy(temp)


class WrappedMNIST(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool):
        self.base = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = mnist_to_sample_input(x)
        return x, y


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="./ckpt_local")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = WrappedMNIST("./MNIST_data", train=True)
    test_ds = WrappedMNIST("./MNIST_data", train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = NetworkT40Local().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        acc = evaluate(model, test_loader, device)
        ckpt = out_dir / f"checkpoint-epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"epoch={epoch} test_acc={acc:.2f}% saved={ckpt}")

    final_ckpt = out_dir / "checkpoint-final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()