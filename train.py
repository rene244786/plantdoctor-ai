import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a plant leaf disease classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output", type=Path, default=Path("models/plantdoctor_mobilenet_v3.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--scratch", action="store_true", help="Do not load pretrained ImageNet weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    weights = None if args.scratch else MobileNet_V3_Small_Weights.DEFAULT
    image_size = 224
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected train and val folders under {args.data_dir}")

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=eval_transform)
    if len(train_data.classes) < 2:
        raise ValueError("Training data must contain at least two class folders")
    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError("Train and val folders must contain identical class names")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(train_data.classes))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_accuracy = -1.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(dim=1) == labels).sum().item()

        val_accuracy = correct / len(val_data)
        mean_train_loss = train_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {mean_train_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": train_data.classes,
                    "image_size": image_size,
                    "validation_accuracy": best_accuracy,
                },
                args.output,
            )

    print(f"Best validation accuracy: {best_accuracy:.4f}; model saved to {args.output}")


if __name__ == "__main__":
    main()