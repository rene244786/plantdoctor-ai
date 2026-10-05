import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained plant disease classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--split", default="test", help="Dataset split folder to evaluate")
    parser.add_argument("--model", type=Path, default=Path("models/plantdoctor_mobilenet_v3.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    test_dir = args.data_dir / args.split
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test split directory not found: {test_dir}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    if test_data.classes != classes:
        raise ValueError(
            f"Test classes must match the model classes. Expected {classes}; found {test_data.classes}."
        )
    if len(test_data) == 0:
        raise ValueError(f"No images found in test split: {test_dir}")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    confusion = [[0 for _ in classes] for _ in classes]
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                confusion[actual][predicted] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(classes)))
    print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")
    print("\nPer-class metrics:")
    print("Class\tPrecision\tRecall\tF1\tSupport")
    for index, class_name in enumerate(classes):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(classes))) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{class_name}\t{precision:.4f}\t\t{recall:.4f}\t{f1:.4f}\t{sum(confusion[index])}")

    print("\nConfusion matrix (rows: actual, columns: predicted):")
    print("actual\\pred\t" + "\t".join(classes))
    for class_name, row in zip(classes, confusion):
        print(class_name + "\t" + "\t".join(str(count) for count in row))


if __name__ == "__main__":
    main()
