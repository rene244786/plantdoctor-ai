import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a class for a plant leaf image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/plantdoctor_mobilenet_v3.pth"))
    return parser.parse_args()


def load_model(model_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, classes, checkpoint["image_size"], device


def predict_image(image: Image.Image, model: nn.Module, classes: list[str], image_size: int, device: torch.device) -> dict[str, object]:
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(preprocess(image).unsqueeze(0).to(device)), dim=1)[0]

    confidence, index = probabilities.max(dim=0)
    return {
        "prediction": classes[index.item()],
        "confidence": confidence.item(),
        "scores": {class_name: score.item() for class_name, score in zip(classes, probabilities)},
    }


def main() -> None:
    args = parse_args()
    model, classes, image_size, device = load_model(args.model)
    image = Image.open(args.image).convert("RGB")
    result = predict_image(image, model, classes, image_size, device)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("Note: predictions are a screening aid, not a confirmed diagnosis.")


if __name__ == "__main__":
    main()