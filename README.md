# PlantDoctorAI

Plant leaf image-classification starter using a MobileNetV3-Small model.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Prepare labeled train, validation, and test images as described in [data/README.md](data/README.md). The trainer uses `train` and `val`; reserve `test` for final evaluation and do not tune against it.

## Train

```powershell
python train.py --data-dir data/images --epochs 10
```

By default, the model starts from pretrained ImageNet weights, which may be downloaded the first time. Use `--scratch` to train without them. The best validation-accuracy checkpoint is written to `models/plantdoctor_mobilenet_v3.pth`.

## Evaluate

After training, evaluate once on the held-out test split:

```powershell
python evaluate.py --data-dir data/images --split test
```

The evaluator reports overall accuracy, per-class precision/recall/F1, and a confusion matrix. Do not use the validation split as final test evidence.

## Predict

```powershell
python predict.py path\to\leaf.jpg
```

## Web app
## Deploy online

The app can be hosted on Streamlit Community Cloud:

1. Push this project to a GitHub repository. Include `models/plantdoctor_mobilenet_v3.pth`; it is required for predictions. Training images remain excluded from Git.
2. In Streamlit Community Cloud, choose **Create app**, select the repository and branch, and set the main file to `app.py`.
3. Deploy. The service installs dependencies from `requirements.txt` and provides a public app URL.

The checkpoint was trained using PlantVillage-derived data. Review the attribution and CC BY-SA terms in [the dataset source notes](data/source/plantvillage_real/README.md) before redistributing it. The app is a screening demo until it has been evaluated on a balanced, independent test set.

## Web app

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, upload a JPG or PNG leaf photo, and review the predicted class and model scores.

Prediction quality depends on representative, correctly labeled images. Treat output as an initial screening result, not a definitive plant-health diagnosis.