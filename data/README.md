# Image data

Place labeled images in this structure. Each class folder name becomes a prediction label.

Recommended starting point for fine-tuning the MobileNetV3-Small classifier:

- Minimum viable: 20 train images + 5 validation images per class
- Better target: 40-60 train images + 10-15 validation images per class
- For classes below 30 total images, do not trust the retrain result without collecting more real samples first

Use the `source/` intake folder to gather real photos before sorting them into `train/` and `val/`.

```text
data/images/
  train/
    healthy/
    tomato_early_blight/
    bacterial_leaf_spot/
    fungal_leaf_disease/
  val/
    healthy/
    tomato_early_blight/
    bacterial_leaf_spot/
    fungal_leaf_disease/
  test/
    healthy/
    tomato_early_blight/
    bacterial_leaf_spot/
    fungal_leaf_disease/
```

Use the same class folders in each split. Keep photos from the same plant or capture session in only one split to reduce data leakage. Start with a balanced dataset and inspect labels before training. This project does not include image data; use a dataset whose terms permit your intended use and retain its attribution/license information.
