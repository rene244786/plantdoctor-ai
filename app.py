from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from predict import load_model, predict_image

MODEL_PATH = Path(__file__).parent / "models" / "plantdoctor_mobilenet_v3.pth"
CLASS_GUIDANCE = {
    "healthy": (
        "The image did not strongly match the disease classes this model was trained to recognize.",
        "Keep monitoring the plant and compare new photos if symptoms appear. A healthy result does not rule out other problems.",
    ),
    "tomato_early_blight": (
        "The image pattern is most similar to the model's tomato early blight examples.",
        "Inspect nearby leaves, avoid wetting foliage when watering, and remove badly affected leaves with clean tools. Confirm the cause with a local extension service before treatment.",
    ),
    "bacterial_leaf_spot": (
        "The image pattern is most similar to the model's bacterial leaf spot examples.",
        "Avoid handling wet plants, clean tools between plants, and seek local confirmation before choosing a treatment.",
    ),
    "fungal_leaf_disease": (
        "The model's fungal leaf disease label is broad and does not identify a specific fungus.",
        "Do not choose a treatment from this result alone. Photograph both sides of the leaf and ask a local plant-health specialist to confirm the cause.",
    ),
    "tomato_septoria_leaf_spot": (
        "The image pattern is most similar to the model's tomato Septoria leaf spot examples.",
        "Inspect surrounding leaves, avoid wetting foliage, and remove badly affected leaves with clean tools. Confirm the cause with a local extension service before treatment.",
    ),
}


@st.cache_resource
def get_model(model_path: str, modified_at: float):
    return load_model(Path(model_path))

st.set_page_config(page_title="Plant Doctor", page_icon="🌿", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --paper: #f3f6f0;
        --ink: #20342b;
        --muted: #64746a;
        --green: #337052;
        --line: #d6e0d6;
        --accent: #d77a54;
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--paper);
        background-image: repeating-linear-gradient(135deg, rgba(51,112,82,.025) 0 1px, transparent 1px 12px);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .main .block-container { max-width: 1080px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }
    h1 { font-size: 2.65rem; font-weight: 500; margin-bottom: .15rem; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,.72);
        border: 1px dashed #9ab3a0;
        border-radius: 8px;
    }
    [data-testid="stProgressBar"] > div > div { background-color: var(--green); }
    .eyebrow { color: var(--green); font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .result-panel {
        background: #fff;
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        animation: appear .35s ease-out;
    }
    .result-label { color: var(--muted); font-size: .8rem; text-transform: uppercase; }
    .result-name { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; font-size: 1.8rem; margin-top: .15rem; }
    .score-label { color: var(--ink); font-size: .9rem; }
    .disclaimer { border-top: 1px solid var(--line); color: var(--muted); font-size: .84rem; margin-top: 1.5rem; padding-top: .85rem; }
    @keyframes appear { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 640px) {
        .main .block-container { padding: 1.2rem 1rem 2rem; }
        h1 { font-size: 2.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Leaf image screening</div>', unsafe_allow_html=True)
st.title("Plant Doctor")
st.caption("Upload a leaf photo to see the model’s prediction and class scores.")
uploaded_image = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    if not MODEL_PATH.is_file():
        st.error(f"Model checkpoint not found: {MODEL_PATH}")
        st.code("python train.py --data-dir data/images --epochs 10", language="powershell")
    else:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            with st.spinner("Reading leaf image…"):
                model, classes, image_size, device = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
                result = predict_image(image, model, classes, image_size, device)

            preview_column, result_column = st.columns([1, 1], gap="large")
            with preview_column:
                st.image(image, caption=uploaded_image.name, width="stretch")
            with result_column:
                prediction = str(result["prediction"])
                confidence = float(result["confidence"])
                st.markdown(
                    '<div class="result-panel">'
                    '<div class="result-label">Predicted class</div>'
                    f'<div class="result-name">{html.escape(prediction.replace("_", " ").title())}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Top model score", f"{confidence:.1%}")
                st.markdown("**Scores by class**")
                scores = result["scores"]
                for class_name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
                    label_column, value_column = st.columns([4, 1])
                    label_column.markdown(f'<div class="score-label">{html.escape(class_name.replace("_", " ").title())}</div>', unsafe_allow_html=True)
                    value_column.markdown(f"<div style='text-align:right'>{float(score):.1%}</div>", unsafe_allow_html=True)
                    st.progress(float(score))

            explanation, next_step = CLASS_GUIDANCE.get(
                str(result["prediction"]),
                ("The model returned an unrecognized class label.", "Ask a local plant-health specialist to review the image."),
            )
            st.subheader("What this result means")
            st.write(explanation)
            st.subheader("Suggested next step")
            st.write(next_step)
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be read as an image. Choose a valid JPG or PNG photo.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.markdown(
    '<div class="disclaimer">Screening aid only. The model score is not a calibrated probability or a confirmed diagnosis.</div>',
    unsafe_allow_html=True,
)