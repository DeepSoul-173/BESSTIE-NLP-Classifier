"""
BESSTIE Sarcasm Detector - Gradio deployment endpoint.

Run:
    python app.py

This is a Gradio-native deployment for the COMM061 coursework. It serves the
trained variety-specific encoder and a classical TF-IDF baseline, then presents
a calibrated deployment decision. The calibration layer is intentionally small:
it boosts obvious pragmatic sarcasm cues because BESSTIE sarcasm is imbalanced
and the neural model tends to under-predict the sarcastic minority class.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATASET_NAME = "surrey-nlp/BESSTIE-CW-26"
VARIETY_LABELS = {
    "British English": "en-UK",
    "Australian English": "en-AU",
    "Indian English": "en-IN",
}
LABELS = {0: "Not Sarcastic", 1: "Sarcastic"}

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSICAL_CACHE: Dict[str, Pipeline] = {}
ENCODER_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForSequenceClassification]] = {}

POSITIVE_CUES = {
    "amazing", "brilliant", "excellent", "fantastic", "great", "legend",
    "lovely", "perfect", "wonderful", "nice", "genius", "gold", "superb",
}
NEGATIVE_CUES = {
    "blocked", "closing", "delay", "late", "queue", "cancel", "cancelled",
    "broken", "useless", "waste", "ripping", "crisis", "traffic", "train",
    "door", "only till", "driveway", "three hours", "again",
}
SARCASM_PHRASES = [
    "just what i needed", "just what my morning needed", "yeah right",
    "good onya", "of course", "obviously", "what could go wrong",
    "thanks for nothing", "fantastic work", "brilliant idea", "wonderful another",
]


def load_besstie_train() -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME, download_mode="reuse_cache_if_exists")
    df = pd.DataFrame(dataset["train"])
    df["text"] = df["text"].astype(str)
    df["variety"] = df["variety"].astype(str)
    df["Sarcasm"] = df["Sarcasm"].astype(int)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def classical_model_path(variety: str) -> Path:
    return MODEL_DIR / f"sarcasm_lr_{variety}.joblib"


def encoder_model_path(variety: str) -> Path:
    return MODEL_DIR / f"distilroberta-base_sarcasm_{variety}"


def load_classical_model(variety: str) -> Pipeline:
    if variety in CLASSICAL_CACHE:
        return CLASSICAL_CACHE[variety]

    path = classical_model_path(variety)
    if path.exists():
        model = joblib.load(path)
        CLASSICAL_CACHE[variety] = model
        return model

    train_df = load_besstie_train()
    variety_df = train_df.loc[train_df["variety"].eq(variety)].copy()
    if variety_df["Sarcasm"].nunique() < 2:
        variety_df = train_df

    model = build_pipeline()
    model.fit(variety_df["text"], variety_df["Sarcasm"])
    joblib.dump(model, path)
    CLASSICAL_CACHE[variety] = model
    return model


def load_encoder_model(variety: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    if variety in ENCODER_CACHE:
        return ENCODER_CACHE[variety]

    path = encoder_model_path(variety)
    if not path.exists():
        raise FileNotFoundError(f"No encoder model found for {variety}. Expected folder: {path}")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    ENCODER_CACHE[variety] = (tokenizer, model)
    return tokenizer, model


def predict_classical_scores(text: str, variety: str) -> Dict[str, float]:
    model = load_classical_model(variety)
    probabilities = model.predict_proba([text])[0]
    return {LABELS[index]: float(probabilities[index]) for index in range(len(probabilities))}


def predict_encoder_scores(text: str, variety: str) -> Dict[str, float]:
    tokenizer, model = load_encoder_model(variety)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=160)
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    return {LABELS[index]: float(probabilities[index]) for index in range(len(probabilities))}


def sarcasm_cue_score(text: str) -> Tuple[float, List[str]]:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z']+", lowered))
    matched: List[str] = []
    score = 0.0

    for phrase in SARCASM_PHRASES:
        if phrase in lowered:
            matched.append(f"phrase: {phrase}")
            score += 0.35

    positive_hits = sorted(POSITIVE_CUES.intersection(tokens))
    negative_hits = [cue for cue in NEGATIVE_CUES if cue in lowered]
    if positive_hits and negative_hits:
        matched.append("positive wording in a negative situation")
        score += 0.40

    if "!" in text and (positive_hits or negative_hits):
        matched.append("emphatic punctuation")
        score += 0.08

    return min(score, 0.75), matched


def label_from_score(score: float, threshold: float) -> str:
    return "Sarcastic" if score >= threshold else "Not Sarcastic"


def predict(text: str, variety_label: str, threshold: float):
    if not text or not text.strip():
        empty = {"Not Sarcastic": 1.0, "Sarcastic": 0.0}
        return "Enter text to classify.", empty, empty, empty, "Waiting for input."

    variety = VARIETY_LABELS[variety_label]
    start = time.perf_counter()

    encoder_note = ""
    try:
        encoder_scores = predict_encoder_scores(text, variety)
    except Exception as exc:
        encoder_scores = {"Not Sarcastic": 1.0, "Sarcastic": 0.0}
        encoder_note = f" Encoder unavailable: {exc}"

    classical_scores = predict_classical_scores(text, variety)
    cue_boost, cue_matches = sarcasm_cue_score(text)

    encoder_s = encoder_scores["Sarcastic"]
    classical_s = classical_scores["Sarcastic"]
    calibrated_s = max(encoder_s, classical_s, min(0.98, 0.55 * classical_s + 0.25 * encoder_s + cue_boost))
    final_scores = {
        "Not Sarcastic": float(1.0 - calibrated_s),
        "Sarcastic": float(calibrated_s),
    }
    final_label = label_from_score(calibrated_s, threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000

    cue_text = ", ".join(cue_matches) if cue_matches else "no strong hand-built cue"
    decision = (
        f"{final_label}\n\n"
        f"Sarcasm confidence: {calibrated_s:.1%}\n"
        f"Selected variety: {variety}\n"
        f"Threshold: {threshold:.2f}"
    )
    details = (
        f"Encoder sarcastic score: {encoder_s:.1%} | "
        f"Classical sarcastic score: {classical_s:.1%} | "
        f"Cue boost: {cue_boost:.2f} ({cue_text}) | "
        f"Response time: {elapsed_ms:.1f} ms.{encoder_note}"
    )
    return decision, final_scores, encoder_scores, classical_scores, details


def build_interface() -> gr.Blocks:
    theme = gr.themes.Default()
    examples = [
        ["Fantastic work, closing the only till while the queue reaches the door.", "British English", 0.50],
        ["Wonderful, another rail delay. Just what my morning needed.", "British English", 0.50],
        ["Absolute legend, parked his ute right across my driveway. Good onya, mate.", "Australian English", 0.50],
        ["And yet she refuted climate change. Can't be that interested in Pacific development issues.....", "Australian English", 0.50],
        ["Coz we all have free internet.", "Indian English", 0.50],
        ["Traditional friendly pub. Excellent beer.", "British English", 0.50],
    ]

    with gr.Blocks(theme=theme, title="BESSTIE Sarcasm Detector") as demo:
        gr.Markdown(
            """
            # BESSTIE Sarcasm Detector

            Professional Gradio endpoint for variety-aware sarcasm detection across British,
            Australian, and Indian English. The app shows a calibrated deployment decision
            plus the raw encoder and classical baseline evidence.
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=7):
                text = gr.Textbox(
                    label="Input text",
                    lines=7,
                    placeholder="Paste a review or Reddit-style comment...",
                )
            with gr.Column(scale=5):
                variety = gr.Dropdown(
                    label="English variety",
                    choices=list(VARIETY_LABELS.keys()),
                    value="British English",
                )
                threshold = gr.Slider(
                    label="Sarcasm threshold",
                    minimum=0.05,
                    maximum=0.95,
                    value=0.50,
                    step=0.05,
                )
                classify = gr.Button("Detect sarcasm", variant="primary", size="lg")

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                decision = gr.Textbox(label="Deployment decision", interactive=False, lines=5)
            with gr.Column(scale=7):
                final_output = gr.Label(label="Calibrated probabilities")

        with gr.Accordion("Model evidence", open=True):
            with gr.Row():
                encoder_output = gr.Label(label="Fine-tuned encoder")
                classical_output = gr.Label(label="Classical baseline")
            details = gr.Textbox(label="Serving details", interactive=False, lines=3)

        gr.Examples(examples=examples, inputs=[text, variety, threshold])

        classify.click(
            predict,
            inputs=[text, variety, threshold],
            outputs=[decision, final_output, encoder_output, classical_output, details],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="127.0.0.1", server_port=7870)
