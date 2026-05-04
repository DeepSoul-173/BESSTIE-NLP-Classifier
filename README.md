# BESSTIE-NLP-Classifier 🚀

A multi-variety English sentiment and sarcasm classifier built on the **BESSTIE** (A BEnchmark for Sentiment and Sarcasm classification for varieTIes of English) benchmark. 

This project addresses linguistic bias in sentiment analysis by evaluating and classifying text across three varieties of English: **Australian**, **British**, and **Indian**.

## ✨ Features
- **Multi-Variety Support**: Fine-tuned classification for diverse English dialects.
- **RoBERTa-based Classification**: Leverages state-of-the-art transformer models for high accuracy.
- **Classical Baselines**: Includes Logistic Regression baselines for performance comparison.
- **Interactive UI**: Built with Gradio for real-time sentiment and sarcasm prediction.

## 🛠️ Tech Stack
- **Languages**: Python
- **Frameworks**: Transformers (Hugging Face), PyTorch, Gradio
- **Models**: RoBERTa, Logistic Regression
- **Dataset**: BESSTIE (from [Hugging Face](https://huggingface.co/datasets/unswnlporg/BESSTIE))

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support for faster training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DeepSoul-173/BESSTIE-NLP-Classifier.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
To launch the Gradio interface:
```bash
python app.py
```

## 📊 Dataset Reference
This project utilizes the BESSTIE dataset, published in the *Findings of the Association for Computational Linguistics: ACL 2025*.

---
Developed by [DeepSoul-173](https://github.com/DeepSoul-173)
