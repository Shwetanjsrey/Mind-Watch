<div align="center">

# MindWatch
### AI-Powered Mental Health Risk Detection Through Social Media Intelligence

*Understanding emotional distress through modern AI, transformer-based NLP, and explainable mental health analytics.*

</div>

---

## The Problem

Mental health struggles often remain invisible until they reach critical stages.

People express stress, hopelessness, emotional exhaustion, and psychological distress online every day—sometimes long before seeking professional help.

Traditional assessment methods rely heavily on clinical observation, interviews, and questionnaires. While valuable, they are often expensive, time-consuming, and not designed for continuous monitoring.

As social media becomes a space for emotional expression, a new question emerges:

> **Can artificial intelligence help identify warning signs of mental distress earlier?**

MindWatch was built to explore that possibility.

---

## What is MindWatch?

MindWatch is an AI-powered mental health risk detection platform that analyzes social media text to identify signs of severe psychological distress, emotional instability, and risk patterns.

Rather than functioning as a black-box classifier, the system combines multiple AI components to provide interpretable and transparent results.

MindWatch does not simply label text.

It explains *why.*

---

## How It Works

MindWatch follows a multi-stage AI analysis pipeline.

### 1. Text Understanding
The user submits a social media post or text sample.

The system preprocesses the content by cleaning, normalizing, and preparing it for transformer-based analysis.

---

### 2. Emotion Intelligence
Using **GoEmotions**, the system identifies fine-grained emotional signals such as:

- Sadness
- Fear
- Disappointment
- Gratitude
- Joy
- Anxiety
- Admiration
- Emotional distress indicators

This helps establish the emotional context behind the text.

---

### 3. Severe Risk Detection
A **DeBERTa-based transformer classifier** evaluates whether the text contains severe mental health risk indicators.

This component was trained specifically for high-risk detection.

---

### 4. Risk Signal Extraction
The system also examines explicit linguistic warning signs through keyword and phrase detection.

Examples include:

- hopeless
- want to die
- ending it
- worthless
- exhausted
- depressed

This creates an additional interpretable risk layer.

---

### 5. Hybrid Risk Fusion
MindWatch combines:

- transformer prediction
- emotional indicators
- explicit risk signals
- confidence estimation

to classify the overall mental health risk into:

# LOW • MEDIUM • HIGH

---

## Architecture

```text
User Input
   ↓
Text Preprocessing
   ↓
Emotion Analysis (GoEmotions)
   ↓
Severe Risk Detection (DeBERTa)
   ↓
Risk Signal Extraction
   ↓
Hybrid Risk Fusion Engine
   ↓
Risk Classification
   ↓
Explainability Layer
   ↓
Interactive Dashboard
```

---

## Why This Matters

AI systems in mental health often suffer from a major problem:

**lack of transparency.**

A prediction without explanation is difficult to trust.

MindWatch addresses this by integrating explainable AI mechanisms that highlight influential signals and provide reasoning alongside classification.

The goal is not just prediction.

The goal is **understanding.**

---

## Key Features

- Severe psychological risk detection
- Fine-grained emotion analysis
- Explainable AI reasoning
- Confidence scoring
- Risk signal highlighting
- LOW / MEDIUM / HIGH classification
- Interactive Flask dashboard
- Modern visualization interface

---

## Performance

The severe-risk detection model achieved:

| Metric | Score |
|--------|-------|
| Accuracy | 90.65% |
| Macro F1 Score | 75.76% |
| Weighted F1 Score | 88.38% |

---

## Technology Stack

MindWatch was built using:

- Python
- Flask
- PyTorch
- Hugging Face Transformers
- DeBERTa
- GoEmotions
- SHAP
- HTML / CSS / JavaScript
- Pandas / NumPy
- Matplotlib

---

## Dashboard Experience

MindWatch provides a modern interactive dashboard featuring:

- emotional analysis visualization
- confidence metrics
- highlighted risk indicators
- risk score breakdown
- explainable predictions
- recommended response guidance

_Add screenshots here_

```md
![Dashboard Preview](screenshots/dashboard.png)
```

---

## Running MindWatch

Clone the repository:

```bash
git clone https://github.com/InvisZero/Mind-Watch.git
cd Mind-Watch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open:

```bash
http://127.0.0.1:5000
```

---

## Future Directions

MindWatch can be extended with:

- multilingual support
- real-time social media monitoring
- user-level emotional trend tracking
- mobile deployment
- cloud hosting
- advanced explainability layers

---

## Final Note

MindWatch is a research-driven AI project exploring how artificial intelligence can support mental health monitoring through interpretable NLP systems.

It is not intended to replace professional diagnosis or clinical care.

But it demonstrates how modern AI can move toward earlier awareness, transparency, and accessible digital support.