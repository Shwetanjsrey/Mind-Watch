<div align="center">

# 🧠 MindWatch

### *AI-Powered Mental Health Risk Detection Through Social Media Intelligence*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Transformers-NLP-blueviolet?style=for-the-badge" />
</p>

---

### Detecting psychological distress with transformer-based AI, emotional intelligence, and explainable analytics.

</div>

---

# The Problem

Mental health struggles often remain unnoticed until they become severe.

People frequently express emotional distress, hopelessness, burnout, anxiety, and psychological pain online—sometimes long before seeking professional help.

Traditional assessment methods depend heavily on interviews, questionnaires, and clinical observation. While valuable, they are expensive, time-consuming, and not suited for continuous large-scale monitoring.

This raises an important question:

> ### Can artificial intelligence help identify warning signs of mental distress earlier?

That question led to the creation of **MindWatch**.

---

# What is MindWatch?

MindWatch is an AI-powered web platform designed to analyze social media text and detect signs of psychological distress, emotional instability, and severe mental health risk.

Rather than acting as a black-box classifier, MindWatch combines multiple AI components to provide interpretable, explainable, and actionable analysis.

It does not simply produce a prediction.

It helps explain **why** that prediction was made.

---

# Product Preview

## Landing Experience

<p align="center">
  <img src="screenshots/home.png" width="1000"/>
</p>

---

## AI Risk Analysis Dashboard

<p align="center">
  <img src="screenshots/analysis.png" width="1000"/>
</p>

---

## Detailed Risk Assessment Output

<p align="center">
  <img src="screenshots/result.png" width="1000"/>
</p>

---

# How MindWatch Works

MindWatch follows a hybrid AI analysis pipeline.

---

## 1. Text Understanding

The user submits a social media post or text sample.

The system preprocesses the content through:

- text cleaning
- normalization
- token preparation
- input formatting

This prepares the text for transformer-based analysis.

---

## 2. Emotion Intelligence

MindWatch uses **GoEmotions**, a transformer-based emotion classification model, to detect fine-grained emotional states.

Examples include:

- sadness
- disappointment
- fear
- confusion
- gratitude
- joy
- admiration
- emotional distress indicators

This helps understand the emotional context behind the message.

---

## 3. Severe Risk Detection

A **DeBERTa-based transformer model** evaluates whether the input contains severe psychological risk indicators.

This model serves as the primary AI risk detector.

---

## 4. Risk Signal Extraction

MindWatch also detects explicit linguistic warning signs using phrase and keyword analysis.

Examples:

- hopeless
- want to die
- ending it
- worthless
- depressed
- exhausted
- no point

This creates an interpretable risk evidence layer.

---

## 5. Hybrid Risk Fusion

The final assessment combines:

- transformer prediction
- emotional indicators
- explicit risk signals
- confidence estimation

to classify the overall risk level.

<div align="center">

# LOW • MEDIUM • HIGH

</div>

---

# System Architecture

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
Interactive Dashboard Output
```

---

# Why This Matters

Mental health AI systems often suffer from a major weakness:

## Lack of transparency

A prediction without explanation is difficult to trust.

MindWatch addresses this through explainable AI techniques, confidence estimation, emotional reasoning, and highlighted risk indicators.

The goal is not only classification.

The goal is **understanding**.

---

# Core Features

✨ Severe Mental Health Risk Detection  
🎭 Fine-Grained Emotion Analysis  
🧠 Explainable AI Reasoning  
📊 LOW / MEDIUM / HIGH Classification  
🎯 Confidence Score Estimation  
🚨 Risk Signal Detection  
📝 Highlighted Risk Text Analysis  
💡 Actionable Guidance Output  
🌐 Interactive Modern Flask Dashboard  

---

# Technology Stack

| Layer | Technology |
|------|------------|
| Backend | Flask |
| Language | Python |
| Deep Learning | PyTorch |
| NLP Framework | Hugging Face Transformers |
| Risk Detection | DeBERTa |
| Emotion Analysis | GoEmotions |
| Explainability | SHAP |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |

---

# Performance

The severe-risk detection model achieved:

| Metric | Score |
|--------|-------|
| Accuracy | **90.65%** |
| Macro F1 Score | **75.76%** |
| Weighted F1 Score | **88.38%** |

---

# Running MindWatch

Clone repository:

```bash
git clone https://github.com/InvisZero/Mind-Watch.git
cd Mind-Watch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

Open:

```bash
http://127.0.0.1:5000
```

---

# Project Structure

```bash
Mind-Watch/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── risk_engine.py
│   ├── emotion_analyzer.py
│   ├── explain_prediction.py
│   └── ...
│
├── templates/
│   ├── index.html
│   └── about.html
│
├── static/
│   └── style.css
│
├── data/
│
└── screenshots/
```

---

# Future Improvements

MindWatch can be extended with:

- multilingual support
- real-time social media monitoring
- user-level emotional trend tracking
- mobile deployment
- cloud hosting
- enhanced explainability modules

---

# Disclaimer

MindWatch is a research-focused AI application.

It is not a medical diagnostic tool and should not replace professional mental health consultation or emergency intervention.