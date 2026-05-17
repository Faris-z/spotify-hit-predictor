# 🎵 Spotify Hit Predictor + Song DNA Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **What makes a song a hit? Can we predict it — and reverse-engineer the formula?**  
> This project combines supervised ML with unsupervised discovery to decode the DNA of popular music.

---

## 📌 Problem Statement

The music industry invests billions in producing and promoting songs, yet most tracks never become popular. This project analyzes **114,000+ Spotify tracks** to predict whether a song will be a hit based on its audio features, and uses clustering to discover hidden **song archetypes** — groups of songs that share a similar audio "personality."

---

## 🧠 What This Project Covers

| Area | Techniques Used |
|---|---|
| **EDA** | Popularity distributions, audio feature analysis, genre comparisons, correlation deep-dive |
| **Feature Engineering** | 6 engineered features (dance-energy ratio, mood score, acoustic signature, tempo bucket, vocal presence, duration) |
| **Imbalance Handling** | SMOTE oversampling (no data leakage) |
| **Model Benchmarking** | 7 models with Stratified 5-Fold cross-validation |
| **Hyperparameter Tuning** | RandomizedSearchCV — 40 combinations, ROC-AUC scoring |
| **Deep Learning** | PyTorch DNN with BatchNorm, Dropout, CosineAnnealingLR |
| **Unsupervised Learning** | K-Means clustering, Elbow method, PCA + t-SNE visualization |
| **Explainability** | SHAP — global summary + individual force plots |
| **Insights** | Radar charts comparing hit DNA vs flop DNA, hit rate per archetype |

---

## 📊 Key Findings

**From SHAP analysis:**
1. **Danceability** and **energy** are the top predictors of a hit
2. High **acousticness** and **instrumentalness** strongly predict a flop
3. Songs with moderate tempo (100–130 BPM) have the highest hit rate
4. The "party banger" archetype produces the most hits
5. Purely instrumental tracks almost never become popular

**From clustering:**
- 5 distinct song archetypes discovered via K-Means
- Each archetype has a unique audio "personality" (visualized via radar charts)
- Hit rates vary dramatically across archetypes

---

## 📁 Project Structure

```
spotify-hit-predictor/
│
├── spotify_hit_predictor.ipynb   # Main notebook (all sections)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── LICENSE                       # MIT License
```

---

## 🚀 Getting Started

### Option A — Google Colab (recommended)

1. Upload `spotify_hit_predictor.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run all cells — dataset downloads automatically via `kagglehub`

### Option B — Run locally

```bash
# Clone
git clone https://github.com/Faris-z/spotify-hit-predictor.git
cd spotify-hit-predictor

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install & run
pip install -r requirements.txt
jupyter notebook spotify_hit_predictor.ipynb
```

**Dataset:** [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

---

## 🛠️ Tech Stack

- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **scikit-learn** — preprocessing, classical ML, clustering, PCA, t-SNE
- **XGBoost / CatBoost** — gradient boosting
- **imbalanced-learn** — SMOTE
- **PyTorch** — deep neural network
- **SHAP** — model explainability

---

## 📓 Notebook Sections

| # | Section | Description |
|---|---|---|
| 0 | Setup | Install & import libraries |
| 1 | Load data | Download from Kaggle, clean |
| 2 | Deep EDA | Distributions, correlations, genre analysis |
| 3 | Feature engineering | 6 new features + encoding |
| 4 | Train/test split | Stratified split + SMOTE |
| 5 | Model benchmark | 7 models, 5-Fold CV |
| 6 | Hyperparameter tuning | RandomizedSearchCV |
| 7 | PyTorch DNN | HitNet with BatchNorm + Dropout |
| 8 | Test evaluation | ROC curves, confusion matrix |
| 9 | SHAP | Global + individual explanations |
| 10 | Song DNA clustering | K-Means, PCA, t-SNE, radar charts |
| 11 | Hit recipe | Final insights dashboard |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Faris Alzahrani**  
[GitHub](https://github.com/Faris-z)
