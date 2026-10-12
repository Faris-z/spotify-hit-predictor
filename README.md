# Spotify Hit Predictor + Song DNA Analyzer

Predicting whether a Spotify track is a hit from its audio features, explaining the model with SHAP, and grouping songs into audio archetypes with K-Means.

## Problem

Two questions on the Kaggle Spotify Tracks dataset:
1. **Supervised:** can a track's audio features and genre predict whether it lands in the top quarter of popularity?
2. **Unsupervised:** do songs fall into distinct audio "archetypes", and do hit rates differ between them?

## Approach

All steps are in `spotify_hit_predictor.ipynb` (random seed 42).

**Data**
- [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset), downloaded with `kagglehub`
- Rows with missing values and duplicate `track_id`s removed, leaving **89,740 unique tracks**
- Target: `is_hit = popularity >= 49`, the 75th percentile of popularity (about 25% positives)

**Features**
- 6 engineered features: dance-energy ratio, mood score, acoustic signature, tempo bucket, vocal presence, duration in minutes
- Genre label-encoded, `explicit` cast to 0/1, text columns (track, artist, album) dropped. 20 input features in total.

**Modeling**
- Stratified 80/20 train/test split, standard scaling fit on the training set only
- SMOTE applied to the training set only; the test set keeps the real class balance
- 5 models compared with Stratified 5-fold CV on the SMOTE-resampled training set: Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost
- XGBoost tuned with `RandomizedSearchCV` (40 candidates, 3-fold stratified CV, ROC-AUC scoring)
- A PyTorch MLP (256-128-64 with BatchNorm and Dropout, cosine-annealed learning rate) trained for comparison
- SHAP `TreeExplainer` on the tuned XGBoost model

**Clustering**
- K-Means on 8 standardized audio features (danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo)
- K = 5 chosen with the elbow method; clusters visualized with PCA and t-SNE (t-SNE on an 8,000-track sample)

## Results

**Classification, evaluation setup:** held-out stratified test set of **17,948 tracks** (4,504 hits), never resampled or used for training or tuning. Metrics are for the hit class.

| Model | Test ROC-AUC | Test F1 |
|---|---|---|
| XGBoost (tuned) | 0.8375 | 0.5933 |

Cross-validation scores are not reported, because the CV folds were drawn from SMOTE-resampled data and overstate performance.

**SHAP:** the features with the highest mean |SHAP value| are `track_genre`, `key` and `tempo`.

**Clustering:** hit rates are similar across most archetypes:

| Archetype (notebook label) | Audio profile | Hit rate |
|---|---|---|
| A | high danceability, energy and valence | 26.8% |
| C | high energy, low acousticness | 25.9% |
| B | acoustic and instrumental-heavy | 25.0% |
| D | high liveness | 17.3% |
| E | very high speechiness (1,059 tracks) | 4.0% |

The danceable, high-energy archetype leads only narrowly, and the instrumental-heavy archetype is close to the overall 25% hit rate. Speech-heavy tracks stand out as rarely being hits.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, imbalanced-learn (SMOTE), PyTorch, SHAP, matplotlib, seaborn, kagglehub, Jupyter.

## How to Run

**Google Colab:** upload `spotify_hit_predictor.ipynb` to [Colab](https://colab.research.google.com) and run all cells. The dataset downloads automatically through `kagglehub`.

**Locally** (Python 3.10+):

```bash
git clone https://github.com/Faris-z/spotify-hit-predictor.git
cd spotify-hit-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook spotify_hit_predictor.ipynb
```

## Limitations / Next Steps

- **CV on resampled data:** SMOTE runs before cross-validation, so synthetic samples leak across folds. Next: apply SMOTE inside each fold with an `imblearn` pipeline and report CV scores.
- **Genre dominates:** `track_genre` is label-encoded and is the top SHAP feature, so the model leans on genre more than on audio. Next: one-hot or target encoding, and a variant without genre.
- **Moderate hit-class F1 (0.59):** the model misses a large share of hits at the default 0.5 threshold. Next: tune the decision threshold and compare against a calibrated baseline.
- **Popularity is relative:** the hit label is a percentile cut on one snapshot of Spotify popularity and does not account for release date.
- **Archetype names** are placeholders (A–E); the profiles above are read from the cluster means.
- The final "summary" cell prints hardcoded findings that the notebook does not compute. Next: replace them with computed values.

## License

MIT, see [LICENSE](LICENSE).
