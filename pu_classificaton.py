"""
Classifying Hindutva-Affiliated NGOs in India Using Positive-Unlabeled Learning
================================================================================

This script implements a positive-unlabeled (PU) learning pipeline to identify
Hindutva-affiliated NGOs in the NGO Darpan registry (~500k organizations).

Steps
--------
1. Fuzzy string matching against a curated seed list to generate initial labels
2. Manual verification of close matches to produce a reliable positive set
3. TF-IDF encoding of organizational keyword fields + categorical features
4. LightGBM trained under the PU assumption (labeled pos. vs. unlabeled)
5. Elkan & Noto (2008) calibration to recover true posterior probabilities
6. Ensemble averaging and unanimous-agreement filtering for high-precision output

References
----------
- Elkan, C. & Noto, K. (2008). Learning classifiers from only positive and
  unlabeled data. KDD '08.
- Bekker, J. & Davis, J. (2020). Learning from positive and unlabeled data:
  A survey. Machine Learning, 109.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, precision_recall_curve
from thefuzz import process

# ---------------------------------------------------------------------------
# Define Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 1996
HOLDOUT_FRACTION = 0.10
HIGH_CONFIDENCE_THRESHOLD = 0.85
ENSEMBLE_THRESHOLD = 0.80
AGREEMENT_THRESHOLD = 0.60
N_ENSEMBLE_MODELS = 5
ENSEMBLE_SEEDS = [1996, 2024, 42, 123, 999]

STATES_AND_UTS = [
    "Andaman and Nicobar Islands", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh",
    "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal"
]

# List of known Hindutva affiliates
HINDUTVA_ORGS = [
    "Rashtriya Swayamsevak Sangh", "Bharatiya Janata Party",
    "Bharatiya Jana Sangh", "Vishva Hindu Parishad", "Bajrang Dal",
    "Durga Vahini", "Hindu Yuva Vahini", "Hindu Jagran Manch",
    "Dharam Jagran Samiti", "Sri Ram Sene", "Sanatan Sanstha",
    "Hindu Rashtra Sena", "Bhartiya Gau Raksha Dal", "Hindu Aikya Vedi",
    "Hindu Munnani", "Hindu Makkal Katchi",
    "Sabarimala Ayyappa Seva Samajam", "Abhinav Bharat",
    "Akhil Bharatiya Vidyarthi Parishad", "Vidya Bharati",
    "Saraswati Shishu Mandir", "Vijnana Bharati",
    "Bharatiya Shikshan Mandal", "Samskrita Bharati",
    "Akhil Bharatiya Shaikshik Mahasangh", "Bharatiya Mazdoor Sangh",
    "Swadeshi Jagaran Manch", "Laghu Udyog Bharati",
    "Akhil Bharatiya Adhivakta Parishad", "National Medicos Organisation",
    "Bharatiya Kisan Sangh", "Sahakar Bharati", "Kreeda Bharati",
    "Sanskar Bharati", "Rashtra Sevika Samiti", "Vanavasi Kalyan Ashram",
    "Seva Bharati", "Bharat Vikas Parishad", "Ekal Vidyalaya", "Sakshama",
    "Akhil Bharatiya Poorva Sainik Seva Parishad",
    "Vivekananda International Foundation", "India Foundation",
    "India Policy Foundation", "Deendayal Research Institute",
    "Bharatiya Vichara Kendra", "Prajna Pravah",
    "Syama Prasad Mookerjee Research Foundation",
    "Hindu Swayamsevak Sangh", "Vishwa Hindu Parishad of America",
    "Sewa International", "Overseas Friends of BJP", "Panchjanya",
    "Hindustan Samachar", "Vishwa Samvad Kendra",
    "Muslim Rashtriya Manch", "Rashtriya Sikh Sangat",
    "Bharat-Tibet Maitri Sangh", "Gau Raksha Dal",
    "Gau Raksha Samiti", "Gau Seva Ayog",
]

FUZZY_MATCH_THRESHOLD = 90

# Feature column groups
KEYWORD_COLS = ["field_of_work_scraped", "field_of_work_dataful"]
CATEGORICAL_COLS = ["state_of_registration", "type_of_ngo", "best_match"]
NUMERIC_COLS = ["distance"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_state_csvs(data_dir: str = "ngodarpan_scraped_data") -> pd.DataFrame:
    """Read per-state CSVs from the scraped NGO Darpan database and concatenate"""
    frames = []
    for state in STATES_AND_UTS:
        slug = state.lower().replace(" ", "_")
        path = Path(data_dir) / f"ngodarpan_{slug}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["state_of_registration"] = slug
            frames.append(df)
        else:
            print(f"[WARN] Missing file: {path}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Fuzzy matching for seed labels
# ---------------------------------------------------------------------------

def fuzzy_match_seed_list(
    names: pd.Series,
    seed_orgs: list[str] = HINDUTVA_ORGS,
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> pd.DataFrame:
    """
    For each NGO name, find the closest match in the seed list using
    token-sort-ratio fuzzy matching.

    Returns a DataFrame with columns: best_match_org, match_score, is_hindutva.
    """
    matches = names.apply(
        lambda x: pd.Series(process.extractOne(x, seed_orgs))
    )
    matches.columns = ["best_match_org", "match_score"]
    matches["is_hindutva"] = (matches["match_score"] >= threshold).astype(int)
    return matches


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def encode_keywords_tfidf(
    series: pd.Series,
    max_features: int = 300,
    min_df: int = 5,
) -> tuple[csr_matrix, TfidfVectorizer]:
    """
    Encode  comma-separated keyword column into a TF-IDF sparse matrix.

    Uses sublinear TF scaling and bigrams, which work well for short
    multi-label keyword fields where raw counts would over-weight
    frequently co-occurring terms
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.8,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"\b\w+\b",
        lowercase=True,
    )
    cleaned = series.fillna("").str.replace(",", " ", regex=False)
    features = vectorizer.fit_transform(cleaned)
    return features, vectorizer


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[csr_matrix, list[str], TfidfVectorizer, TfidfVectorizer]:
    """
    Assemble combined feature matrix from keyword TF-IDF columns,
    categorical codes, and numeric features

    Returns
    -------
    X : sparse matrix
    feature_names : list of str
    kw_vec_scraped : fitted TfidfVectorizer for scraped keywords
    kw_vec_dataful : fitted TfidfVectorizer for Dataful keywords
    """
    # Keyword columns
    kw_scraped, kw_vec_scraped = encode_keywords_tfidf(df["field_of_work_scraped"])
    kw_dataful, kw_vec_dataful = encode_keywords_tfidf(df["field_of_work_dataful"])

    # Categorical + numeric
    X_tabular = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    for col in CATEGORICAL_COLS:
        X_tabular[col] = pd.Categorical(X_tabular[col]).codes

    X = hstack([csr_matrix(X_tabular.values), kw_scraped, kw_dataful])

    # Construct readable feature names for interpretation
    feature_names = (
        CATEGORICAL_COLS
        + NUMERIC_COLS
        + [f"kw_scraped_{t}" for t in kw_vec_scraped.get_feature_names_out()]
        + [f"kw_dataful_{t}" for t in kw_vec_dataful.get_feature_names_out()]
    )

    return X, feature_names, kw_vec_scraped, kw_vec_dataful


# ---------------------------------------------------------------------------
# PU learning: training and calibration
# ---------------------------------------------------------------------------

def build_lgbm(
    n_unlabeled: int,
    n_positive: int,
    conservative: bool = False,
    seed: int = RANDOM_STATE,
) -> lgb.LGBMClassifier:
    """
    Returns a configured LightGBM classifier for the PU setting.

    `conservative=True`, uses deeper trees, lower learning rate, and
    stronger regularization — better suited for a high-precision
    """
    if conservative:
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=10,
            min_child_samples=100,
            min_split_gain=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            scale_pos_weight=n_unlabeled / n_positive,
            n_jobs=-1,
            verbose=-1,
            random_state=seed,
        )
    return lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=n_unlabeled / n_positive,
        n_jobs=-1,
        verbose=-1,
        random_state=seed,
    )


def estimate_label_frequency(
    model: lgb.LGBMClassifier,
    X_holdout: csr_matrix,
) -> float:
    """
    Estimate the label frequency *c* = P(labeled | positive) from a held-out
    set of known positives, following Elkan & Noto (2008, §3).
    """
    return model.predict_proba(X_holdout)[:, 1].mean()


def calibrate_scores(raw_scores: np.ndarray, c: float) -> np.ndarray:
    """Elkan-Noto calibration: p(positive | x) ≈ p(labeled=1 | x) / c."""
    return np.clip(raw_scores / c, 0.0, 1.0)


def train_pu_model(
    X: csr_matrix,
    y: np.ndarray,
    conservative: bool = False,
    seed: int = RANDOM_STATE,
) -> tuple[lgb.LGBMClassifier, float, np.ndarray]:
    """
    Full PU training loop:
      1. Hold out a fraction of labeled positives for calibration.
      2. Train LightGBM on (remaining positives + all unlabeled).
      3. Estimate label frequency c on the holdout.
      4. Return calibrated posterior probabilities over the full dataset.
    """
    pos_idx = np.where(y == 1)[0]
    unl_idx = np.where(y == 0)[0]

    pos_train, pos_holdout = train_test_split(
        pos_idx, test_size=HOLDOUT_FRACTION, random_state=seed
    )

    train_idx = np.concatenate([pos_train, unl_idx])
    X_train = X[train_idx]
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(unl_idx))])

    model = build_lgbm(
        n_unlabeled=len(unl_idx),
        n_positive=len(pos_train),
        conservative=conservative,
        seed=seed,
    )
    model.fit(X_train, y_train)

    c = estimate_label_frequency(model, X[pos_holdout])
    raw_scores = model.predict_proba(X)[:, 1]
    calibrated = calibrate_scores(raw_scores, c)

    return model, c, calibrated


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def train_ensemble(
    X: csr_matrix,
    y: np.ndarray,
    seeds: list[int] = ENSEMBLE_SEEDS,
) -> tuple[list[lgb.LGBMClassifier], float, np.ndarray]:
    """
    Train multiple PU models with different random seeds and average their
    calibrated predictions. Seed variation induces diversity through
    subsampling and feature bagging inside LightGBM.
    """
    models = []
    calibration_estimates = []

    for i, seed in enumerate(seeds, 1):
        print(f"  Training ensemble member {i}/{len(seeds)} (seed={seed})")
        model, c, _ = train_pu_model(X, y, conservative=True, seed=seed)
        models.append(model)
        calibration_estimates.append(c)

    # Use the mean calibration constant across folds
    c_mean = np.mean(calibration_estimates)

    # Average raw scores, then calibrate once
    raw_avg = np.mean(
        [m.predict_proba(X)[:, 1] for m in models], axis=0
    )
    calibrated = calibrate_scores(raw_avg, c_mean)

    return models, c_mean, calibrated


def unanimous_agreement(
    models: list[lgb.LGBMClassifier],
    X: csr_matrix,
    c: float,
    threshold: float = AGREEMENT_THRESHOLD,
) -> np.ndarray:
    """
    Return a binary mask where *all* ensemble members independently predict
    a calibrated probability above `threshold`. This trades recall for very
    high precision.
    """
    per_model = np.array([
        calibrate_scores(m.predict_proba(X)[:, 1], c) > threshold
        for m in models
    ])
    return per_model.all(axis=0).astype(int)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def threshold_analysis(
    probs: np.ndarray,
    positive_indices: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Report prediction counts and recall on labeled positives across a
    range of decision thresholds.
    """
    if thresholds is None:
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

    rows = []
    for t in thresholds:
        preds = probs > t
        rows.append({
            "threshold": t,
            "n_predicted": int(preds.sum()),
            "pct_predicted": preds.mean(),
            "recall_on_labeled": preds[positive_indices].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str | None = "figures/feature_importance.png",
) -> pd.DataFrame:
    """Bar chart of LightGBM gain-based feature importance."""
    importance = model.booster_.feature_importance(importance_type="gain")
    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance_gain": importance})
        .sort_values("importance_gain", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    top = df_imp.head(top_n)
    ax.barh(range(len(top)), top["importance_gain"], color="#3b6d8c")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    ax.set_title("LightGBM Feature Importance — Top 20")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df_imp


def compute_shap_summary(
    model: lgb.LGBMClassifier,
    X: csr_matrix,
    feature_names: list[str],
    sample_size: int = 5_000,
    save_dir: str = "figures",
) -> pd.DataFrame:
    """
    Compute SHAP values on a random subsample and produce summary plots.
    Returns a DataFrame of mean |SHAP| per feature.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X.shape[0], size=min(sample_size, X.shape[0]), replace=False)
    X_sample = X[idx].toarray()

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Beeswarm plot
    fig_bee = plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=feature_names, max_display=20, show=False,
    )
    fig_bee.tight_layout()
    fig_bee.savefig(f"{save_dir}/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig_bee)

    # Bar plot
    fig_bar = plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=feature_names, plot_type="bar", max_display=20, show=False,
    )
    fig_bar.tight_layout()
    fig_bar.savefig(f"{save_dir}/shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig_bar)

    mean_abs = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    return mean_abs


def extract_decision_rules(
    X: csr_matrix,
    predictions: np.ndarray,
    feature_names: list[str],
    max_depth: int = 4,
    save_path: str | None = "figures/decision_rules.txt",
) -> str:
    """
    Fit a shallow decision tree on the model's own predictions to extract
    human-readable rules that approximate the classifier's behavior.
    """
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=50, random_state=RANDOM_STATE
    )
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    tree.fit(X_dense, predictions)
    rules = export_text(tree, feature_names=feature_names, max_depth=max_depth)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(rules)

    return rules


# ---------------------------------------------------------------------------
# Top-keyword profiling for high-confidence predictions
# ---------------------------------------------------------------------------

def profile_high_confidence(
    X: csr_matrix,
    probs: np.ndarray,
    threshold: float,
    feature_names: list[str],
    kw_vec_scraped: TfidfVectorizer,
    kw_vec_dataful: TfidfVectorizer,
    top_n: int = 20,
) -> dict:
    """
    For observations above the confidence threshold, report the most
    prevalent keywords and categorical feature values.
    """
    mask = probs > threshold
    X_hc = X[mask]

    n_tabular = len(CATEGORICAL_COLS) + len(NUMERIC_COLS)
    kw1_end = n_tabular + len(kw_vec_scraped.get_feature_names_out())
    kw2_end = kw1_end + len(kw_vec_dataful.get_feature_names_out())

    def _top_terms(X_slice, vectorizer):
        sums = np.asarray(X_slice.sum(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()
        return (
            pd.DataFrame({"keyword": terms, "count": sums})
            .sort_values("count", ascending=False)
            .head(top_n)
        )

    return {
        "n_high_confidence": int(mask.sum()),
        "top_keywords_scraped": _top_terms(X_hc[:, n_tabular:kw1_end], kw_vec_scraped),
        "top_keywords_dataful": _top_terms(X_hc[:, kw1_end:kw2_end], kw_vec_dataful),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PU Learning Pipeline — Hindutva NGO Classification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and prepare labels
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    df = pd.read_csv("labelled_ngo_df_final.csv")
    df["is_hindutva"] = df["is_hindutva"].fillna(0).astype(int)

    y = df["is_hindutva"].values
    pos_idx = np.where(y == 1)[0]
    unl_idx = np.where(y == 0)[0]
    print(f"  Labeled positives: {len(pos_idx):,}")
    print(f"  Unlabeled:         {len(unl_idx):,}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\n[2/6] Building feature matrix...")
    X, feat_names, kw_vec_s, kw_vec_d = build_feature_matrix(df)
    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]:,} columns")

    # ------------------------------------------------------------------
    # 3. Single high-precision model
    # ------------------------------------------------------------------
    print("\n[3/6] Training single PU model (conservative)...")
    model, c, probs = train_pu_model(X, y, conservative=True)
    print(f"  Estimated label frequency c = {c:.3f}")

    ta = threshold_analysis(probs, pos_idx)
    print("\n  Threshold analysis:")
    print(ta.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Ensemble
    # ------------------------------------------------------------------
    print("\n[4/6] Training ensemble...")
    models, c_ens, probs_ens = train_ensemble(X, y)
    unan = unanimous_agreement(models, X, c_ens)

    ens_preds = (probs_ens > ENSEMBLE_THRESHOLD).astype(int)
    print(f"\n  Ensemble (threshold={ENSEMBLE_THRESHOLD}):")
    print(f"    Predicted positives: {ens_preds.sum():,}")
    print(f"    Recall on labeled:   {ens_preds[pos_idx].mean():.1%}")
    print(f"\n  Unanimous agreement (threshold={AGREEMENT_THRESHOLD}):")
    print(f"    Predicted positives: {unan.sum():,}")
    print(f"    Recall on labeled:   {unan[pos_idx].mean():.1%}")

    # ------------------------------------------------------------------
    # 5. Interpretation
    # ------------------------------------------------------------------
    print("\n[5/6] Generating interpretability outputs...")
    imp_df = plot_feature_importance(model, feat_names)
    print(f"  Top 5 features by gain:")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance_gain']:.1f}")

    shap_df = compute_shap_summary(model, X, feat_names)
    print(f"  Top 5 features by mean |SHAP|:")
    for _, row in shap_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")

    hc_predictions = (probs > HIGH_CONFIDENCE_THRESHOLD).astype(int)
    rules = extract_decision_rules(X, hc_predictions, feat_names)
    print(f"  Decision rules saved to figures/decision_rules.txt")

    # ------------------------------------------------------------------
    # 6. Save predictions
    # ------------------------------------------------------------------
    print("\n[6/6] Saving predictions...")
    df["prob_single"] = probs
    df["prob_ensemble"] = probs_ens
    df["pred_high_conf"] = hc_predictions
    df["pred_ensemble"] = ens_preds
    df["pred_unanimous"] = unan

    out_path = "ngo_predictions_pu.csv"
    df.to_csv(out_path, index=False)
    print(f"  Wrote {out_path} ({len(df):,} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()