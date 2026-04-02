# Classifying Hindutva-Affiliated NGOs Using Positive-Unlabeled Learning (PUL)

This repository contains code and selected outputs from a chapter of my dissertation on political violence and civil society in the United States and India.

## Research Problem

India's [NGO Darpan](https://ngodarpan.gov.in/) registry contains roughly 500,000 registered civil society organizations. A small but significant subset are affiliated with the Hindu-nationalist movement (i.e. Sangh Parivar and affiliated groups). Identifying these organizations at scale is difficult: we can compile a curated list of *known* affiliates from secondary sources, but we cannot assume that every organization absent from that list is unaffiliated, as most are simply "unlabeled."

To attempt to gather better data on these orgs, I employ **positive-unlabeled (PU) learning**. Standard binary classifiers trained on "known affiliates vs. everything else" would treat unlabeled affiliates as negative examples, clearly systematically biasing the decision boundary and likely massively underestimating the true prevalence of these groups.

## Approach

```
Assemble known list of Hindutva orgs/affiliates in Sangh Parivar
        │
        ▼
Apply fuzzy string matching (token-sort-ratio ≥ 90)
        │
        ▼
Manual verification ==> Labeled positive set
        │
        ▼
TF-IDF keyword encoding + categorical features
        │
        ▼
LightGBM classifier, trained under PU assumption
        │
        ▼
Elkan & Noto (2008) calibration → P(affiliated | x)
        │
        ▼
5-model ensemble + unanimous-agreement filter
```

1. **Label generation.** Fuzzy string matching against a curated seed list of 61 Hindutva-affiliated organizations, followed by manual review of close matches (score ≥ 90) to produce a reliable positive set.

2. **Features.** TF-IDF encodings of two organizational keyword fields (scraped and from the Dataful supplement), plus categorical codes for state, NGO type, and best seed-list match, plus a continuous distance feature.

3. **PU learning.** LightGBM trained on labeled positives vs. the full unlabeled set. A 10% holdout of known positives estimates the label frequency *c* = P(labeled | positive), used for Elkan-Noto posterior calibration.

4. **High-precision ensemble.** Five LightGBM models (varying random seeds) are averaged, then filtered to cases where all five independently exceed a calibrated threshold of 0.60 — trading recall for very high precision.

5. **Interpretability.** SHAP values, gain-based feature importance, and a shallow surrogate decision tree provide human-readable explanations of what the classifier learned.

## Repository structure

```
├── ngo_pu_classification.py   # Full pipeline: data loading → training → evaluation → export
├── figures/
│   ├── feature_importance.png # LightGBM gain-based importance (top 20)
│   └── decision_rules.txt     # Surrogate tree rules
└── README.md
```

> **Note:** The underlying data (NGO Darpan scrapes and the manually verified label set) are not included in this repository due to size and use constraints. The code is structured to run end-to-end given the input CSVs described in the data loading functions.

## Key results

- The single high-precision model identifies ~1,200 organizations (0.24% of the registry) as ``likely affiliated" at a calibrated threshold of 0.85, with > 95% recall on the labeled positive set.
- The unanimous-agreement ensemble filter produces an even more conservative set, suitable for downstream analysis where false positives are difficult to manually filter.
- Besides locaton, most informative features are based on keywords including in organization's registered scope of activities: terms related to cultural education, cow protection, tribal welfare, and specific organizational naming conventions dominate both SHAP and gain importance.

## Dependencies

```
python >= 3.10
lightgbm
scikit-learn
scipy
pandas
numpy
matplotlib
seaborn
shap
thefuzz
python-Levenshtein
```

## References

- Elkan, C. & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.*
- Bekker, J. & Davis, J. (2020). Learning from positive and unlabeled data: A survey. *Machine Learning*, 109, 719–760.
