# 🎓 Student Dropout & Academic Success: Multi-Model Classification

**ML Assignment 2: M.Tech (AIML), BITS Pilani WILP**

## a. Problem Statement

Higher-education institutions lose students to dropout every year, and the earlier an at-risk
student is identified, the more effective support interventions (tutoring, counselling, financial
aid) can be. The goal of this project is to **predict a student's academic outcome:
`Dropout`, `Enrolled` (still studying after the normal duration) or `Graduate`**, from information
available at enrollment plus their 1st- and 2nd-semester academic performance.

This is a **3-class classification problem**. Five classical ML classifiers are trained on the same
data, compared on six evaluation metrics, and served through an interactive Streamlit app that
evaluates the saved models on an uploaded test CSV.

## b. Dataset Description

| Item | Detail |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, [Dataset #697](https://archive.ics.uci.edu/dataset/697) (Realinho et al., 2021) |
| Instances | **4,424** students (Polytechnic Institute of Portalegre, Portugal) |
| Features | **36** (all numeric / integer-encoded) |
| Target | `Target` ∈ {Dropout, Enrolled, Graduate} |
| Class balance | Graduate 2,209 (49.9%) · Dropout 1,421 (32.1%) · Enrolled 794 (17.9%) |
| Missing values | None |

**Feature groups**

- **Demographic:** Marital status, Nacionality, Gender, Age at enrollment, Displaced, International, Educational special needs
- **Socio-economic:** Mother's/Father's qualification and occupation, Debtor, Tuition fees up to date, Scholarship holder
- **Academic path at enrollment:** Application mode, Application order, Course, Daytime/evening attendance, Previous qualification (+ grade), Admission grade
- **Academic performance:** Curricular units 1st & 2nd semester: credited, enrolled, evaluations, approved, grade, without evaluations (12 features)
- **Macro-economic:** Unemployment rate, Inflation rate, GDP

**Pre-processing**

- The raw file is `;`-separated and one header (`Daytime/evening attendance`) contains a stray tab, which is stripped.
- The target is label-encoded (`Dropout=0, Enrolled=1, Graduate=2`) and the encoder is saved to `model/label_encoder.joblib`.
- **Stratified 80/20 train-test split** (`random_state=26`): 3,539 rows for training and 885 rows for testing. The test split is exported as `test_data.csv`.
- `StandardScaler` is applied inside the pipeline for scale-sensitive models (Logistic Regression, KNN, SVM). Tree and Naive Bayes models use raw values. Because the scaler lives inside each saved pipeline, the app needs no separate preprocessing step.

## c. GitHub Repository Link

- **Repository:** https://github.com/shekhargujari250-debug/ml-assignment-2
- **Live Streamlit App:** https://shekhargujari250-debug-ml-assignment-2-app-82johd.streamlit.app

```
ml-assignment-2/
├── app.py                        # Streamlit app (inference only, no retraining)
├── train_models.py               # Trains, evaluates and saves all 5 models
├── metrics_utils.py              # Shared metric computation (train + app)
├── requirements.txt
├── README.md
├── test_data.csv                 # 885-row held-out test split (features + Target)
├── data/
│   └── students_dropout_full.csv # Original UCI file (4,424 rows)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── label_encoder.joblib      # preprocessing artifact
    ├── metadata.json             # feature list, classes, test metrics
    └── metrics_comparison.csv
```

**Reproduce**

```bash
pip install -r requirements.txt
python train_models.py      # regenerates model/ and test_data.csv
streamlit run app.py
```

## d. Models Used

| # | Model | Configuration |
|---|---|---|
| 1 | Logistic Regression | StandardScaler → multinomial LR, `C=1.0`, `max_iter=2000` |
| 2 | Decision Tree Classifier | `max_depth=8`, `min_samples_leaf=10` (pruned to limit over-fitting) |
| 3 | K-Nearest Neighbor Classifier | StandardScaler → `k=15`, distance-weighted |
| 4 | Naive Bayes Classifier | Gaussian NB |
| 5 | Support Vector Machine | StandardScaler → RBF kernel, `C=1.0`, `gamma='scale'`, `probability=True` |

### Comparison Table (held-out test set, n = 885)

Precision, Recall and F1 are **macro-averaged** over the three classes. AUC is **one-vs-rest macro AUC**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.7559** | 0.8754 | 0.6908 | **0.6593** | **0.6675** | **0.5943** |
| Decision Tree | 0.7277 | 0.8358 | 0.6754 | 0.6526 | 0.6602 | 0.5519 |
| KNN | 0.7130 | 0.8170 | 0.6751 | 0.5927 | 0.6033 | 0.5259 |
| Naive Bayes | 0.6701 | 0.7787 | 0.5950 | 0.5785 | 0.5833 | 0.4515 |
| Support Vector Machine | 0.7525 | **0.8755** | **0.6977** | 0.6522 | 0.6628 | 0.5887 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Best overall: highest Accuracy (75.6%), macro-F1 (0.668) and MCC (0.594), with AUC (0.875) level with SVM. The strongest predictors (approved units and semester grades) relate to the outcome in a roughly monotonic, near-linear way, so a linear decision boundary on standardized features is enough. Graduate recall is very high (0.93) and Dropout recall is good (0.74), but Enrolled recall is only 0.31. The model is also fast, stable and interpretable through its coefficients. |
| **Decision Tree** | Mid-table (Acc 0.728, MCC 0.552). Its lower AUC (0.836) comes from coarse, piece-wise constant leaf probabilities. However, it has the **best Enrolled-class recall (0.40)** of all five models, because axis-aligned splits carve out the "partially-passed" region of semester units. Depth and leaf-size limits were needed; an unpruned tree over-fitted heavily. |
| **KNN** | Weaker (Acc 0.713, MCC 0.526) and has the largest gap between precision (0.675) and recall (0.593). It predicts Graduate very aggressively (recall 0.96) and almost ignores Enrolled (recall 0.20), because the majority class dominates most neighbourhoods. Many of the 36 dimensions are noisy categorical codes (course, occupation), which weakens Euclidean distance even after scaling. |
| **Naive Bayes** | Lowest on every metric (Acc 0.670, MCC 0.452). Its feature-independence assumption is badly violated: the 1st- and 2nd-semester features are strongly correlated (enrolled/approved/grade), so the same evidence is counted several times. Many features are integer-coded categories, not Gaussian. It remains a useful fast baseline. |
| **Support Vector Machine** | Nearly tied with Logistic Regression: the highest AUC (0.8755) and macro-Precision (0.698), with Acc 0.753 and MCC 0.589. It has the best Dropout precision (0.86), so students it flags as dropouts are very likely to actually drop out. The RBF kernel adds little over a linear boundary here, which again suggests the problem is largely linearly separable. It is also the slowest model to train. |
| **Overall, Winner for your dataset?** | **Logistic Regression**, which has the best Accuracy, F1 and MCC (MCC is the most reliable single metric under class imbalance). SVM is a very close second and slightly better on AUC and precision. All models struggle with the minority **Enrolled** class (F1 ≤ 0.41), because "still enrolled" students look like a mix of graduates and dropouts. Class-weighting or resampling would be the next improvement. |

## Streamlit App Features

- **CSV upload** for test data, with validation of the required columns and labels. Both `,` and `;` separated files are accepted, and a bundled `test_data.csv` can be used or downloaded.
- **Model selection dropdown** covering all five classifiers.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC, shown as metric cards.
- **Confusion matrix** heat-map and a per-class **classification report**.
- **Compare All Models** tab: a side-by-side metric table with the best value in each column highlighted, a bar chart for any chosen metric, and the best model by MCC.
- **Download predictions** as CSV, plus a data preview with the target distribution.
- The app only **loads the saved `.joblib` pipelines**. No model is retrained on uploaded data.
