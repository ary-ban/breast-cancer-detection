## Breast Cancer Detection

This project tackles the problem of automatically classifying breast tumours
as **malignant** or **benign** using machine learning.  We use the
**Wisconsin Breast Cancer Dataset**, a widely studied collection of
measurements derived from digitised images of fine‑needle aspirates of
breast masses.  Each of the 569 instances is described by 30 real‑valued
features, such as radius, texture and perimeter, representing properties
computed from the nuclei present in the image.

### Goal

Develop a binary classifier that predicts whether a tumour is cancerous
(malignant) or non‑cancerous (benign) based on these features, enabling
early detection and supporting medical decision‑making.

### Methodology

* **Data loading** – The dataset is loaded via
  `sklearn.datasets.load_breast_cancer`.  The target labels are encoded
  as `0` for malignant and `1` for benign cases.
* **Preprocessing** – Features are scaled with `StandardScaler` to ensure
  all variables are on the same scale.  This step benefits algorithms
  like logistic regression that assume comparable feature scales.
* **Models** – Train and compare three classifiers: `LogisticRegression`,
  `RandomForestClassifier` and a linear `SupportVectorMachine` with
  probability estimates.  Random forests capture non‑linear
  relationships, and SVMs can provide robust decision boundaries.
* **Evaluation** – Use an 80/20 train/test split combined with 5‑fold
  cross‑validation to estimate generalisation performance.  Report test
  accuracy, cross‑validated accuracy and the area under the ROC curve
  (AUC) for each model.  High AUC values indicate excellent
  discriminative power.

### Usage

Run the script from this directory to train and evaluate the model:

```bash
python breast_cancer_detection.py
```

The console will display the overall accuracy and a breakdown of
performance for malignant and benign classes.

### Possible extensions

* **Feature selection** – Investigate which features contribute most to
  classification.  You can use methods like recursive feature
  elimination or LASSO regularisation.
* **Alternative models** – Try tree‑based algorithms such as Random
  Forests or Gradient Boosting Machines, which may capture non‑linear
  relationships.
* **Model interpretability** – Use SHAP values or coefficient analysis
  to understand how each feature influences the predictions.
