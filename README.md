# ML Lab — PCCSL508 | APJ Abdul Kalam Technological University
## B.Tech Semester 5 · Computer Science & Engineering

> **Course Code:** PCCSL508 — Machine Learning Lab  
> **University:** APJ Abdul Kalam Technological University (APJKTU)  
> **Program:** B.Tech CSE, Semester 5  
> **Language:** Python 3  

---

## 📋 Experiment Index

| # | Experiment | Algorithm / Technique | Dataset |
|---|------------|----------------------|---------|
| 01 | [Simple Linear Regression](Exp1.py) | Ordinary Least Squares | California Housing |
| 02 | [Polynomial Regression](Exp2.py) | Polynomial Feature Expansion | Seaborn MPG |
| 03 | [Ridge & Lasso Regression](Exp3.py) | L2 / L1 Regularisation (RidgeCV, LassoCV) | sklearn Diabetes |
| 04 | [Logistic Regression](Exp4.py) | Binary Classification + StandardScaler | California Housing |
| 05 | [MLE vs MAP Estimation](Exp5.py) | Maximum Likelihood vs L1/L2 MAP priors | sklearn Breast Cancer |
| 06 | [Naïve Bayes Classifier](Exp6.py) | Multinomial NB & Bernoulli NB | 20 Newsgroups |
| 07 | [K-Nearest Neighbours](Exp7.py) | KNN (k = 1, 3, 5) | Fashion-MNIST |
| 08 | [Decision Tree (ID3)](Exp8.py) | Entropy-based Decision Tree | Online Retail |
| 09 | [SVM — Linear Kernel](Exp9.py) | Support Vector Machine | Iris |
| 10 | [SVM — Kernel Comparison](Exp10.py) | Linear / Polynomial / RBF kernels | Fashion-MNIST |
| 11 | [Multilayer Perceptron](Exp11.py) | MLP with varying depth (Keras) | WineQT |
| 12 | [Activation Function Comparison](Exp12.py) | Sigmoid / Tanh / ReLU on MNIST (Keras) | MNIST |
| 13 | [K-Means Clustering](Exp13.py) | Elbow Method + Silhouette Score | sklearn Digits |

> Experiments 1–13 correspond to the prescribed syllabus (PCCSL508). Experiment 14 (EM Algorithm) is covered in the theory component.

---

## 🗂 Repository Structure

```
ML_LAB_S5_CSE/
├── Datasets/
│   ├── housing.csv           # California Housing dataset
│   ├── WineQT.csv            # Wine Quality dataset
│   ├── Online Retail.xlsx    # UCI Online Retail transaction data
│   └── diabetes.csv          # Pima Indians Diabetes dataset
│
├── Output Diagrams/          # Saved plots from each experiment
│   ├── simple_linear_regression_exp1.png
│   ├── polynomial_regression_exp2.png
│   ├── ridge_lasso_exp3.png
│   ├── ridge_lasso_fig2_exp3.png
│   ├── logistic_regression_exp4.png
│   ├── logistic_regression_fig2_exp4.png
│   ├── mle_map_exp5.png
│   ├── decision_tree_exp8.png
│   ├── svm_iris_exp9.png
│   ├── svm_kernels_comparison_exp10.png
│   ├── mlp_wine_comparison_exp11.png
│   ├── activation_comparison_curves.png
│   ├── val_accuracy_exp12.png
│   ├── val_loss_exp12.png
│   ├── elbow_method_exp13.png
│   ├── silhouette_score_exp13.png
│   └── kmeans_metrics_combined_exp13.png
│
├── Exp1.py   → Simple Linear Regression
├── Exp2.py   → Polynomial Regression
├── Exp3.py   → Ridge & Lasso Regression
├── Exp4.py   → Logistic Regression
├── Exp5.py   → MLE vs MAP Estimation
├── Exp6.py   → Naïve Bayes Classifier
├── Exp7.py   → K-Nearest Neighbours
├── Exp8.py   → Decision Tree (ID3/Entropy)
├── Exp9.py   → SVM — Linear Kernel
├── Exp10.py  → SVM — Kernel Comparison
├── Exp11.py  → Multilayer Perceptron (MLP)
├── Exp12.py  → Activation Function Comparison
├── Exp13.py  → K-Means Clustering
│
├── MACHINE_LEARNING_LAB_PCCSL508_SYLLABUS.pdf
├── .gitignore
└── README.md
```

---

## 🔧 Setup & Requirements

### Prerequisites
- Python 3.8+
- pip

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras openpyxl
```

Or using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras openpyxl
```

### Running an experiment

All scripts are run from the **repository root** (so relative dataset paths resolve correctly):

```bash
python Exp1.py
python Exp2.py
# ... and so on
```

---

## 📊 Experiment Summaries

### Exp 1 — Simple Linear Regression
Implements OLS from scratch using NumPy on the California Housing dataset. Reports slope, intercept, MSE, and R² without using sklearn's `LinearRegression`.

### Exp 2 — Polynomial Regression
Compares linear regression against polynomial fits of degree 2, 3, and 5 on the MPG dataset (displacement vs fuel efficiency) using `PolynomialFeatures`.

### Exp 3 — Ridge & Lasso Regression
Demonstrates shrinkage regularisation using `RidgeCV` and `LassoCV` on the sklearn Diabetes dataset. Plots coefficient magnitude to visualise feature selection (L1 sparsity).

### Exp 4 — Logistic Regression
Binary classification on the California Housing dataset, comparing model performance with and without feature scaling (`StandardScaler`). Reports accuracy, precision, recall, F1, and confusion matrix.

### Exp 5 — MLE vs MAP Estimation
Compares unconstrained logistic regression (MLE) against L1 and L2 penalised variants (MAP priors) on the Breast Cancer dataset. Visualises coefficient magnitude to show how priors shrink/zero out weights.

### Exp 6 — Naïve Bayes Classifier
Applies `MultinomialNB` and `BernoulliNB` to the 20 Newsgroups text corpus using bag-of-words features. Compares accuracy and macro F1 across three news categories.

### Exp 7 — K-Nearest Neighbours
Classifies Fashion-MNIST images (flattened 784-dim vectors) using KNN for k ∈ {1, 3, 5}. Runs on a 10 000-sample training subset for tractable computation.

### Exp 8 — Decision Tree (ID3)
Segments Online Retail customers into high/low value groups using an entropy-based decision tree (max depth 4). Visualises the tree and reports feature importances.

### Exp 9 — SVM Linear Kernel
Trains a linear SVM on the Iris dataset (2 features, binary split), plots the decision boundary, margin hyperplanes, and support vectors.

### Exp 10 — SVM Kernel Comparison
Benchmarks Linear, Polynomial (degree 3), and RBF SVM kernels on Fashion-MNIST (5 000 train / 1 000 test subset). Reports per-kernel accuracy.

### Exp 11 — Multilayer Perceptron
Builds and compares three Keras MLP architectures (simple / medium / deep) on WineQT quality classification. Uses `LabelEncoder` + `to_categorical` for multi-class output.

### Exp 12 — Activation Function Comparison
Trains the same two-hidden-layer Keras network on MNIST with Sigmoid, Tanh, and ReLU activations. Plots validation accuracy and loss curves over 5 epochs.

### Exp 13 — K-Means Clustering
Applies K-Means++ to the sklearn Digits dataset for K ∈ [2, 14]. Uses the Elbow Method (inertia) and Silhouette Score to find the optimal number of clusters.

---

## 📄 Syllabus Reference

The full prescribed syllabus is included as [`MACHINE_LEARNING_LAB_PCCSL508_SYLLABUS.pdf`](MACHINE_LEARNING_LAB_PCCSL508_SYLLABUS.pdf).

---

## 👤 Author

**Aravind Lal**  
B.Tech CSE, Semester 5  
APJ Abdul Kalam Technological University  
GitHub: [@mfscpayload-690](https://github.com/mfscpayload-690)

---

*All experiments are written for academic purposes as part of the PCCSL508 lab curriculum.*
