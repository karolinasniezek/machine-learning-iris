# Iris Classification

## Overview

Machine Learning classification project based on the Iris dataset. The objective is to classify iris flowers into one of three species using multiple supervised learning algorithms.

## Technologies

* Python 3.13
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Dataset

Source: `iris.csv`

Target variable:

* `species`

Features:

* sepal_length
* sepal_width
* petal_length
* petal_width

## Data Preparation

### Feature Scaling

All numerical features are standardized using:

```python
StandardScaler()
```

### Label Encoding

Target labels are encoded using:

```python
LabelEncoder()
```

## Train/Test Split

```python
train_size = 0.8
random_state = 0
```

## Models

### K-Nearest Neighbors

```python
KNeighborsClassifier(
    n_neighbors=3
)
```

### Support Vector Classifier

```python
SVC(
    random_state=0
)
```

### Gaussian Process Classifier

```python
GaussianProcessClassifier(
    random_state=0
)
```

## Workflow

1. Load dataset
2. Separate features and target
3. Scale numerical features
4. Encode target labels
5. Split data into training and testing sets
6. Train KNN model
7. Train SVC model
8. Train Gaussian Process model
9. Generate predictions
10. Compare model performance

## Project Structure

```text
machine-learning-iris/
├── data/
│   └── iris.csv
├── src/
│   ├── 01_load_data.py
│   ├── 02_scaling.py
│   ├── 03_train_test_split.py
│   ├── 04_knn_model.py
│   ├── 05_knn_training.py
│   ├── 06_label_encoding.py
│   ├── 07_svc_model.py
│   ├── 08_gaussian_process.py
│   └── 09_compare_models.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/09_compare_models.py
```
