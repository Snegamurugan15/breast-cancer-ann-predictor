# Breast Cancer ANN Predictor

A Streamlit machine-learning demo that predicts whether a breast-cancer sample is malignant or benign using the Breast Cancer Wisconsin Diagnostic dataset and an Artificial Neural Network classifier.

> Educational use only: this project is a learning/demo application and is not medical diagnosis software. Do not use it for clinical decisions.

## Features

- Explore the built-in scikit-learn breast cancer dataset.
- Visualize class distribution and feature correlations.
- Train an ANN model with configurable hidden layers, activation, solver, and iteration count.
- Save and reuse `ann_model.pkl` and `scaler.pkl`.
- Enter selected feature values and view the predicted class with probabilities.

## Dataset

The app uses `sklearn.datasets.load_breast_cancer`, which contains 569 samples and 30 numeric features computed from digitized images of breast mass cell nuclei. The target labels are malignant and benign.

## Project Structure

```text
.
|-- streamlit_app.py      # Interactive Streamlit app
|-- train_model.py        # Reproducible ANN training pipeline
|-- ann_model.pkl         # Saved trained ANN model
|-- scaler.pkl            # Saved feature scaler
|-- requirements.txt      # Minimal runtime dependencies
|-- README.md             # Project documentation
`-- image*.png            # App screenshots
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run the App

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Train the Model

```bash
python train_model.py
```

The training script:

- Loads the Breast Cancer Wisconsin Diagnostic dataset.
- Selects the top 10 features using `SelectKBest` with ANOVA F-values.
- Splits and scales the data.
- Trains an `MLPClassifier` with two hidden layers of 50 neurons.
- Saves `ann_model.pkl`, `scaler.pkl`, and `confusion_matrix.png`.

## Screenshots

![Application Home Page](image.png)

![Data Analysis Tab](image-1.png)

![Model Training Tab](image-2.png)

![Classification Report](image-3.png)

![Prediction Tab](image-4.png)

## Notes

This repository was cleaned from an earlier coursework version. The committed virtual environment was removed, the oversized dependency freeze was replaced with a minimal `requirements.txt`, and the training workflow was moved into `train_model.py`.
