"""Train the Breast Cancer ANN model used by the Streamlit app.

The script uses the Breast Cancer Wisconsin Diagnostic dataset bundled with
scikit-learn, selects the top 10 features with ANOVA F-values, trains an ANN,
and saves the fitted model and scaler as pickle files.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

MODEL_FILE = Path("ann_model.pkl")
SCALER_FILE = Path("scaler.pkl")
CONFUSION_MATRIX_FILE = Path("confusion_matrix.png")


def load_and_prepare_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the dataset and split it into feature and target arrays."""
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()
    return df, X, y, dataset.feature_names, dataset.target_names


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the highest scoring features for training."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = np.array(feature_names)[selector.get_support()]
    return X_selected, selected_features


def split_and_scale_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Create train/test splits and fit the scaler on training data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_ann_model(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """Train the ANN classifier with the app's default architecture."""
    model = MLPClassifier(
        hidden_layer_sizes=(50, 50),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def save_artifacts(model: MLPClassifier, scaler: StandardScaler) -> None:
    """Persist model artifacts used by the Streamlit app."""
    with MODEL_FILE.open("wb") as model_file:
        pickle.dump(model, model_file)
    with SCALER_FILE.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)


def evaluate_model(
    model: MLPClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: np.ndarray,
) -> None:
    """Print model metrics and save a confusion matrix image."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(matrix)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)


def main() -> None:
    """Run the full training pipeline."""
    _, X, y, feature_names, target_names = load_and_prepare_data()
    X_selected, selected_features = select_features(X, y, feature_names, k=10)
    print("Selected Features:")
    for feature in selected_features:
        print(f"- {feature}")

    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X_selected, y)
    model = train_ann_model(X_train, y_train)
    save_artifacts(model, scaler)
    evaluate_model(model, X_test, y_test, target_names)
    print(f"Saved {MODEL_FILE}, {SCALER_FILE}, and {CONFUSION_MATRIX_FILE}")


if __name__ == "__main__":
    main()
