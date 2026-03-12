import argparse
from ensurepip import bootstrap

from matplotlib.pylab import gamma
import numpy as np
import pandas as pd
import os
import joblib

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


# --------------------
# TORCH-BASED NEURAL NETWORK MODELS (scikit-learn compatible)
# --------------------
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, Any, Tuple


class TorchCNNLSTMNet(nn.Module):
    def __init__(self, input_dim, out_dim, conv_channels, kernel_size, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        padding = max(0, kernel_size // 2)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if (dropout > 0 and lstm_layers > 1) else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, seq_len)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class _TorchBaseClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn compatible wrapper for PyTorch classifiers.

    Requirements for sklearn cloning:
      - __init__ must only assign parameters to attributes *without modification*.
      - All learned attributes must be suffixed with "_" and created in fit().
    """

    def __init__(
        self,
        input_dim=None,
        num_classes=2,
        epochs=25,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        val_split=0.1,
        patience=5,
        random_state=42,
        verbose=0,
    ):
        # Store parameters exactly as provided (no casting) for sklearn.clone compatibility
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self):
        raise NotImplementedError

    def _prepare_y(self, y_np):
        if int(self.num_classes) == 2:
            return torch.tensor(y_np.astype(np.float32).reshape(-1, 1))
        return torch.tensor(y_np.astype(np.int64).reshape(-1))

    def _loss_fn(self):
        if int(self.num_classes) == 2:
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()
    
    def save_torchscript(self, path: str, example_batch_size: int = 1):
        """
        Saves a TorchScript module (NOT a pickled python dict).
        This is what C++/libtorch runtimes typically expect.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Model not fitted; call fit() before saving.")

        self.model_.eval()
        input_dim = int(getattr(self, "input_dim_", self.input_dim))
        example = torch.zeros((example_batch_size, input_dim), dtype=torch.float32)

        scripted = torch.jit.trace(self.model_.cpu(), example)  # trace is fine here (static control flow)
        scripted.save(path)

    def save_torch(self, path: str):
        if not hasattr(self, "model_"):
            raise RuntimeError("Model not fitted; call fit() before saving.")
        payload = {
            "state_dict": self.model_.state_dict(),
            "params": self.get_params(deep=False),
            "input_dim_": getattr(self, "input_dim_", None),
            "num_classes_": getattr(self, "num_classes_", None),
            "classes_": getattr(self, "classes_", None),
        }
        torch.save(payload, path)



    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y)

        # Learned attributes (with trailing "_")
        self.input_dim_ = int(self.input_dim) if self.input_dim is not None else int(X_np.shape[1])
        self.num_classes_ = int(self.num_classes)

        # Reproducibility
        rs = int(self.random_state) if self.random_state is not None else 42
        np.random.seed(rs)
        torch.manual_seed(rs)

        self.model_ = self._build_model()
        device = torch.device("cpu")
        self.model_.to(device)

        # Train/val split
        n = X_np.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        val_split = float(self.val_split)
        n_val = int(max(1, val_split * n)) if val_split > 0 else 0
        val_idx = idx[:n_val] if n_val > 0 else np.array([], dtype=int)
        tr_idx = idx[n_val:] if n_val > 0 else idx

        X_tr = torch.tensor(X_np[tr_idx], dtype=torch.float32, device=device)
        y_tr = self._prepare_y(y_np[tr_idx]).to(device)

        if n_val > 0:
            X_val = torch.tensor(X_np[val_idx], dtype=torch.float32, device=device)
            y_val = self._prepare_y(y_np[val_idx]).to(device)
        else:
            X_val, y_val = None, None

        # Optimizer / loss
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = self._loss_fn()

        # Training loop with early stopping on val loss (if val_split > 0)
        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        epochs = int(self.epochs)
        batch_size = int(self.batch_size)

        for epoch in range(epochs):
            self.model_.train()
            perm = torch.randperm(X_tr.shape[0])
            for i in range(0, X_tr.shape[0], batch_size):
                b = perm[i : i + batch_size]
                xb = X_tr[b]
                yb = y_tr[b]

                optimizer.zero_grad()
                out = self.model_(xb)

                if self.num_classes_ == 2:
                    loss = loss_fn(out, yb)
                else:
                    loss = loss_fn(out, yb)

                loss.backward()
                optimizer.step()

            if X_val is None:
                continue

            self.model_.eval()
            with torch.no_grad():
                out_val = self.model_(X_val)
                if self.num_classes_ == 2:
                    vloss = loss_fn(out_val, y_val).item()
                else:
                    vloss = loss_fn(out_val, y_val).item()

            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs} - val_loss={vloss:.5f}")

            if vloss + 1e-8 < best_val:
                best_val = vloss
                best_state = {k: v.clone().detach().cpu() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # sklearn convention
        self.classes_ = np.unique(y_np) if self.num_classes_ != 2 else np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t).cpu().numpy()

        if self.num_classes_ == 2:
            probs_pos = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
            probs = np.vstack([1.0 - probs_pos, probs_pos]).T
            return probs

        # multiclass
        logits = logits.reshape(-1, self.num_classes_)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        if probs.shape[1] == 2:
            return (probs[:, 1] >= 0.5).astype(int)
        return probs.argmax(axis=1)


class TorchMLPClassifier(_TorchBaseClassifier):
    """Feed-forward MLP for tabular data."""

    def __init__(
        self,
        input_dim=None,
        num_classes=2,
        hidden_dims=(128, 64, 32),
        dropout=0.2,
        epochs=25,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        val_split=0.1,
        patience=5,
        random_state=42,
        verbose=0,
    ):
        # Store params exactly as provided for sklearn.clone compatibility
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self):
        hidden = tuple(int(h) for h in self.hidden_dims)
        drop = float(self.dropout)
        out_dim = 1 if int(self.num_classes) == 2 else int(self.num_classes)

        layers = []
        in_dim = int(self.input_dim_)  # learned in fit()
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if drop > 0:
                layers.append(nn.Dropout(drop))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)


class TorchCNNLSTMClassifier(_TorchBaseClassifier):
    """CNN-LSTM hybrid adapted for tabular data by treating features as a 1D sequence.

    Included for benchmarking only; for AI4I-style engineered tabular features,
    MLPs/trees are typically more appropriate.
    """

    def __init__(
        self,
        input_dim=None,
        num_classes=2,
        conv_channels=32,
        kernel_size=3,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.2,
        epochs=25,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        val_split=0.1,
        patience=5,
        random_state=42,
        verbose=0,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self):
        out_dim = 1 if int(self.num_classes) == 2 else int(self.num_classes)
        input_dim = int(self.input_dim_)  # learned in fit()

        conv_channels = int(self.conv_channels)
        kernel_size = int(self.kernel_size)
        lstm_hidden = int(self.lstm_hidden)
        lstm_layers = int(self.lstm_layers)
        drop = float(self.dropout)

        return TorchCNNLSTMNet(
            input_dim=input_dim,
            out_dim=out_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=drop,
        )

class ScaledTorchModel(nn.Module):
    def __init__(self, net: nn.Module, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.scale
        return self.net(x)  # logits


# --------------------
# CONFIG
# --------------------
DATA_PATH = "ai4i2020.csv"
N_SPLITS = 5
RANDOM_STATE = 42

BINARY_TARGET = "Machine failure"

# Failure type columns in AI4I 2020 dataset
FAILURE_TYPE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Mapping of failure type columns to integer labels
FAILURE_TYPE_MAPPING = {
    "TWF": 0,  # Tool Wear Failure
    "HDF": 1,  # Heat Dissipation Failure
    "PWF": 2,  # Power Failure
    "OSF": 3,  # Overstrain Failure
    "RNF": 4,  # Random Failure
}

FAILURE_TYPE_NAMES = {
    0: "Tool Wear Failure (TWF)",
    1: "Heat Dissipation Failure (HDF)",
    2: "Power Failure (PWF)",
    3: "Overstrain Failure (OSF)",
    4: "Random Failure (RNF)",
}


# --------------------
# UTILS
# --------------------
def sanitize_column_names(columns):
    """Make column names safe for XGBoost (no [, ], <, >, spaces, parentheses)."""
    safe_cols = []
    for col in columns:
        c = str(col)
        for bad in ["[", "]", "<", ">", "(", ")", " "]:
            c = c.replace(bad, "_")
        safe_cols.append(c)
    return safe_cols


def load_and_prepare_data(path: str):
    """Load AI4I 2020 dataset and build:
    - X_all: features (no leakage)
    - y_binary: Machine failure (0/1)
    - X_fail: features for failed machines only
    - y_fail_type: multiclass failure type label for failed machines
    """
    df = pd.read_csv(path)

    # Drop meta / non-feature columns
    drop_cols = []
    for col in df.columns:
        if col in ["UDI", "Product ID"]:
            drop_cols.append(col)
        if "Unnamed" in str(col):
            drop_cols.append(col)
    drop_cols = list(dict.fromkeys(drop_cols))
    if drop_cols:
        print("Dropping columns (meta/non-feature):", drop_cols)
        df = df.drop(columns=drop_cols)

    # Check necessary columns
    for col in [BINARY_TARGET] + FAILURE_TYPE_COLS:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. Available columns: {list(df.columns)}"
            )

    # ----- Stage 1: binary target -----
    y_binary = df[BINARY_TARGET]

    # Features WITHOUT any failure-type columns (to avoid leakage)
    feature_cols = [
        col
        for col in df.columns
        if col not in [BINARY_TARGET] + FAILURE_TYPE_COLS
    ]

    X_all = df[feature_cols].copy()

    # One-hot encode categorical features (e.g. Type)
    X_all = pd.get_dummies(X_all, drop_first=True)

    # Sanitize feature names for XGBoost
    X_all.columns = sanitize_column_names(X_all.columns)

    print("Full dataset:")
    print("  Data shape:", df.shape)
    print("  Feature matrix shape (X_all):", X_all.shape)
    print("  Features:", list(X_all.columns))
    print("  Binary target:", BINARY_TARGET, "| classes:", sorted(y_binary.unique()))

    # ----- Stage 2: multiclass failure type (only failed machines) -----
    fail_mask = y_binary == 1
    df_fail = df.loc[fail_mask].copy()

    # Build multiclass label y_fail_type
    y_fail_type_list = []
    kept_indices = []
    missing_flags = []
    for idx, row in df_fail.iterrows():
        label = None
        for ft_col, ft_idx in FAILURE_TYPE_MAPPING.items():
            val = row.get(ft_col) if isinstance(row, pd.Series) else row[ft_col]
            if pd.isna(val):
                continue
            try:
                if int(val) == 1:
                    label = ft_idx
                    break
            except Exception:
                continue

        if label is None:
            missing_flags.append(idx)
            continue

        y_fail_type_list.append(label)
        kept_indices.append(idx)

    if missing_flags:
        print(
            f"Warning: {len(missing_flags)} failed samples have no failure-type flag and will be skipped."
        )

    if not y_fail_type_list:
        raise ValueError("No valid failed samples with failure-type flags found after filtering.")

    y_fail_type = pd.Series(y_fail_type_list, index=kept_indices, name="failure_type")

    X_fail = X_all.loc[kept_indices].copy()

    print("\nSubset of failed machines (for Stage 2):")
    print("  Number of failed samples:", X_fail.shape[0])
    print("  Feature matrix shape (X_fail):", X_fail.shape)
    class_counts = y_fail_type.value_counts().sort_index()
    print("  Failure type distribution:")
    for label, count in class_counts.items():
        print(f"    {label}: {FAILURE_TYPE_NAMES[label]} -> {count} samples")

    return X_all, y_binary, X_fail, y_fail_type



# --------------------
# MODEL BUILDERS
# --------------------
def build_binary_models(input_dim: int, random_state: int = 42):
    """Binary classifiers for Stage 1 (Machine failure).

    This version uses FIXED (already-tuned) NN hyperparameters.
    """
    models = {}

    # SVM with RBF kernel + scaling
    svm_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=38158.24483486982,
                gamma=0.19400795098997506,
                probability=True,
                random_state=random_state,
                class_weight="balanced",
            )),
        ]
    )
    models["SVM"] = svm_clf

    # Random Forest
    rf_clf = RandomForestClassifier(
        bootstrap = True,
        n_estimators=678,
        max_depth=64,
        max_features = None,
        min_samples_leaf=1,
        min_samples_split = 3,
        # max_features = "sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    models["RandomForest"] = rf_clf

    # KNN 
    knn_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=3,
                p = 2,
                weights="uniform",
                metric="euclidean",
            )),
        ]
    )
    models["KNN"] = knn_clf

    # XGBoost binary classifier
    xgb_clf = XGBClassifier(
        n_estimators=174,
        learning_rate=0.5,
        max_depth=16,
        subsample=1.0,
        colsample_bytree=0.8882789316208037,
        min_child_weight = 1,
        reg_alpha = 0.00019277781202900005,
        reg_lambda = 7.530077446392522e-05,
        scale_pos_weight = 11.775324835750336,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )
    models["XGBoost"] = xgb_clf

    # --------------------
    # FIXED / TUNED TORCH NNs (no RandomizedSearchCV)
    # --------------------
    torch_mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", TorchMLPClassifier(
                input_dim=None,            # infer from X in fit()
                num_classes=2,
                hidden_dims=(512, 256),
                dropout=0.7,
                epochs=266,
                batch_size=64,
                lr=0.0005956167817839583,
                weight_decay=1.3703491270567322e-06,
                val_split=0.11741911784783401,
                patience=40,
                random_state=random_state,
                verbose=0,
            )),
        ]
    )
    models["TorchMLP"] = torch_mlp

    torch_cnn_lstm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", TorchCNNLSTMClassifier(
                input_dim=None,            # infer from X in fit()
                num_classes=2,
                conv_channels=512,
                kernel_size=2,
                lstm_hidden=64,
                lstm_layers=1,
                dropout=0.5564460235189254,
                epochs=281,
                batch_size=64,
                lr=0.0002725235478354031,
                weight_decay=3.9897868464771114e-05,
                val_split=0.17344223136997572,
                patience=15,
                random_state=random_state,
                verbose=0,
            )),
        ]
    )
    models["TorchCNNLSTM"] = torch_cnn_lstm

    return models


def build_multiclass_models(input_dim: int, num_classes: int, random_state: int = 42):
    """Multiclass classifiers for Stage 2 (failure type).

    This version uses FIXED (already-tuned) NN hyperparameters.
    """
    models = {}

    # SVM with RBF kernel + scaling
    svm_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=54390.355234820105,
                gamma=0.00011130872555105644,
                probability=True,
                random_state=random_state,
                class_weight="balanced",
            )),
        ]
    )
    models["SVM"] = svm_clf

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=3000,
        max_depth=8,
        min_samples_leaf=3,
        max_features = "log2",
        min_samples_split = 2,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    models["RandomForest"] = rf_clf

    # KNN
    knn_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=2,
                p=3,
                weights="uniform",
                metric="euclidean",
            )),
        ]
    )
    models["KNN"] = knn_clf

    # XGBoost multiclass
    xgb_clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.7,
        gamma=0.0,
        min_child_weight=1,
        reg_alpha=0.00545360600018444,
        reg_lambda=1.3920591060072682e-05,
        max_depth=7,
        subsample=0.6093098825706535,
        colsample_bytree=1.0,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=1,
    )
    models["XGBoost"] = xgb_clf

    # --------------------
    # FIXED / TUNED TORCH NNs (no RandomizedSearchCV)
    # --------------------
    torch_mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", TorchMLPClassifier(
                input_dim=None,            # infer from X in fit()
                num_classes=num_classes,
                hidden_dims=(64, 32),
                dropout=0.34200295600842817,
                epochs=300,
                batch_size=128,
                lr=0.0025571398746863793,
                weight_decay=1e-09,
                val_split=0.05434371670067722,
                patience=40,
                random_state=random_state,
                verbose=0,
            )),
        ]
    )
    models["TorchMLP"] = torch_mlp

    torch_cnn_lstm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", TorchCNNLSTMClassifier(
                input_dim=None,            # infer from X in fit()
                num_classes=num_classes,
                conv_channels=512,
                kernel_size=5,
                lstm_hidden=32,
                lstm_layers=1,
                dropout=0.0,
                epochs=178,
                batch_size=128,
                lr=0.007015498894931689,
                weight_decay=9.985633241696608e-06,
                val_split=0.1627700455879219,
                patience=42,
                random_state=random_state,
                verbose=0,
            )),
        ]
    )
    models["TorchCNNLSTM"] = torch_cnn_lstm

    return models


# --------------------
# EVALUATION
# --------------------
def evaluate_models(X, y, models, n_splits=5, random_state=42, multiclass=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = (["accuracy", "precision", "recall", "f1"]
               if not multiclass else
               ["accuracy", "precision_macro", "recall_macro", "f1_macro"])

    results = {}

    for name, model in models.items():
        cv_results = cross_validate(
            model, X, y, cv=skf, scoring=scoring,
            return_train_score=False, n_jobs=1,
        )

        lines = [f"\n===== Evaluating {name} ====="]
        metrics_summary = {}

        for metric in scoring:
            scores = cv_results[f"test_{metric}"]
            metrics_summary[metric] = (scores.mean(), scores.std())
            lines.append(f"{metric:>14}: {scores.mean():.4f} ± {scores.std():.4f}")

        print("\n".join(lines))
        results[name] = metrics_summary

    f1_key = "f1" if not multiclass else "f1_macro"
    best_model_name = max(results, key=lambda m: results[m][f1_key][0])
    print(f"\nBest model by mean {f1_key}: {best_model_name}")

    return results, best_model_name

def plot_model_comparison(results, multiclass=False, title_prefix="", save_path=None):
    """
    Plot bar charts with error bars for each metric across models.

    results: dict from evaluate_models (model_name -> metric -> (mean, std))
    multiclass: if True, use 'precision_macro', etc. just for labeling.
    title_prefix: string added to plot title, e.g. "Stage 1 - Binary"
    save_path: if not None, path to save figure (e.g. 'figures/stage1_metrics.png')
    """

    # Decide which metrics we expect based on problem type
    if not multiclass:
        metrics = ["accuracy", "precision", "recall", "f1"]
        ylabel = "Score"
    else:
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        ylabel = "Score (macro)"

    models = list(results.keys())

    # Prepare bar positions
    x = np.arange(len(models))
    width = 0.18  # width of each bar

    plt.figure(figsize=(12, 6))

    # For each metric, collect mean and std
    for i, metric in enumerate(metrics):
        means = [results[m][metric][0] for m in models]
        stds = [results[m][metric][1] for m in models]
        # Shift each metric's bars so they don't overlap
        plt.bar(
            x + (i - len(metrics) / 2) * width + width / 2,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=metric,
        )

    plt.xticks(x, models, rotation=30)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.05)
    plt.title(f"{title_prefix} - Cross-Validation Metrics")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved model comparison figure to {save_path}")

    plt.show()

def detailed_evaluation(X, y, model, n_splits: int = 5, random_state: int = 42, multiclass: bool = False, title_prefix: str = "", save_cm_path: str = None):
    """Confusion matrix & classification report for the best model."""
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=1)

    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)

    if not multiclass:
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
    else:
        prec = precision_score(y, y_pred, average="macro", zero_division=0)
        rec = recall_score(y, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print("\n===== Confusion Matrix =====")
    print(cm)
    print("\n===== Metrics (cross-validated predictions) =====")
    print(f"Accuracy : {acc:.4f}")
    if not multiclass:
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("\n===== Classification Report =====")
        print(classification_report(y, y_pred, digits=4))
    else:
        print(f"Precision_macro: {prec:.4f}")
        print(f"Recall_macro   : {rec:.4f}")
        print(f"F1_macro       : {f1:.4f}")
        print("\n===== Classification Report (per class) =====")
        target_names = [FAILURE_TYPE_NAMES[i] for i in sorted(np.unique(y))]
        print(classification_report(y, y_pred, digits=4, target_names=target_names))

    # --- Plot confusion matrix as heatmap ---
    plt.figure(figsize=(6, 5))

    if multiclass:
        labels = [FAILURE_TYPE_NAMES[i] for i in sorted(np.unique(y))]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No failure", "Failure"],
            yticklabels=["No failure", "Failure"],
        )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{title_prefix} - Confusion Matrix")
    plt.tight_layout()

    if save_cm_path is not None:
        plt.savefig(save_cm_path, dpi=300)
        print(f"Saved confusion matrix figure to {save_cm_path}")

    plt.show()



# --------------------
# STAGE 1
# --------------------
def run_stage1_binary():
    # Load data
    X_all, y_binary, _, _ = load_and_prepare_data(DATA_PATH)

    print("\n====================")
    print("STAGE 1: Binary classification - Predict Machine failure (0/1)")
    print("====================")

    binary_models = build_binary_models(
        input_dim=X_all.shape[1],
        random_state=RANDOM_STATE
    )

    binary_results, best_binary_name = evaluate_models(
        X_all,
        y_binary,
        binary_models,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        multiclass=False,
    )

    plot_model_comparison(
        binary_results,
        multiclass=False,
        title_prefix="Stage 1 - Binary (Machine failure)",
        save_path="stage1_binary_model_metrics.png",
    )

    best_binary_model = binary_models[best_binary_name]

    print(f"\n[Stage 1] Best model by F1: {best_binary_name}")
    print("[Stage 1] Refitting best model on full data...")
    best_binary_model.fit(X_all, y_binary)

    detailed_evaluation(
        X_all,
        y_binary,
        best_binary_model,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        multiclass=False,
        title_prefix=f"Stage 1 - Best model: {best_binary_name}",
        save_cm_path="stage1_binary_confusion_matrix.png",
    )

    # Save fitted model
    if isinstance(best_binary_model, Pipeline):
        scaler = best_binary_model.named_steps.get("scaler", None)
        clf = best_binary_model.named_steps.get("clf", None)
    else:
        scaler, clf = None, None

    if clf is not None and hasattr(clf, "model_") and scaler is not None:
        net = clf.model_.cpu().eval()
        scaled_net = ScaledTorchModel(net, scaler.mean_, scaler.scale_).cpu().eval()

        example = torch.zeros((1, int(clf.input_dim_)), dtype=torch.float32)
        scripted = torch.jit.trace(scaled_net, example)
        scripted.save("stage1_scaled_model.ts.pt")

        saved_model_path = "stage1_scaled_model.ts.pt"
        print(f"Saved Stage 1 single-file TorchScript model: {saved_model_path}")
    else:
        joblib.dump(best_binary_model, "binary_failure_model.pkl")
        saved_model_path = "binary_failure_model.pkl"
        print(f"Saved Stage 1 sklearn pipeline: {saved_model_path}")

    return {
        "stage": "binary",
        "best_model_name": best_binary_name,
        "metrics_plot": "stage1_binary_model_metrics.png",
        "confusion_matrix": "stage1_binary_confusion_matrix.png",
        "saved_model": saved_model_path,
    }


# --------------------
# STAGE 2
# --------------------
def run_stage2_multiclass():
    # Load data
    _, _, X_fail, y_fail_type = load_and_prepare_data(DATA_PATH)

    print("\n====================")
    print("STAGE 2: Multiclass classification - Predict failure type (for failed machines)")
    print("====================")

    num_classes = len(FAILURE_TYPE_MAPPING)

    multiclass_models = build_multiclass_models(
        input_dim=X_fail.shape[1],
        num_classes=num_classes,
        random_state=RANDOM_STATE
    )

    multiclass_results, best_multi_name = evaluate_models(
        X_fail,
        y_fail_type,
        multiclass_models,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        multiclass=True,
    )

    plot_model_comparison(
        multiclass_results,
        multiclass=True,
        title_prefix="Stage 2 - Multiclass (Failure type)",
        save_path="stage2_multiclass_model_metrics.png",
    )

    best_multiclass_model = multiclass_models[best_multi_name]

    print(f"\n[Stage 2] Best model by F1_macro: {best_multi_name}")
    print("[Stage 2] Refitting best model on full failed subset...")
    best_multiclass_model.fit(X_fail, y_fail_type)

    detailed_evaluation(
        X_fail,
        y_fail_type,
        best_multiclass_model,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        multiclass=True,
        title_prefix=f"Stage 2 - Best model: {best_multi_name}",
        save_cm_path="stage2_multiclass_confusion_matrix.png",
    )

    if isinstance(best_multiclass_model, XGBClassifier):
        best_multiclass_model.get_booster().save_model("failure_type_model.json")
        saved_model_path = "failure_type_model.json"
        print(f"Saved Stage 2 model: {saved_model_path}")
    else:
        joblib.dump(best_multiclass_model, "failure_type_model.pkl")
        saved_model_path = "failure_type_model.pkl"
        print(f"Saved Stage 2 model: {saved_model_path}")

    return {
        "stage": "multiclass",
        "best_model_name": best_multi_name,
        "metrics_plot": "stage2_multiclass_model_metrics.png",
        "confusion_matrix": "stage2_multiclass_confusion_matrix.png",
        "saved_model": saved_model_path,
    }


# --------------------
# MAIN
# --------------------
def main(mode="all"):
    results = {}

    if mode in ("binary", "all"):
        results["binary"] = run_stage1_binary()

    if mode in ("multiclass", "all"):
        results["multiclass"] = run_stage2_multiclass()

    print("\n====================")
    print("TRAINING FINISHED")
    print("====================")

    for key, value in results.items():
        print(f"\n[{key.upper()}]")
        print(f"Best model: {value['best_model_name']}")
        print(f"Metrics plot: {value['metrics_plot']}")
        print(f"Confusion matrix: {value['confusion_matrix']}")
        print(f"Saved model: {value['saved_model']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive maintenance models.")
    parser.add_argument(
        "--mode",
        choices=["binary", "multiclass", "all"],
        default="all",
        help="Training mode: binary, multiclass, or all",
    )

    args = parser.parse_args()
    main(mode=args.mode)