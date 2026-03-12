"""inference_models_robust.py

Robust inference layer for the PLC AI service.

Supports:
  - TorchScript: *.ts.pt / *.pt (recommended for "PLC-like" deployment on an edge PC)
  - Torch payload dict: joblib/pickle file containing {torch_payload, scaler}
  - sklearn/joblib pipelines: *.pkl
  - XGBoost booster: *.json

Also supports external scaler files (*.npz or *.json) and a schema.json that
describes the feature order.

This file fixes the main failure mode in your current setup: your training script
saves TorchScript (e.g. stage1_torch_model.ts.pt) while the service was trying to
load a non-existent *.pkl torch payload. fileciteturn0file2 fileciteturn0file1
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import joblib
import xgboost as xgb

import torch
import torch.nn as nn


# Default feature order (kept for backward compatibility)
FEATURES: List[str] = [
    "Air_temperature__K_",
    "Process_temperature__K_",
    "Rotational_speed__rpm_",
    "Torque__Nm_",
    "Tool_wear__min_",
    "Type_L",
    "Type_M",
]


FAILURE_TYPE_CODES = {0: "TWF", 1: "HDF", 2: "PWF", 3: "OSF", 4: "RNF"}
FAILURE_TYPE_NAMES = {
    0: "Tool Wear Failure (TWF)",
    1: "Heat Dissipation Failure (HDF)",
    2: "Power Failure (PWF)",
    3: "Overstrain Failure (OSF)",
    4: "Random Failure (RNF)",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_scaler(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    """Load a scaler from .npz or .json.

    Expected keys: mean_ + scale_ (np arrays).
    """
    if not path:
        return None
    p = _abs(path)
    if not os.path.exists(p):
        return None

    ext = os.path.splitext(p)[1].lower()
    if ext == ".npz":
        data = np.load(p)
        return {
            "mean_": np.asarray(data["mean_"], dtype=np.float32),
            "scale_": np.asarray(data["scale_"], dtype=np.float32),
        }
    if ext == ".json":
        d = _load_json(p)
        # tolerate either mean/scale or mean_/scale_
        mean = d.get("mean_") if "mean_" in d else d.get("mean")
        scale = d.get("scale_") if "scale_" in d else d.get("scale")
        if mean is None or scale is None:
            raise ValueError(f"Scaler JSON missing mean/scale keys: {p}")
        return {
            "mean_": np.asarray(mean, dtype=np.float32),
            "scale_": np.asarray(scale, dtype=np.float32),
        }

    raise ValueError(f"Unsupported scaler extension: {ext} (use .npz or .json)")


def _load_schema(path: Optional[str]) -> Optional[List[str]]:
    """Load schema.json containing {"feature_names": [...]} (preferred).

    If missing, fallback to FEATURES constant.
    """
    if not path:
        return None
    p = _abs(path)
    if not os.path.exists(p):
        return None
    d = _load_json(p)
    feats = d.get("feature_names") or d.get("features")
    if not isinstance(feats, list) or not feats:
        raise ValueError(f"Schema JSON missing feature_names list: {p}")
    return [str(x) for x in feats]


class TorchCNNLSTMNet(nn.Module):
    def __init__(self, input_dim, out_dim, conv_channels, kernel_size, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        padding = max(0, kernel_size // 2)
        self.conv = nn.Conv1d(1, conv_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if (dropout and dropout > 0 and lstm_layers > 1) else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def build_torch_model_from_payload(torch_payload: dict) -> nn.Module:
    params = torch_payload["params"]
    input_dim = int(torch_payload["input_dim_"])
    num_classes = int(torch_payload["num_classes_"])
    model_class = torch_payload.get("model_class")

    out_dim = 1 if num_classes == 2 else num_classes

    if model_class == "TorchMLPClassifier":
        hidden_dims = params["hidden_dims"]
        dropout = float(params["dropout"])
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, out_dim))
        model = nn.Sequential(*layers)
    elif model_class == "TorchCNNLSTMClassifier":
        model = TorchCNNLSTMNet(
            input_dim=input_dim,
            out_dim=out_dim,
            conv_channels=int(params["conv_channels"]),
            kernel_size=int(params["kernel_size"]),
            lstm_hidden=int(params["lstm_hidden"]),
            lstm_layers=int(params["lstm_layers"]),
            dropout=float(params["dropout"]),
        )
    else:
        raise ValueError(f"Unknown torch model_class: {model_class}")

    model.load_state_dict(torch_payload["state_dict"])
    model.eval()
    return model


@dataclass
class LoadedModel:
    kind: str
    obj: Any
    scaler: Optional[Dict[str, np.ndarray]] = None
    num_classes: Optional[int] = None

    def apply_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        mean_ = np.asarray(self.scaler["mean_"], dtype=np.float32)
        scale_ = np.asarray(self.scaler["scale_"], dtype=np.float32)
        return (X - mean_) / scale_


def load_model(model_path: str, scaler_path: Optional[str] = None) -> LoadedModel:
    """Load a model and (optionally) an external scaler."""
    mp = _abs(model_path)
    if not os.path.exists(mp):
        raise FileNotFoundError(f"Model file not found: {mp}")

    ext = os.path.splitext(mp)[1].lower()
    scaler = _load_scaler(scaler_path)

    # TorchScript
    if ext in {".pt", ".ts"} or mp.endswith(".ts.pt"):
        m = torch.jit.load(mp, map_location="cpu")
        m.eval()
        # TorchScript itself does not expose num_classes reliably; infer from output at runtime
        return LoadedModel(kind="torchscript", obj=m, scaler=scaler)

    # XGBoost Booster
    if ext == ".json":
        booster = xgb.Booster()
        booster.load_model(mp)
        return LoadedModel(kind="xgb_booster", obj=booster, scaler=scaler)

    # joblib/pickle
    if ext == ".pkl":
        obj = joblib.load(mp)

        # torch payload dict packed into pkl
        if isinstance(obj, dict) and "torch_payload" in obj:
            torch_payload = obj["torch_payload"]
            model = build_torch_model_from_payload(torch_payload)
            packed_scaler = obj.get("scaler")
            # external scaler overrides packed scaler
            use_scaler = scaler if scaler is not None else packed_scaler
            return LoadedModel(
                kind="torch_payload",
                obj=model,
                scaler=use_scaler,
                num_classes=int(torch_payload.get("num_classes_", 2)),
            )

        # normal sklearn object
        return LoadedModel(kind="sklearn", obj=obj, scaler=scaler)

    raise ValueError(f"Unsupported model extension: {ext} (expected .pt/.ts.pt, .json, or .pkl)")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _torchscript_has_internal_scaler(tsm) -> bool:
    """Detect if TorchScript model includes mean/scale buffers (ScaledTorchModel)."""
    try:
        names = {name for name, _ in tsm.named_buffers()}
        return ("mean" in names) and ("scale" in names)
    except Exception:
        return False

class Predictor:
    """Two-stage predictor with robust model + schema/scaler handling."""

    def __init__(
        self,
        stage1_model: str,
        stage1_scaler: Optional[str] = None,
        stage1_schema: Optional[str] = None,
        stage2_model: Optional[str] = None,
        stage2_scaler: Optional[str] = None,
        stage2_schema: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.threshold = float(threshold)
        self.warnings: List[str] = []

        # schema
        s1_feats = _load_schema(stage1_schema) or FEATURES
        self.features = s1_feats

        # stage-1 model must exist
        self.stage1: Optional[LoadedModel] = None
        self.stage2: Optional[LoadedModel] = None

        try:
            self.stage1 = load_model(stage1_model, scaler_path=stage1_scaler)
        except Exception as e:
            self.warnings.append(f"Stage-1 load failed: {e}")

        # stage-2 is optional
        if stage2_model:
            try:
                _ = _load_schema(stage2_schema)  # not used here because stage-2 uses same features
                self.stage2 = load_model(stage2_model, scaler_path=stage2_scaler)
            except Exception as e:
                self.warnings.append(f"Stage-2 load failed: {e}")

        self.ready = self.stage1 is not None

    @property
    def stage1_loaded(self) -> bool:
        return self.stage1 is not None

    @property
    def stage2_loaded(self) -> bool:
        return self.stage2 is not None

    @property
    def stage1_kind(self) -> Optional[str]:
        return None if self.stage1 is None else self.stage1.kind

    @property
    def stage2_kind(self) -> Optional[str]:
        return None if self.stage2 is None else self.stage2.kind

    def _vector(self, sample: Dict[str, Any]) -> np.ndarray:
        missing = [f for f in self.features if f not in sample]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return np.array([[float(sample[f]) for f in self.features]], dtype=np.float32)

    def _stage1_prob(self, X: np.ndarray) -> float:
        assert self.stage1 is not None
        m = self.stage1

        if m.kind == "torchscript":
            # If the TorchScript already contains scaling, do NOT apply external scaler.
            X_in = X.astype(np.float32)
            if not _torchscript_has_internal_scaler(m.obj):
                X_in = m.apply_scaler(X_in).astype(np.float32)

            xt = torch.tensor(X_in, dtype=torch.float32)
            with torch.no_grad():
                logits = m.obj(xt).cpu().numpy().reshape(-1)
            return float(_sigmoid(logits)[0])

        if m.kind == "torch_payload":
            # torch_payload never includes sklearn scaler unless you pack it;
            # so we keep external scaling here.
            Xs = m.apply_scaler(X).astype(np.float32)
            xt = torch.tensor(Xs, dtype=torch.float32)
            with torch.no_grad():
                logits = m.obj(xt).cpu().numpy().reshape(-1)
            return float(_sigmoid(logits)[0])

        if m.kind == "xgb_booster":
            return float(m.obj.predict(xgb.DMatrix(X))[0])

        if m.kind == "sklearn":
            # pipeline may contain scaler, so don't apply external scaler unless you know you need it
            return float(m.obj.predict_proba(X)[0, 1])

        raise ValueError(f"Unsupported stage-1 kind: {m.kind}")

    def _stage2_class(self, X: np.ndarray) -> Optional[int]:
        if self.stage2 is None:
            return None
        m = self.stage2

        if m.kind == "torchscript":
            X_in = X.astype(np.float32)
            if not _torchscript_has_internal_scaler(m.obj):
                X_in = m.apply_scaler(X_in).astype(np.float32)
            xt = torch.tensor(X_in, dtype=torch.float32)
            with torch.no_grad():
                logits = m.obj(xt).cpu().numpy().reshape(-1)
            return int(np.argmax(logits))

        if m.kind == "torch_payload":
            Xs = m.apply_scaler(X).astype(np.float32)
            xt = torch.tensor(Xs, dtype=torch.float32)
            with torch.no_grad():
                logits = m.obj(xt).cpu().numpy().reshape(-1)
            return int(np.argmax(logits))

        if m.kind == "xgb_booster":
            preds = m.obj.predict(xgb.DMatrix(X))
            return int(np.argmax(preds))

        if m.kind == "sklearn":
            return int(m.obj.predict(X)[0])

        raise ValueError(f"Unsupported stage-2 kind: {m.kind}")

    def predict_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.stage1 is None:
            raise RuntimeError("Stage-1 model is not loaded")

        X = self._vector(sample)
        failure_prob = self._stage1_prob(X)
        failure = int(failure_prob >= self.threshold)

        result: Dict[str, Any] = {
            "machine_failure": failure,
            "failure_probability": float(failure_prob),
            "failure_type": None,
            "failure_type_code": None,
            "failure_type_name": None,
        }

        if failure == 1:
            ft = self._stage2_class(X)
            if ft is not None:
                result["failure_type"] = int(ft)
                result["failure_type_code"] = FAILURE_TYPE_CODES.get(int(ft), "UNKNOWN")
                result["failure_type_name"] = FAILURE_TYPE_NAMES.get(int(ft), "Unknown")

        return result
