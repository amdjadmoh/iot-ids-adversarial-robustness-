"""Detector definitions: Random Forest, XGBoost, and a small PyTorch MLP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from xgboost import XGBClassifier


def build_random_forest(config: dict[str, Any]) -> RandomForestClassifier:
    """Build a Random Forest using the project configuration."""
    params = config["models"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=int(params.get("random_state", config["seed"])),
    )


def build_xgboost(num_classes: int, config: dict[str, Any]) -> XGBClassifier:
    """Build an XGBoost classifier for binary or multiclass targets."""
    params = config["models"]["xgboost"]
    kwargs: dict[str, Any] = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "n_jobs": int(params.get("n_jobs", -1)),
        "random_state": int(params.get("random_state", config["seed"])),
        "tree_method": "hist",
        "verbosity": 0,
    }
    if num_classes == 2:
        kwargs["objective"] = "binary:logistic"
        kwargs["eval_metric"] = "logloss"
    else:
        kwargs["objective"] = "multi:softprob"
        kwargs["num_class"] = num_classes
        kwargs["eval_metric"] = "mlogloss"
    return XGBClassifier(**kwargs)


class MLP(nn.Module):
    """Small feed-forward detector for tabular flow features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(hidden_dim)
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_mlp(
    input_dim: int,
    num_classes: int,
    config: dict[str, Any],
) -> MLP:
    """Build the MLP architecture described by the configuration."""
    params = config["models"]["mlp"]
    return MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=params["hidden_dims"],
        dropout=float(params.get("dropout", 0.0)),
    )


def save_mlp(
    model: MLP,
    input_dim: int,
    num_classes: int,
    path: str | Path,
    config: dict[str, Any],
) -> None:
    """Save the MLP weights together with the architecture parameters."""
    params = config["models"]["mlp"]
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_dims": list(params["hidden_dims"]),
        "dropout": float(params.get("dropout", 0.0)),
    }
    torch.save(checkpoint, path)


def load_mlp(path: str | Path, device: torch.device) -> MLP:
    """Load an MLP saved by :func:`save_mlp`."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MLP(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        hidden_dims=checkpoint["hidden_dims"],
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
