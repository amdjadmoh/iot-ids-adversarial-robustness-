from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models import (
    build_mlp,
    build_random_forest,
    build_xgboost,
    load_mlp,
    save_mlp,
)


def _config() -> dict[str, Any]:
    return {
        "seed": 11,
        "models": {
            "random_forest": {
                "n_estimators": 5,
                "max_depth": None,
                "min_samples_leaf": 1,
                "n_jobs": 1,
                "random_state": 11,
            },
            "xgboost": {
                "n_estimators": 5,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "n_jobs": 1,
                "random_state": 11,
            },
            "mlp": {
                "hidden_dims": [8, 4],
                "dropout": 0.0,
                "batch_size": 8,
                "epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            },
        },
    }


def test_build_random_forest_uses_config() -> None:
    model = build_random_forest(_config())

    assert model.n_estimators == 5
    assert model.random_state == 11


def test_build_xgboost_uses_binary_objective() -> None:
    model = build_xgboost(2, _config())

    assert model.get_xgb_params()["objective"] == "binary:logistic"


def test_build_xgboost_uses_multiclass_objective() -> None:
    model = build_xgboost(4, _config())

    params = model.get_xgb_params()
    assert params["objective"] == "multi:softprob"
    assert int(params["num_class"]) == 4


def test_mlp_forward_shape() -> None:
    model = build_mlp(input_dim=6, num_classes=3, config=_config())

    output = model(torch.randn(5, 6))

    assert output.shape == (5, 3)


def test_mlp_save_and_load_round_trip(tmp_path: Path) -> None:
    config = _config()
    model = build_mlp(input_dim=6, num_classes=3, config=config)
    path = tmp_path / "model.pt"
    save_mlp(model, input_dim=6, num_classes=3, path=path, config=config)

    loaded = load_mlp(path, torch.device("cpu"))
    features = torch.randn(4, 6)
    model.eval()
    with torch.inference_mode():
        expected = model(features)
        actual = loaded(features)

    assert torch.allclose(expected, actual)
