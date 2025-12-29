import importlib.util
from pathlib import Path

import torch

from pmrisk.models.model_builder import build_sequence_model
from pmrisk.models.model_versions import save_sequence_model_production


def _load_run_benchmark():
    bench_path = Path(__file__).resolve().parents[1] / "scripts" / "bench_inference.py"
    spec = importlib.util.spec_from_file_location("bench_inference", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load bench_inference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_benchmark


def test_bench_inference_smoke(tmp_path: Path) -> None:
    hparams = {"model_type": "cnn", "n_features": 3, "window_l": 10, "hidden_channels": 8}
    model = build_sequence_model(hparams)

    save_sequence_model_production(
        model_name="seq",
        version="v1",
        state_dict=model.state_dict(),
        metadata={"hparams": hparams},
        root=tmp_path,
        set_active=True,
    )

    x = torch.randn(2, 10, 3)

    run_benchmark = _load_run_benchmark()
    results = run_benchmark(model, x, iters=5, warmup=1)

    assert isinstance(results["avg_ms_per_iter"], float)
    assert isinstance(results["avg_ms_per_sample"], float)
    assert isinstance(results["total_time_s"], float)

    assert results["avg_ms_per_iter"] > 0.0
    assert results["avg_ms_per_sample"] > 0.0
    assert results["total_time_s"] > 0.0
