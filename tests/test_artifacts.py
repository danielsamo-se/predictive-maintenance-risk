import json
from pathlib import Path

def test_production_artifacts_exist():
    production_dir = Path("models/production")
    
    assert (production_dir / "model.joblib").exists(), "model.joblib not found"
    assert (production_dir / "metadata.json").exists(), "metadata.json not found"


def test_metadata_valid_json():
    production_dir = Path("models/production")
    metadata_path = production_dir / "metadata.json"
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    assert isinstance(metadata, dict), "metadata.json is not a valid dict"


def test_metadata_required_keys():
    production_dir = Path("models/production")
    metadata_path = production_dir / "metadata.json"
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    required_keys = [
        "model_type",
        "featureset_version",
        "horizon_n",
        "window_l",
        "feature_columns",
        "threshold",
        "bucket_cutoffs",
        "target_recall",
        "min_precision",
        "val_metrics",
    ]
    
    for key in required_keys:
        assert key in metadata, f"Missing required key: {key}"


def test_metadata_constraints():
    production_dir = Path("models/production")
    metadata_path = production_dir / "metadata.json"
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    assert metadata["model_type"] in {"logreg", "lightgbm"}, "model_type must be 'logreg' or 'lightgbm'"
    
    assert isinstance(metadata["feature_columns"], list), "feature_columns must be a list"
    assert len(metadata["feature_columns"]) > 0, "feature_columns must not be empty"
    assert all(isinstance(col, str) for col in metadata["feature_columns"]), "feature_columns must contain strings"
    
    threshold = metadata["threshold"]
    assert isinstance(threshold, (int, float)), "threshold must be numeric"
    assert 0.0 <= threshold <= 1.0, "threshold must be in [0, 1]"
    
    bucket_cutoffs = metadata["bucket_cutoffs"]
    assert isinstance(bucket_cutoffs, list), "bucket_cutoffs must be a list"
    assert len(bucket_cutoffs) == 2, "bucket_cutoffs must have length 2"
    assert all(isinstance(x, (int, float)) for x in bucket_cutoffs), "bucket_cutoffs must contain numbers"
    assert 0 <= bucket_cutoffs[0] < bucket_cutoffs[1] <= 1, "bucket_cutoffs must satisfy 0 <= a < b <= 1"


def test_metadata_model_filename():
    production_dir = Path("models/production")
    metadata_path = production_dir / "metadata.json"
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    if "model_filename" in metadata:
        assert metadata["model_filename"] == "model.joblib", "model_filename must be 'model.joblib'"
