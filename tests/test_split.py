from pmrisk.split.splitter import split_engine_ids


def test_no_overlaps():
    engine_ids = list(range(1, 101))
    result = split_engine_ids(engine_ids, 0.7, 0.15, 0.15, 42)
    
    train_set = set(result["train"])
    val_set = set(result["val"])
    test_set = set(result["test"])
    
    assert not (train_set & val_set)
    assert not (train_set & test_set)
    assert not (val_set & test_set)
    
    assert len(result["train"]) + len(result["val"]) + len(result["test"]) == len(engine_ids)
    assert train_set | val_set | test_set == set(engine_ids)


def test_deterministic():
    engine_ids = list(range(1, 101))
    result1 = split_engine_ids(engine_ids, 0.7, 0.15, 0.15, 42)
    result2 = split_engine_ids(engine_ids, 0.7, 0.15, 0.15, 42)
    
    assert result1["train"] == result2["train"]
    assert result1["val"] == result2["val"]
    assert result1["test"] == result2["test"]


def test_different_seeds():
    engine_ids = list(range(1, 101))
    result1 = split_engine_ids(engine_ids, 0.7, 0.15, 0.15, 42)
    result2 = split_engine_ids(engine_ids, 0.7, 0.15, 0.15, 123)
    
    combined1 = result1["train"] + result1["val"] + result1["test"]
    combined2 = result2["train"] + result2["val"] + result2["test"]
    
    assert combined1 != combined2
