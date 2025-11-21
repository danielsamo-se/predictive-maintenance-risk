import random


def split_engine_ids(
    engine_ids: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    if not engine_ids:
        raise ValueError("engine_ids cannot be empty")
    
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    rng = random.Random(seed)
    shuffled = engine_ids.copy()
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]
    
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Overlapping engine_ids between splits")
    
    if len(train_ids) + len(val_ids) + len(test_ids) != len(engine_ids):
        raise ValueError("Split lengths do not sum to total engine_ids count")
    
    if train_set | val_set | test_set != set(engine_ids):
        raise ValueError("Not all engine_ids are covered in splits")
    
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
