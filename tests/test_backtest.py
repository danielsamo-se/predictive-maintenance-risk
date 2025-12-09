from pmrisk.eval.backtest import make_expanding_engine_folds, split_train_val_engines


def test_make_expanding_engine_folds_deterministic():
    engine_ids = [5, 1, 4, 2, 3, 6, 7, 8, 9, 10, 11, 12]
    n_folds = 3
    train_min_engines = 4
    test_engines_per_fold = 2

    folds = make_expanding_engine_folds(engine_ids, n_folds, train_min_engines, test_engines_per_fold)

    assert len(folds) == 3
    assert [f["fold"] for f in folds] == [0, 1, 2]

    assert folds[0]["train_engines"] == [1, 2, 3, 4]
    assert folds[0]["test_engines"] == [5, 6]

    assert folds[1]["train_engines"] == [1, 2, 3, 4, 5, 6]
    assert folds[1]["test_engines"] == [7, 8]

    assert folds[2]["train_engines"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert folds[2]["test_engines"] == [9, 10]

    prev_train_len = None
    all_test_engines = set()

    for fold in folds:
        train_set = set(fold["train_engines"])
        test_set = set(fold["test_engines"])

        assert train_set & test_set == set()
        assert len(fold["test_engines"]) == test_engines_per_fold

        if prev_train_len is not None:
            assert len(fold["train_engines"]) == prev_train_len + test_engines_per_fold
        prev_train_len = len(fold["train_engines"])

        assert all_test_engines & test_set == set()
        all_test_engines |= test_set


def test_make_expanding_engine_folds_stop_condition():
    engine_ids = list(range(1, 8))
    train_min_engines = 4
    test_engines_per_fold = 2
    n_folds = 10

    folds = make_expanding_engine_folds(engine_ids, n_folds, train_min_engines, test_engines_per_fold)

    assert len(folds) == 1
    assert folds[0]["train_engines"] == [1, 2, 3, 4]
    assert folds[0]["test_engines"] == [5, 6]


def test_split_train_val_engines_normal():
    train_engines = list(range(1, 11))
    val_ratio = 0.2

    train, val = split_train_val_engines(train_engines, val_ratio)

    assert val == [9, 10]
    assert train == list(range(1, 9))


def test_split_train_val_engines_minimum():
    train_engines = [1, 2]
    val_ratio = 0.2

    train, val = split_train_val_engines(train_engines, val_ratio)

    assert len(val) >= 1
    assert len(train) >= 1
    assert len(val) + len(train) == 2
    assert set(train) | set(val) == {1, 2}
    assert set(train) & set(val) == set()


def test_split_train_val_engines_single_engine():
    train_engines = [1]
    val_ratio = 0.2

    train, val = split_train_val_engines(train_engines, val_ratio)

    assert val == []
    assert train == [1]
