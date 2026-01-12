Expanding Engine Backtest Results

Configuration
- n_folds: 3
- test_engines_per_fold: 10
- train_min_engines: 40
- val_ratio_within_train: 0.2
- model_candidates: ['logreg', 'lightgbm']
- target_recall: 0.85
- min_precision: 0.3

Fold results

Fold 0:
- engines: train=32, val=8, test=10
- selected_model: lightgbm
- threshold: 0.9015
- val_pr_auc: 0.9781
  Test metrics:
  - pr_auc: 0.9886
  - roc_auc: 0.9973
  - precision: 0.9774
  - recall: 0.8667
  - f1: 0.9187
  - brier_score: 0.0166
  - confusion_matrix: tn=1287, fp=6, fn=40, tp=260
  Top false positives (max 5):
    - engine_id=47, cycle=214, y_proba=0.994, remaining=0, fp_kind=at_or_past_failure
    - engine_id=42, cycle=196, y_proba=0.981, remaining=0, fp_kind=at_or_past_failure
    - engine_id=44, cycle=192, y_proba=0.966, remaining=0, fp_kind=at_or_past_failure
    - engine_id=48, cycle=200, y_proba=0.966, remaining=31, fp_kind=premature_alarm
    - engine_id=46, cycle=225, y_proba=0.940, remaining=31, fp_kind=premature_alarm
  Top false negatives (max 5):
    - engine_id=41, cycle=188, y_proba=0.895, remaining=28
    - engine_id=50, cycle=175, y_proba=0.891, remaining=23
    - engine_id=42, cycle=169, y_proba=0.890, remaining=27
    - engine_id=45, cycle=139, y_proba=0.890, remaining=19
    - engine_id=43, cycle=177, y_proba=0.888, remaining=30

Fold 1:
- engines: train=40, val=10, test=10
- selected_model: lightgbm
- threshold: 0.9236
- val_pr_auc: 0.9873
  Test metrics:
  - pr_auc: 0.9584
  - roc_auc: 0.9907
  - precision: 0.9419
  - recall: 0.8100
  - f1: 0.8710
  - brier_score: 0.0309
  - confusion_matrix: tn=1228, fp=15, fn=57, tp=243
  Top false positives (max 5):
    - engine_id=56, cycle=242, y_proba=0.998, remaining=33, fp_kind=premature_alarm
    - engine_id=56, cycle=243, y_proba=0.998, remaining=32, fp_kind=premature_alarm
    - engine_id=56, cycle=244, y_proba=0.998, remaining=31, fp_kind=premature_alarm
    - engine_id=56, cycle=241, y_proba=0.997, remaining=34, fp_kind=premature_alarm
    - engine_id=52, cycle=182, y_proba=0.997, remaining=31, fp_kind=premature_alarm
  Top false negatives (max 5):
    - engine_id=54, cycle=232, y_proba=0.919, remaining=25
    - engine_id=59, cycle=205, y_proba=0.910, remaining=26
    - engine_id=51, cycle=189, y_proba=0.905, remaining=24
    - engine_id=56, cycle=273, y_proba=0.904, remaining=2
    - engine_id=60, cycle=149, y_proba=0.903, remaining=23

Fold 2:
- engines: train=48, val=12, test=10
- selected_model: lightgbm
- threshold: 0.9250
- val_pr_auc: 0.9705
  Test metrics:
  - pr_auc: 0.9353
  - roc_auc: 0.9871
  - precision: 0.8653
  - recall: 0.8567
  - f1: 0.8610
  - brier_score: 0.0451
  - confusion_matrix: tn=1358, fp=40, fn=43, tp=257
  Top false positives (max 5):
    - engine_id=69, cycle=331, y_proba=0.998, remaining=31, fp_kind=premature_alarm
    - engine_id=69, cycle=329, y_proba=0.997, remaining=33, fp_kind=premature_alarm
    - engine_id=69, cycle=330, y_proba=0.997, remaining=32, fp_kind=premature_alarm
    - engine_id=69, cycle=328, y_proba=0.997, remaining=34, fp_kind=premature_alarm
    - engine_id=69, cycle=327, y_proba=0.996, remaining=35, fp_kind=premature_alarm
  Top false negatives (max 5):
    - engine_id=61, cycle=184, y_proba=0.925, remaining=1
    - engine_id=66, cycle=197, y_proba=0.924, remaining=5
    - engine_id=70, cycle=136, y_proba=0.918, remaining=1
    - engine_id=62, cycle=157, y_proba=0.912, remaining=23
    - engine_id=61, cycle=162, y_proba=0.910, remaining=23
