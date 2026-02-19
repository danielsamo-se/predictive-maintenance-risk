Model Comparison

Split: engine-based 0.7/0.15/0.15
Threshold policy: recall >= 0.85, precision >= 0.3

Logistic Regression
- threshold: 0.8062
  Train:
  - PR-AUC: 0.9176
  - ROC-AUC: 0.9923
  - Precision: 0.9491
  - Recall: 0.9243
  - F1: 0.9366
  - Brier: 0.0229
  - CM: tn=9000, fp=104, fn=159, tp=1941
  Val:
  - PR-AUC: 0.8728
  - ROC-AUC: 0.9839
  - Precision: 0.8825
  - Recall: 0.8511
  - F1: 0.8665
  - Brier: 0.0420
  - CM: tn=1824, fp=51, fn=67, tp=383
  Test:
  - PR-AUC: 0.9055
  - ROC-AUC: 0.9888
  - Precision: 0.9234
  - Recall: 0.9111
  - F1: 0.9172
  - Brier: 0.0327
  - CM: tn=1718, fp=34, fn=40, tp=410

LightGBM
- threshold: 0.8974
  Train:
  - PR-AUC: 1.0000
  - ROC-AUC: 1.0000
  - Precision: 1.0000
  - Recall: 0.9990
  - F1: 0.9995
  - Brier: 0.0014
  - CM: tn=9104, fp=0, fn=2, tp=2098
  Val:
  - PR-AUC: 0.9738
  - ROC-AUC: 0.9932
  - Precision: 0.9319
  - Recall: 0.8511
  - F1: 0.8897
  - Brier: 0.0302
  - CM: tn=1847, fp=28, fn=67, tp=383
  Test:
  - PR-AUC: 0.9840
  - ROC-AUC: 0.9954
  - Precision: 0.9659
  - Recall: 0.8822
  - F1: 0.9222
  - Brier: 0.0263
  - CM: tn=1738, fp=14, fn=53, tp=397

1D-CNN (PyTorch)
- threshold: 0.0670
  Train:
  - PR-AUC: 0.8199
  - ROC-AUC: 0.9385
  - Precision: 0.5012
  - Recall: 0.9747
  - F1: 0.6620
  - Brier: 0.0912
  - CM: tn=3639, fp=2036, fn=53, tp=2046
  Val:
  - PR-AUC: 0.7522
  - ROC-AUC: 0.8969
  - Precision: 0.6218
  - Recall: 0.8511
  - F1: 0.7186
  - Brier: 0.1455
  - CM: tn=907, fp=233, fn=67, tp=383
  Test:
  - PR-AUC: 0.8593
  - ROC-AUC: 0.9557
  - Precision: 0.5656
  - Recall: 0.9867
  - F1: 0.7190
  - Brier: 0.0948
  - CM: tn=676, fp=341, fn=6, tp=444

GRU (PyTorch)
- threshold: 0.7581
  Train:
  - PR-AUC: 0.9792
  - ROC-AUC: 0.9928
  - Precision: 0.9516
  - Recall: 0.8433
  - F1: 0.8942
  - Brier: 0.0293
  - CM: tn=5585, fp=90, fn=329, tp=1770
  Val:
  - PR-AUC: 0.9713
  - ROC-AUC: 0.9887
  - Precision: 0.9364
  - Recall: 0.8511
  - F1: 0.8917
  - Brier: 0.0386
  - CM: tn=1114, fp=26, fn=67, tp=383
  Test:
  - PR-AUC: 0.9736
  - ROC-AUC: 0.9894
  - Precision: 0.9420
  - Recall: 0.7933
  - F1: 0.8613
  - Brier: 0.0398
  - CM: tn=995, fp=22, fn=93, tp=357

Overfitting check (train vs val PR-AUC)

- Logistic Regression: train=0.9176, val=0.8728, gap=0.0448
- LightGBM: train=1.0000, val=0.9738, gap=0.0262
- 1D-CNN (PyTorch): train=0.8199, val=0.7522, gap=0.0676
- GRU (PyTorch): train=0.9792, val=0.9713, gap=0.0079

Selected model: LightGBM (highest val PR-AUC)

IsolationForest (unsupervised baseline)

  Val:
  - top_1pct_precision: 0.5417
  - top_1pct_lift: 2.7986
  - top_1pct_capture: 0.0289
  - top_5pct_precision: 0.5983
  - top_5pct_lift: 3.0912
  - top_5pct_capture: 0.1556
  Test:
  - top_1pct_precision: 0.7391
  - top_1pct_lift: 3.6168
  - top_1pct_capture: 0.0378
  - top_5pct_precision: 0.8018
  - top_5pct_lift: 3.9235
  - top_5pct_capture: 0.1978
