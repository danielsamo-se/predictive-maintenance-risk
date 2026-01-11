Anomaly Baseline (IsolationForest)

Config:
- contamination: 0.02
- seed: 42
- horizon_n: 30
- window_l: 50
- feature_set_version: v1
- fit: train only
- scoring: val/test

Notes:
- top_1pct_precision = positive rate within the top 1% anomaly scores
- top_1pct_lift = top_1pct_precision / base_rate
- top_1pct_capture = fraction of all positives captured in the top 1%

Metrics

Val:
- n: 2325
- positives: 450
- base_rate: 0.1935
- top_1pct_m: 24
- top_1pct_precision: 0.5417
- top_1pct_lift: 2.7986
- top_1pct_capture: 0.0289
- top_5pct_m: 117
- top_5pct_precision: 0.7949
- top_5pct_lift: 4.1068
- top_5pct_capture: 0.2067

Top 10 anomalies (VAL):
- 1. engine_id=94, cycle=258, label=0, remaining=0, anomaly_score=0.588608
- 2. engine_id=94, cycle=257, label=1, remaining=1, anomaly_score=0.586665
- 3. engine_id=54, cycle=255, label=1, remaining=2, anomaly_score=0.569921
- 4. engine_id=65, cycle=153, label=0, remaining=0, anomaly_score=0.564354
- 5. engine_id=1, cycle=192, label=0, remaining=0, anomaly_score=0.564090
- 6. engine_id=90, cycle=154, label=0, remaining=0, anomaly_score=0.563654
- 7. engine_id=72, cycle=212, label=1, remaining=1, anomaly_score=0.562618
- 8. engine_id=94, cycle=256, label=1, remaining=2, anomaly_score=0.562285
- 9. engine_id=65, cycle=149, label=1, remaining=4, anomaly_score=0.557775
- 10. engine_id=78, cycle=231, label=0, remaining=0, anomaly_score=0.557726

Test:
- n: 2202
- positives: 450
- base_rate: 0.2044
- top_1pct_m: 23
- top_1pct_precision: 0.7391
- top_1pct_lift: 3.6168
- top_1pct_capture: 0.0378
- top_5pct_m: 111
- top_5pct_precision: 0.8108
- top_5pct_lift: 3.9676
- top_5pct_capture: 0.2000

Top 10 anomalies (TEST):
- 1. engine_id=18, cycle=195, label=0, remaining=0, anomaly_score=0.646834
- 2. engine_id=18, cycle=194, label=1, remaining=1, anomaly_score=0.625426
- 3. engine_id=18, cycle=193, label=1, remaining=2, anomaly_score=0.616105
- 4. engine_id=4, cycle=189, label=0, remaining=0, anomaly_score=0.613545
- 5. engine_id=82, cycle=214, label=0, remaining=0, anomaly_score=0.613036
- 6. engine_id=18, cycle=192, label=1, remaining=3, anomaly_score=0.600098
- 7. engine_id=82, cycle=210, label=1, remaining=4, anomaly_score=0.598942
- 8. engine_id=4, cycle=188, label=1, remaining=1, anomaly_score=0.598359
- 9. engine_id=82, cycle=211, label=1, remaining=3, anomaly_score=0.592123
- 10. engine_id=55, cycle=191, label=1, remaining=2, anomaly_score=0.589971

