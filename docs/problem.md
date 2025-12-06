Problem

Goal: Predict failure risk in the horizon (t, t+N] from sensor time series.

Defaults (configurable):

- N (horizon) = 30
- L (window length) = 50

Label:

- y(t) = 1 if a failure happens in (t, t+N]
- else y(t) = 0

Main metric:

- PR-AUC
