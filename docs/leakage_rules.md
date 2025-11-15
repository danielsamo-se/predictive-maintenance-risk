# Leakage Rules (Minimal)

- Past-only: inputs/features at time t use data <= t only.
- Scaling/normalization: fit on TRAIN only; apply to val/test/serving.
- No random split across time; use time-/engine-based splits.

