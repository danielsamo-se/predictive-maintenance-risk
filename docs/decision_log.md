Decision Log

- Defaults: N=30, L=50 (from config).
- Canonical data stored in `data/processed/` (raw optional in `data/raw/`).
- API will take a window as input (not engine_id-only).
- No random split for time series to avoid leakage.
