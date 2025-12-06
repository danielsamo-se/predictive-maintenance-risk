Contracts (Minimal)

Processed data (canonical table)

Location: `data/processed/`

Schema (FD001-style):

- engine_id (int)
- cycle (int)
- op_setting_1..3 (float)
- sensor_1..21 (float)

Invariants:

- sorted by (engine_id, cycle)
- no duplicates on (engine_id, cycle)

Model location

Current model will live in: `models/production/`

API (later)

POST `/predict`

- input: `window` with exactly L rows (latest history)
- output: `risk_score` in [0, 1]
