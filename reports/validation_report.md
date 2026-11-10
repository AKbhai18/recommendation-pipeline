# Data Validation Report

_Generated: 2026-11-10 13:10:04_

## Dimension tables (schema validation)

| Table | Rows | Schema OK | Issues |
|-------|------|-----------|--------|
| users | 1000 | yes | 0 |
| items | 500 | yes | 0 |

## Interactions (row-level checks)

| Metric | Count |
|--------|-------|
| Total ingested rows | 61800 |
| Clean rows (passed) | 53084 |
| Error rows (quarantined) | 8716 (14.10%) |
| Missing / non-numeric rating | 1809 |
| Rating out of range | 1854 |
| Invalid timestamp | 1848 |
| Unknown user_id (FK) | 0 |
| Unknown item_id (FK) | 1855 |
| Duplicate interactions | 1595 |

Invalid rows are quarantined in `data/errors/invalid_interactions.csv` with a per-row `error_reason`.
