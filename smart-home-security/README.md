# HomeShield Lab: Smart Home Cybersecurity Capstone (AI/ML + Streamlit)

A single local Streamlit app that simulates smart-home IoT network flows, trains two scikit-learn models, scores traffic,
stores results in SQLite, and presents everything in a dark glassmorphism dashboard.

> **All data is synthetic.** Flows are produced by `smarthome/data.py` from random distributions that *I wrote to resemble*
> each attack type. Reported metrics are computed on a held-out 25% split of that same simulator's output. They show the
> pipeline works end to end; they are **not** evidence the models would detect attacks on real networks. Near-perfect
> scores are expected because simulated classes are cleanly separable (20% of attacks use "stealth" parameters
> that overlap normal traffic, but that is still far easier than real traffic). No real-world validation has been done.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python init_project.py            # generates data, trains + saves models, fills SQLite (≈10 s)
streamlit run app.py
```
`init_project.py --rows 10000 --seed 7` changes size/seed. If you skip it, the app offers a one-click initialise button.
Models are loaded from `models/` on each launch; they are only retrained when you ask (Model Performance → Retrain).

## Tests
```bash
python -m pytest -q          # or: python -m unittest discover -s tests
```

## What's inside
| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (7 pages) |
| `init_project.py` | One-shot data → train → score → DB pipeline |
| `smarthome/data.py` | Synthetic generator (normal, port_scan, brute_force, dos, suspicious_outbound), upload validation, feature prep |
| `smarthome/ml.py` | Random Forest classifier, Isolation Forest, risk scoring, metric computation, model persistence |
| `smarthome/db.py` | SQLite tables: `batches`, `predictions`, `events` |
| `smarthome/pipeline.py` | End-to-end initialisation |
| `tests/test_core.py` | Reproducibility, schema, training/persistence, scoring ranges, DB round trip |

Generated at init (git-ignored): `data/synthetic_iot_traffic.csv`, `data/smart_home_security.db`, `models/*.joblib`, `models/metrics.json`.

## Dashboard pages
Overview (KPIs, class mix, risk over time) · Traffic Analysis (filters, volume, scatter, ports) · Device Monitoring (per-device status, heatmap, timeline) ·
Threat Detection (CSV upload, single-flow simulator, fresh synthetic batches, save to DB) · Security Events (filter, change status, export) ·
Model Performance (metrics, confusion matrix, feature importance, retrain) · Data & Export.

## Models and risk score
- **Random Forest** (200 trees, balanced class weights) classifies a flow into 5 classes.
- **Isolation Forest** (trained on *normal* training rows only) yields an anomaly score; its precision/recall are measured as "flagged vs. any attack".
- **Risk score (0-100)** is a transparent heuristic, not a calibrated probability:
  `100 × (0.5·P(attack) + 0.3·anomaly_score + 0.2·severity·P(attack))`. Levels: Low <30, Medium <55, High <75, Critical ≥75.
  Flows ≥55 are stored as security events (status: open / investigating / resolved / false_positive).
- Features: duration, packets/s, bytes sent/received, unique dst ports, failed logins, SYN ratio, connections/min, dst port, is_external, protocol. Heavy-tailed ones are log1p-transformed.

## CSV upload format
Required columns: `duration, packets_per_sec, bytes_sent, bytes_received, unique_dst_ports, failed_logins, syn_ratio, conn_per_min, dst_port, is_external, protocol` (TCP/UDP/ICMP).
Optional: `timestamp, device_id, device_type, src_ip, dst_ip, label`. A template download is on the Threat Detection page.

## Limitations
- Synthetic data only; the model has learned the simulator, not real attacker behaviour.
- Flow-level features only (no payloads, no sequence modelling). Risk weights are hand-chosen.
- Model files are scikit-learn-version specific; re-run `init_project.py` after upgrading scikit-learn.
- Local, single-user SQLite app; no authentication.
