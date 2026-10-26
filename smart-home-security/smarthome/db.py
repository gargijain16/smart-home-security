"""SQLite persistence for scored predictions and security events."""
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config

PRED_COLS = ["timestamp", "device_id", "device_type", "src_ip", "dst_ip", "dst_port", "protocol", "duration",
             "packets_per_sec", "bytes_sent", "bytes_received", "unique_dst_ports", "failed_logins", "syn_ratio",
             "conn_per_min", "is_external", "label", "predicted", "confidence", "p_attack", "anomaly_score",
             "is_anomaly", "risk_score", "risk_level"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, source TEXT, n_rows INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, timestamp TEXT, device_id TEXT, device_type TEXT,
  src_ip TEXT, dst_ip TEXT, dst_port INTEGER, protocol TEXT, duration REAL, packets_per_sec REAL,
  bytes_sent REAL, bytes_received REAL, unique_dst_ports INTEGER, failed_logins INTEGER, syn_ratio REAL,
  conn_per_min REAL, is_external INTEGER, label TEXT, predicted TEXT, confidence REAL, p_attack REAL,
  anomaly_score REAL, is_anomaly INTEGER, risk_score REAL, risk_level TEXT);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, timestamp TEXT, device_id TEXT, src_ip TEXT, dst_ip TEXT,
  dst_port INTEGER, attack_type TEXT, confidence REAL, anomaly_score REAL, risk_score REAL, risk_level TEXT,
  status TEXT DEFAULT 'open', created_at TEXT);
CREATE INDEX IF NOT EXISTS ix_pred_batch ON predictions(batch_id);
CREATE INDEX IF NOT EXISTS ix_evt_batch ON events(batch_id);
"""
STATUSES = ["open", "investigating", "resolved", "false_positive"]


def connect(path=None):
    con = sqlite3.connect(str(path or config.DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def init_db(path=None, reset=False):
    Path(path or config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(path)) as con:
        if reset:
            con.executescript("DROP TABLE IF EXISTS predictions; DROP TABLE IF EXISTS events; DROP TABLE IF EXISTS batches;")
        con.executescript(SCHEMA)
        con.commit()


def save_scored(df, source, path=None, threshold=config.EVENT_THRESHOLD):
    """Persist every scored row + a security event for each row at/above the risk threshold. Returns (batch_id, n_events)."""
    init_db(path)
    batch_id = uuid.uuid4().hex[:8]
    now = datetime.now().isoformat(timespec="seconds")
    d = df.copy()
    if "label" not in d:
        d["label"] = None
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    d = d[PRED_COLS].astype(object).where(d[PRED_COLS].notna(), None)
    ev = d[d["risk_score"] >= threshold]
    with closing(connect(path)) as con:
        con.execute("INSERT INTO batches VALUES (?,?,?,?)", (batch_id, source, len(d), now))
        con.executemany(f"INSERT INTO predictions (batch_id,{','.join(PRED_COLS)}) VALUES ({','.join('?' * (len(PRED_COLS) + 1))})",
                        [(batch_id, *r) for r in d.itertuples(index=False, name=None)])
        con.executemany(
            "INSERT INTO events (batch_id,timestamp,device_id,src_ip,dst_ip,dst_port,attack_type,confidence,anomaly_score,"
            "risk_score,risk_level,status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(batch_id, r.timestamp, r.device_id, r.src_ip, r.dst_ip, r.dst_port, r.predicted, r.confidence,
              r.anomaly_score, r.risk_score, r.risk_level, "open", now) for r in ev.itertuples()])
        con.commit()
    return batch_id, len(ev)


def list_batches(path=None):
    init_db(path)
    with closing(connect(path)) as con:
        return pd.read_sql_query("SELECT * FROM batches ORDER BY created_at DESC, rowid DESC", con)


def load_predictions(batch_id=None, path=None):
    init_db(path)
    q, args = "SELECT * FROM predictions", ()
    if batch_id:
        q, args = q + " WHERE batch_id=?", (batch_id,)
    with closing(connect(path)) as con:
        df = pd.read_sql_query(q, con, params=args)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_events(batch_id=None, path=None):
    init_db(path)
    q, args = "SELECT * FROM events", ()
    if batch_id:
        q, args = q + " WHERE batch_id=?", (batch_id,)
    with closing(connect(path)) as con:
        df = pd.read_sql_query(q + " ORDER BY risk_score DESC", con, params=args)
    return df


def update_event_status(ids, status, path=None):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    with closing(connect(path)) as con:
        cur = con.execute(f"UPDATE events SET status=? WHERE id IN ({','.join('?' * len(ids))})", (status, *ids))
        con.commit()
        return cur.rowcount
