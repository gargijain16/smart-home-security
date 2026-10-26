"""Reproducible synthetic smart-home IoT flow generator (NOT real traffic)."""
import numpy as np
import pandas as pd

CLASSES = ["normal", "port_scan", "brute_force", "dos", "suspicious_outbound"]
PROTO = {"TCP": 0, "UDP": 1, "ICMP": 2}
FEATURES = ["duration", "packets_per_sec", "bytes_sent", "bytes_received", "unique_dst_ports",
            "failed_logins", "syn_ratio", "conn_per_min", "dst_port", "is_external", "proto_code"]
REQUIRED_UPLOAD = ["duration", "packets_per_sec", "bytes_sent", "bytes_received", "unique_dst_ports",
                   "failed_logins", "syn_ratio", "conn_per_min", "dst_port", "is_external", "protocol"]
LOG_COLS = ["duration", "packets_per_sec", "bytes_sent", "bytes_received", "conn_per_min"]

# device_id, type, ip, normal ports, pps, bytes_sent, bytes_recv, conn/min, p_external
DEVICES = [
    ("cam-front-door", "IP camera", "192.168.1.21", [554, 443], 90, 400_000, 3_000, 6, 0.9),
    ("cam-backyard", "IP camera", "192.168.1.22", [554, 443], 80, 350_000, 3_000, 6, 0.9),
    ("thermostat-hall", "Thermostat", "192.168.1.31", [8883, 443], 2, 1_500, 800, 2, 0.8),
    ("smart-lock-main", "Smart lock", "192.168.1.41", [8883, 443], 1, 700, 500, 1, 0.7),
    ("speaker-kitchen", "Smart speaker", "192.168.1.51", [443, 8009], 40, 20_000, 120_000, 8, 0.85),
    ("tv-livingroom", "Smart TV", "192.168.1.61", [443, 80], 120, 15_000, 900_000, 5, 0.9),
    ("plug-office", "Smart plug", "192.168.1.71", [8883, 1883], 1, 500, 400, 1, 0.6),
    ("bulb-bedroom", "Smart bulb", "192.168.1.72", [1883, 443], 1, 400, 300, 1, 0.5),
    ("hub-gateway", "Hub", "192.168.1.2", [443, 8883, 53], 30, 12_000, 14_000, 15, 0.6),
    ("nas-storage", "NAS", "192.168.1.90", [445, 443, 22], 60, 80_000, 120_000, 4, 0.2),
    ("doorbell-front", "Video doorbell", "192.168.1.23", [443, 554], 70, 250_000, 4_000, 5, 0.9),
    ("laptop-guest", "Laptop", "192.168.1.101", [443, 80, 53], 55, 40_000, 150_000, 12, 0.9),
]
DEVICE_TYPES = {d[0]: d[1] for d in DEVICES}
BASE_TIME = pd.Timestamp("2026-01-01 00:00:00")


def _ext_ip(rng):
    # RFC 5737 documentation ranges: guaranteed not to be real hosts
    return f"{rng.choice(['203.0.113', '198.51.100'])}.{rng.integers(1, 255)}"


def _ln(rng, mean, sigma=0.5):
    return float(rng.lognormal(np.log(max(mean, 1e-3)), sigma))


def _row(rng, label):
    dev = DEVICES[rng.integers(len(DEVICES))]
    dev_id, _, dev_ip, ports, pps, bs, br, cpm, p_ext = dev
    stealth = label != "normal" and rng.random() < 0.20  # mild attacks overlap with normal
    r = dict(duration=_ln(rng, 20, 0.9), packets_per_sec=_ln(rng, pps, 0.5), bytes_sent=_ln(rng, bs, 0.6),
             bytes_received=_ln(rng, br, 0.6), unique_dst_ports=int(1 + rng.poisson(0.6)),
             failed_logins=int(rng.poisson(0.15)), syn_ratio=float(rng.beta(2, 8)),
             conn_per_min=_ln(rng, cpm, 0.5), dst_port=int(rng.choice(ports)),
             is_external=int(rng.random() < p_ext), protocol=str(rng.choice(["TCP", "UDP"], p=[0.85, 0.15])))
    src, dst = dev_ip, ("192.168.1.1" if not r["is_external"] else _ext_ip(rng))
    if label == "port_scan":
        r.update(unique_dst_ports=int(rng.integers(6, 25) if stealth else rng.integers(25, 400)),
                 syn_ratio=float(rng.beta(5, 2) if stealth else rng.beta(9, 1.5)),
                 packets_per_sec=_ln(rng, 40 if stealth else 90, 0.6), bytes_sent=_ln(rng, 3_000, 0.6),
                 bytes_received=_ln(rng, 800, 0.6), conn_per_min=_ln(rng, 60 if stealth else 220, 0.5),
                 dst_port=int(rng.integers(1, 65535)), duration=_ln(rng, 6, 0.6), protocol="TCP",
                 is_external=int(rng.random() < 0.3), failed_logins=0)
        src, dst = (_ext_ip(rng) if r["is_external"] else "192.168.1.250"), dev_ip
    elif label == "brute_force":
        r.update(failed_logins=int(rng.integers(3, 12) if stealth else rng.integers(10, 90)),
                 dst_port=int(rng.choice([22, 23, 2323, 21, 80, 8080])), unique_dst_ports=1,
                 conn_per_min=_ln(rng, 15 if stealth else 45, 0.5), packets_per_sec=_ln(rng, 12, 0.6),
                 syn_ratio=float(rng.beta(3, 4)), bytes_sent=_ln(rng, 2_500, 0.5), bytes_received=_ln(rng, 1_800, 0.5),
                 duration=_ln(rng, 45, 0.6), protocol="TCP", is_external=int(rng.random() < 0.8))
        src, dst = (_ext_ip(rng) if r["is_external"] else "192.168.1.250"), dev_ip
    elif label == "dos":
        udp = rng.random() < 0.4
        r.update(packets_per_sec=_ln(rng, 450 if stealth else 3_500, 0.5), conn_per_min=_ln(rng, 400 if stealth else 2_500, 0.5),
                 bytes_sent=_ln(rng, 150_000 if stealth else 900_000, 0.6), bytes_received=_ln(rng, 2_000, 0.6),
                 syn_ratio=float(rng.beta(2, 3) if udp else rng.beta(8, 2)), unique_dst_ports=int(rng.integers(1, 4)),
                 dst_port=int(rng.choice([80, 443, 53, 8883])), duration=_ln(rng, 8, 0.6),
                 protocol="UDP" if udp else "TCP", is_external=int(rng.random() < 0.85), failed_logins=0)
        src, dst = _ext_ip(rng), dev_ip
    elif label == "suspicious_outbound":
        mult = rng.uniform(3, 8) if stealth else rng.uniform(15, 150)
        r.update(bytes_sent=bs * mult, bytes_received=_ln(rng, 600, 0.6), is_external=1, duration=_ln(rng, 240, 0.5),
                 dst_port=int(rng.choice([443, 8443, 4444, 6667, 31337, 9001, 1337])), unique_dst_ports=1,
                 packets_per_sec=_ln(rng, max(pps, 20) * 1.5, 0.5), conn_per_min=_ln(rng, 4, 0.6),
                 syn_ratio=float(rng.beta(2, 7)), protocol="TCP", failed_logins=0)
        src, dst = dev_ip, _ext_ip(rng)
    r["failed_logins"] = int(r["failed_logins"])
    r.update(device_id=dev_id, device_type=DEVICE_TYPES[dev_id], src_ip=src, dst_ip=dst, label=label)
    return r


def generate_dataset(n=6000, seed=42, days=7):
    """Return a synthetic flow table. Same (n, seed) -> identical output."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(CLASSES, size=n, p=[0.70, 0.08, 0.08, 0.07, 0.07])
    df = pd.DataFrame([_row(rng, l) for l in labels])
    secs = np.sort(rng.integers(0, days * 86400, size=n))
    df.insert(0, "timestamp", BASE_TIME + pd.to_timedelta(secs, unit="s"))
    for c in ["duration", "packets_per_sec", "bytes_sent", "bytes_received", "conn_per_min"]:
        df[c] = df[c].round(2)
    df["syn_ratio"] = df["syn_ratio"].clip(0, 1).round(3)
    df["bytes_sent"] = df["bytes_sent"].round(0)
    df["bytes_received"] = df["bytes_received"].round(0)
    return df


def validate_upload(df):
    """Return list of missing required columns."""
    return [c for c in REQUIRED_UPLOAD if c not in df.columns]


def normalize_upload(df):
    """Fill optional columns so an uploaded CSV can be scored and stored."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df else BASE_TIME
    df["timestamp"] = df["timestamp"].fillna(BASE_TIME)
    for c, v in [("device_id", "unknown-device"), ("device_type", "Unknown"), ("src_ip", "0.0.0.0"), ("dst_ip", "0.0.0.0")]:
        if c not in df:
            df[c] = v
    df["protocol"] = df["protocol"].astype(str).str.upper().where(lambda s: s.isin(PROTO), "TCP")
    for c in REQUIRED_UPLOAD:
        if c != "protocol":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[c for c in REQUIRED_UPLOAD if c != "protocol"]).reset_index(drop=True)


def featurize(df):
    """Model matrix. Heavy-tailed columns are log1p-transformed."""
    X = pd.DataFrame(index=df.index)
    X["proto_code"] = df["protocol"].map(PROTO).fillna(0).astype(int)
    for c in FEATURES:
        if c != "proto_code":
            X[c] = np.log1p(df[c].clip(lower=0)) if c in LOG_COLS else df[c].astype(float)
    return X[FEATURES]
