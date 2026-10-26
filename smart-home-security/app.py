"""HomeShield Lab: Smart Home Cybersecurity dashboard (SYNTHETIC data). Run: streamlit run app.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smarthome import config, data, db, ml, pipeline  # noqa: E402

st.set_page_config(page_title="HomeShield Lab", page_icon="🛡️", layout="wide")

CLASS_COLORS = {"normal": "#5b7cfa", "port_scan": "#22d3ee", "brute_force": "#f59e0b", "dos": "#f43f5e",
                "suspicious_outbound": "#a78bfa"}
LEVEL_COLORS = {"Low": "#34d399", "Medium": "#fbbf24", "High": "#fb923c", "Critical": "#f43f5e"}
PAGES = ["Overview", "Traffic Analysis", "Device Monitoring", "Threat Detection", "Security Events",
         "Model Performance", "Data & Export"]

CSS = """
<style>
.stApp{background:radial-gradient(1100px 600px at 8% -10%,rgba(99,102,241,.25),transparent 60%),
 radial-gradient(900px 520px at 100% 0%,rgba(20,184,166,.17),transparent 55%),#070a13;color:#e6e9f2}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;max-width:1450px}
[data-testid="stSidebar"]{background:transparent}
[data-testid="stSidebar"]>div:first-child{margin:14px 0 14px 14px;border-radius:22px;background:rgba(255,255,255,.045);
 backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.09);
 box-shadow:0 24px 70px rgba(0,0,0,.55)}
.brand{font-size:1.25rem;font-weight:700;letter-spacing:.2px;margin:.2rem 0 0}
.brand small{display:block;font-weight:400;font-size:.76rem;color:#8b94ad;margin-top:2px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:6px 0 18px}
.kpi{background:rgba(255,255,255,.05);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
 border:1px solid rgba(255,255,255,.09);border-radius:18px;padding:16px 18px;box-shadow:0 14px 40px rgba(0,0,0,.4);
 border-top:2px solid var(--c,#5b7cfa)}
.kl{font-size:.8rem;color:#98a2bd}.kv{font-size:1.9rem;font-weight:700;line-height:1.25;color:#f4f6fb}
.ks{font-size:.76rem;color:#7f89a6}
.notice{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.28);color:#fcd98a;border-radius:14px;
 padding:9px 14px;font-size:.82rem;margin:0 0 16px}
h1,h2,h3{letter-spacing:.1px}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.08);
 border-radius:18px;padding:8px;box-shadow:0 12px 34px rgba(0,0,0,.35)}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.06)}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#5eead4;color:#5eead4}
@media(max-width:800px){.kv{font-size:1.5rem}.block-container{padding:1rem .6rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- helpers ----------
def style(fig, h=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      margin=dict(l=12, r=12, t=44, b=12), font=dict(color="#cbd5e1"), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)")
    return fig


def kpis(items):
    html = "".join(f'<div class="kpi" style="--c:{c}"><div class="kl">{l}</div><div class="kv">{v}</div><div class="ks">{s}</div></div>'
                   for l, v, s, c in items)
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def notice():
    st.markdown(f'<div class="notice">⚠ {config.SYNTHETIC_NOTICE}</div>', unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return ml.load_models()


def table(df, **kw):
    st.dataframe(df, width="stretch", hide_index=True, **kw)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ---------- bootstrap: models must exist ----------
bundle = get_bundle()
if bundle is None or db.list_batches().empty:
    st.title("🛡️ HomeShield Lab")
    notice()
    st.info("No trained models / stored data found. Run `python init_project.py` once, or initialise here (≈10-20 s).")
    c1, c2 = st.columns(2)
    rows = c1.number_input("Synthetic flows", 1000, 30000, config.DEFAULT_ROWS, 500)
    seed = c2.number_input("Random seed", 0, 99999, config.DEFAULT_SEED)
    if st.button("Generate synthetic data & train models", type="primary"):
        with st.spinner("Generating data, training Random Forest + Isolation Forest, scoring..."):
            pipeline.initialize(int(rows), int(seed))
        get_bundle.clear()
        st.rerun()
    st.stop()

metrics = bundle["metrics"]

# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand">🛡️ HomeShield Lab<small>Smart-home threat analytics · synthetic data</small></div>',
                unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    batches = db.list_batches()
    labels = {r.batch_id: f"{r.source} · {r.n_rows} rows" for r in batches.itertuples()}
    batch_id = st.selectbox("Data batch", list(labels), format_func=labels.get)
    st.caption(f"Models trained {metrics['trained_at']} (seed {metrics['seed']}). Loaded from disk, not retrained.")

df = db.load_predictions(batch_id)
events = db.load_events(batch_id)
has_label = df["label"].notna().any()

st.title(page)
notice()

# ---------- pages ----------
if page == "Overview":
    attacks = df[df["predicted"] != "normal"]
    crit_dev = df[df["risk_level"].isin(["High", "Critical"])]["device_id"].nunique()
    kpis([("Flows analysed", f"{len(df):,}", "in selected batch", "#5b7cfa"),
          ("Predicted attacks", f"{len(attacks):,}", f"{len(attacks) / max(len(df), 1):.1%} of flows", "#f43f5e"),
          ("Open events", f"{(events['status'] == 'open').sum():,}", f"{len(events):,} total", "#fb923c"),
          ("Mean risk score", f"{df['risk_score'].mean():.1f}", "0-100 heuristic", "#fbbf24"),
          ("Devices at risk", f"{crit_dev}", f"of {df['device_id'].nunique()} monitored", "#a78bfa")])
    a, b = st.columns(2)
    cnt = df["predicted"].value_counts().reset_index()
    cnt.columns = ["predicted", "flows"]
    a.plotly_chart(style(px.pie(cnt, names="predicted", values="flows", hole=.55, color="predicted",
                                color_discrete_map=CLASS_COLORS, title="Predicted traffic classes")), key="ov_pie")
    lv = df["risk_level"].value_counts().reindex(config.RISK_LABELS).fillna(0).reset_index()
    lv.columns = ["risk_level", "flows"]
    b.plotly_chart(style(px.bar(lv, x="risk_level", y="flows", color="risk_level", color_discrete_map=LEVEL_COLORS,
                                title="Flows by risk level")), key="ov_lv")
    t = df.set_index("timestamp").resample("6h").agg(mean_risk=("risk_score", "mean"), attacks=("predicted", lambda s: (s != "normal").sum())).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["timestamp"], y=t["mean_risk"], name="Mean risk", fill="tozeroy", line=dict(color="#5eead4")))
    fig.add_trace(go.Bar(x=t["timestamp"], y=t["attacks"], name="Predicted attacks", yaxis="y2", marker_color="rgba(244,63,94,.45)"))
    fig.update_layout(title="Risk and attack volume over time (6h buckets)", yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(style(fig, 360), key="ov_t")
    top = df.groupby("device_id")["risk_score"].mean().sort_values().tail(8).reset_index()
    st.plotly_chart(style(px.bar(top, x="risk_score", y="device_id", orientation="h", title="Highest mean-risk devices",
                                 color="risk_score", color_continuous_scale="Turbo"), 340), key="ov_top")

elif page == "Traffic Analysis":
    c1, c2, c3 = st.columns(3)
    devs = c1.multiselect("Devices", sorted(df["device_id"].unique()))
    protos = c2.multiselect("Protocols", sorted(df["protocol"].unique()))
    classes = c3.multiselect("Predicted class", sorted(df["predicted"].unique()))
    f = df
    if devs: f = f[f["device_id"].isin(devs)]
    if protos: f = f[f["protocol"].isin(protos)]
    if classes: f = f[f["predicted"].isin(classes)]
    if f.empty:
        st.warning("No flows match these filters.")
        st.stop()
    kpis([("Flows", f"{len(f):,}", "after filters", "#5b7cfa"),
          ("Bytes sent", f"{f['bytes_sent'].sum() / 1e6:,.1f} MB", "total outbound", "#22d3ee"),
          ("Peak pkts/s", f"{f['packets_per_sec'].max():,.0f}", "single flow", "#f59e0b"),
          ("Failed logins", f"{int(f['failed_logins'].sum()):,}", "sum", "#f43f5e")])
    v = f.set_index("timestamp").groupby([pd.Grouper(freq="3h"), "predicted"]).size().reset_index(name="flows")
    st.plotly_chart(style(px.area(v, x="timestamp", y="flows", color="predicted", color_discrete_map=CLASS_COLORS,
                                  title="Flows over time by predicted class"), 340), key="ta_area")
    a, b = st.columns(2)
    s = f.sample(min(len(f), 2500), random_state=1)
    a.plotly_chart(style(px.scatter(s, x="packets_per_sec", y="bytes_sent", color="predicted", log_x=True, log_y=True,
                                    color_discrete_map=CLASS_COLORS, opacity=.7, title="Packet rate vs bytes sent (log)")), key="ta_sc")
    b.plotly_chart(style(px.box(f, x="predicted", y="unique_dst_ports", color="predicted", color_discrete_map=CLASS_COLORS,
                                log_y=True, title="Unique destination ports per flow")), key="ta_box")
    c, d = st.columns(2)
    pt = f["dst_port"].value_counts().head(12).reset_index()
    pt.columns = ["dst_port", "flows"]
    pt["dst_port"] = pt["dst_port"].astype(str)
    c.plotly_chart(style(px.bar(pt, x="dst_port", y="flows", title="Top destination ports")), key="ta_port")
    d.plotly_chart(style(px.histogram(f, x="failed_logins", color="predicted", color_discrete_map=CLASS_COLORS, nbins=30,
                                      title="Failed-login distribution")), key="ta_fl")
    if has_label:
        ct = pd.crosstab(f["label"], f["predicted"])
        st.caption("Simulated ground-truth label (rows) vs. model prediction (columns):")
        st.dataframe(ct, width="stretch")

elif page == "Device Monitoring":
    g = df.groupby(["device_id", "device_type"]).agg(flows=("risk_score", "size"), mean_risk=("risk_score", "mean"),
                                                     max_risk=("risk_score", "max"), anomalies=("is_anomaly", "sum"),
                                                     high_share=("risk_level", lambda s: s.isin(["High", "Critical"]).mean()),
                                                     bytes_sent_mb=("bytes_sent", lambda s: s.sum() / 1e6)).reset_index()
    avg = max(float(g["high_share"].mean()), 1e-9)  # status is relative to the batch average
    g["status"] = np.where(g["high_share"] >= 1.3 * avg, "Highest exposure", np.where(g["high_share"] >= 1.1 * avg, "Elevated", "Typical"))
    g = g.round(2).sort_values("mean_risk", ascending=False)
    kpis([("Devices", f"{len(g)}", "monitored", "#5b7cfa"), ("Highest exposure", f"{(g.status == 'Highest exposure').sum()}", "high-risk share ≥1.3× batch avg", "#f43f5e"),
          ("Elevated", f"{(g.status == 'Elevated').sum()}", "1.1-1.3× batch avg", "#fb923c"), ("Typical", f"{(g.status == 'Typical').sum()}", "<1.1× batch avg", "#34d399")])
    table(g)
    heat = pd.crosstab(df["device_id"], df["predicted"])
    st.plotly_chart(style(px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Magma",
                                    title="Predicted class counts per device"), 420), key="dm_heat")
    dev = st.selectbox("Inspect device", g["device_id"])
    d = df[df["device_id"] == dev]
    st.plotly_chart(style(px.scatter(d, x="timestamp", y="risk_score", color="predicted", color_discrete_map=CLASS_COLORS,
                                     title=f"Risk timeline: {dev}"), 340), key="dm_dev")

elif page == "Threat Detection":
    t1, t2, t3 = st.tabs(["Upload CSV", "Single-flow simulator", "Generate new synthetic batch"])

    def show_result(scored, key):
        kpis([("Flows scored", f"{len(scored):,}", "", "#5b7cfa"),
              ("Predicted attacks", f"{(scored['predicted'] != 'normal').sum():,}", "", "#f43f5e"),
              ("High/Critical", f"{scored['risk_level'].isin(['High', 'Critical']).sum():,}", "→ become events", "#fb923c"),
              ("Mean risk", f"{scored['risk_score'].mean():.1f}", "", "#fbbf24")])
        if "label" in scored and scored["label"].notna().any():
            acc = (scored["label"] == scored["predicted"]).mean()
            st.info(f"Agreement with the file's own `label` column: {acc:.1%} (only meaningful if labels use the same classes: {', '.join(data.CLASSES)}).")
        cnt = scored["predicted"].value_counts().reset_index()
        cnt.columns = ["predicted", "flows"]
        st.plotly_chart(style(px.bar(cnt, x="predicted", y="flows", color="predicted", color_discrete_map=CLASS_COLORS,
                                     title="Predictions"), 300), key=f"{key}_c")
        table(scored.sort_values("risk_score", ascending=False).head(500))
        st.download_button("Download scored CSV", csv_bytes(scored), "scored_flows.csv", "text/csv", key=f"{key}_dl")

    with t1:
        st.caption(f"Required columns: {', '.join(data.REQUIRED_UPLOAD)}. Optional: timestamp, device_id, device_type, src_ip, dst_ip, label.")
        tpl = data.generate_dataset(25, 123)[["timestamp", "device_id", *data.REQUIRED_UPLOAD]]
        st.download_button("Download CSV template (synthetic sample)", csv_bytes(tpl), "template.csv", "text/csv")
        up = st.file_uploader("Upload flow CSV", type="csv")
        if up is not None:
            try:
                raw = pd.read_csv(up)
                miss = data.validate_upload(raw)
                if miss:
                    st.error(f"Missing required columns: {', '.join(miss)}")
                else:
                    clean = data.normalize_upload(raw)
                    st.session_state["up"] = (up.name, ml.score(clean, bundle), len(raw) - len(clean))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read file: {e}")
        if "up" in st.session_state:
            name, scored, dropped = st.session_state["up"]
            if dropped: st.warning(f"{dropped} rows with non-numeric values were dropped.")
            show_result(scored, "up")
            if st.button("Save to database", key="up_save"):
                bid, n = db.save_scored(scored, f"upload:{name}")
                st.success(f"Saved batch {bid} with {n} security events. Pick it in the sidebar.")

    with t2:
        presets = {
            "Normal camera stream": dict(duration=30, packets_per_sec=90, bytes_sent=400000, bytes_received=3000, unique_dst_ports=1, failed_logins=0, syn_ratio=.2, conn_per_min=6, dst_port=443, is_external=1, protocol="TCP"),
            "Port scan": dict(duration=5, packets_per_sec=100, bytes_sent=3000, bytes_received=800, unique_dst_ports=150, failed_logins=0, syn_ratio=.95, conn_per_min=250, dst_port=1024, is_external=0, protocol="TCP"),
            "SSH brute force": dict(duration=45, packets_per_sec=12, bytes_sent=2500, bytes_received=1800, unique_dst_ports=1, failed_logins=40, syn_ratio=.4, conn_per_min=50, dst_port=22, is_external=1, protocol="TCP"),
            "DoS flood": dict(duration=8, packets_per_sec=4000, bytes_sent=900000, bytes_received=2000, unique_dst_ports=2, failed_logins=0, syn_ratio=.9, conn_per_min=2500, dst_port=80, is_external=1, protocol="TCP"),
            "Data exfiltration": dict(duration=240, packets_per_sec=60, bytes_sent=20000000, bytes_received=600, unique_dst_ports=1, failed_logins=0, syn_ratio=.2, conn_per_min=4, dst_port=4444, is_external=1, protocol="TCP")}
        pick = st.selectbox("Preset", list(presets))
        p = presets[pick]
        cols = st.columns(4)
        vals = {}
        for i, (k, v) in enumerate(p.items()):
            if k == "protocol":
                vals[k] = cols[i % 4].selectbox(k, ["TCP", "UDP", "ICMP"], index=["TCP", "UDP", "ICMP"].index(v), key=f"{k}_{pick}")
            else:
                vals[k] = cols[i % 4].number_input(k, value=float(v), key=f"{k}_{pick}")
        one = ml.score(data.normalize_upload(pd.DataFrame([vals])), bundle).iloc[0]
        a, b = st.columns([1, 1])
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(one["risk_score"]), title={"text": f"Risk: {one['risk_level']}"},
                                       gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=LEVEL_COLORS[one["risk_level"]]))))
        a.plotly_chart(style(gauge, 300), key="sim_g")
        pr = pd.DataFrame({"class": bundle["rf"].classes_, "probability": bundle["rf"].predict_proba(data.featurize(data.normalize_upload(pd.DataFrame([vals]))))[0]})
        b.plotly_chart(style(px.bar(pr, x="class", y="probability", color="class", color_discrete_map=CLASS_COLORS,
                                    title=f"Predicted: {one['predicted']} (anomaly score {one['anomaly_score']:.2f})", range_y=[0, 1]), 300), key="sim_p")

    with t3:
        c1, c2 = st.columns(2)
        n = c1.slider("Rows", 500, 10000, 2000, 500)
        sd = c2.number_input("Seed", 0, 99999, 7)
        if st.button("Generate & score"):
            st.session_state["gen"] = (f"synthetic_seed{int(sd)}_n{n}", ml.score(data.generate_dataset(n, int(sd)), bundle))
        if "gen" in st.session_state:
            name, scored = st.session_state["gen"]
            st.caption("Fresh synthetic flows the models have not seen (different seed from training).")
            show_result(scored, "gen")
            if st.button("Save to database", key="gen_save"):
                bid, n_ev = db.save_scored(scored, name)
                st.success(f"Saved batch {bid} with {n_ev} security events.")

elif page == "Security Events":
    if events.empty:
        st.info("No events in this batch.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    stf = c1.multiselect("Status", db.STATUSES, default=["open", "investigating"])
    lvf = c2.multiselect("Severity", config.RISK_LABELS, default=["High", "Critical"])
    tyf = c3.multiselect("Attack type", sorted(events["attack_type"].unique()))
    f = events[events["status"].isin(stf) & events["risk_level"].isin(lvf)]
    if tyf: f = f[f["attack_type"].isin(tyf)]
    kpis([("Matching events", f"{len(f):,}", f"of {len(events):,}", "#fb923c"),
          ("Critical", f"{(f['risk_level'] == 'Critical').sum():,}", "", "#f43f5e"),
          ("Resolved / FP", f"{events['status'].isin(['resolved', 'false_positive']).sum():,}", "whole batch", "#34d399")])
    table(f.drop(columns=["batch_id"]).head(1000))
    with st.expander("Update event status"):
        ids = st.multiselect("Event IDs (top 300 shown)", f["id"].head(300).tolist())
        new = st.selectbox("New status", db.STATUSES, index=1)
        if st.button("Apply") and ids:
            st.success(f"Updated {db.update_event_status(ids, new)} events.")
            st.rerun()
    st.download_button("Export events CSV", csv_bytes(events), "security_events.csv", "text/csv")

elif page == "Model Performance":
    rf, iso = metrics["random_forest"], metrics["isolation_forest"]
    st.caption(f"{metrics['data_note']} Train {metrics['n_train']:,} / test {metrics['n_test']:,} rows. "
               "Near-perfect scores are expected here because the simulator makes attack classes statistically distinct; "
               "real home traffic is far messier.")
    kpis([("RF accuracy", f"{rf['accuracy']:.3f}", "held-out synthetic", "#5eead4"), ("RF macro-F1", f"{rf['macro_f1']:.3f}", "", "#5b7cfa"),
          ("IF precision", f"{iso['precision_vs_attack']:.3f}", "anomaly vs attack", "#a78bfa"), ("IF recall", f"{iso['recall_vs_attack']:.3f}", "", "#f59e0b"),
          ("IF false-positive rate", f"{iso['false_positive_rate_on_normal']:.3f}", "on normal flows", "#f43f5e")])
    cm = np.array(rf["confusion_matrix"])
    norm = st.toggle("Normalise rows (recall view)", value=False)
    z = cm / cm.sum(1, keepdims=True).clip(min=1) if norm else cm
    a, b = st.columns(2)
    a.plotly_chart(style(px.imshow(z, x=rf["labels"], y=rf["labels"], text_auto=".2f" if norm else True, aspect="auto",
                                   color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"), title="Confusion matrix (test set)"), 420), key="mp_cm")
    fi = pd.Series(rf["feature_importance"]).sort_values().reset_index()
    fi.columns = ["feature", "importance"]
    b.plotly_chart(style(px.bar(fi, x="importance", y="feature", orientation="h", title="Random Forest feature importance"), 420), key="mp_fi")
    pc = pd.DataFrame(rf["per_class"]).T[["precision", "recall", "f1-score", "support"]].round(3).reset_index().rename(columns={"index": "class"})
    table(pc)
    st.subheader("Retrain")
    st.caption("Regenerates synthetic data with the chosen seed, retrains both models, and replaces the stored database.")
    c1, c2 = st.columns(2)
    rn = c1.number_input("Rows", 1000, 30000, metrics["n_rows"], 500)
    rs = c2.number_input("Seed", 0, 99999, metrics["seed"])
    if st.button("Retrain models"):
        with st.spinner("Training..."):
            pipeline.initialize(int(rn), int(rs))
        get_bundle.clear()
        st.rerun()

elif page == "Data & Export":
    st.caption("Stored in SQLite at `data/smart_home_security.db`. Models in `models/`.")
    kpis([("Batches", f"{len(batches)}", "in database", "#5b7cfa"), ("Rows in view", f"{len(df):,}", "", "#22d3ee"),
          ("Events in view", f"{len(events):,}", "", "#fb923c")])
    table(batches)
    st.subheader("Preview of scored flows")
    table(df.head(300))
    c1, c2 = st.columns(2)
    c1.download_button("Download scored predictions (this batch)", csv_bytes(df), "predictions.csv", "text/csv")
    if config.DATASET_PATH.exists():
        c2.download_button("Download raw synthetic training dataset", config.DATASET_PATH.read_bytes(), "synthetic_iot_traffic.csv", "text/csv")
    st.subheader("Feature schema")
    table(pd.DataFrame({"column": data.REQUIRED_UPLOAD, "note": ["log1p-transformed for models" if c in data.LOG_COLS else "" for c in data.REQUIRED_UPLOAD]}))
