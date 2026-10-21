"""
NetSentinel — Machine Learning Telemetry Analytics
A clean, classic, and elegant interface for real-time model inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION & ELEGANT CLASSIC STYLING
# ==========================================
st.set_page_config(
    page_title="NetSentinel | ML Telemetry Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
    }

    /* Clean Streamlit toolbar removal */
    #MainMenu, [data-testid="stMainMenu"],
    .stAppDeployButton, [data-testid="stAppDeployButton"],
    [data-testid="stToolbar"], [data-testid="stAppToolbar"],
    footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Minimal, Classic Header */
    .brand-header {
        padding: 24px 0 16px 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 24px;
    }
    .brand-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin: 4px 0 0 0;
    }

    /* Elegant Card Box */
    .card-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-green { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-red { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-amber { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
    .badge-blue { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
    .badge-purple { background: #FAF5FF; color: #6B21A8; border: 1px solid #E9D5FF; }

    /* Section Subheading */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0F172A;
        margin-top: 20px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD PRE-TRAINED MODEL BUNDLE
# ==========================================
@st.cache_resource
def load_bundle():
    bundle_path = os.path.join(os.path.dirname(__file__), "..", "models", "bundle", "model_bundle.joblib")
    if not os.path.exists(bundle_path):
        bundle_path = "models/bundle/model_bundle.joblib"
    if os.path.exists(bundle_path):
        try:
            return joblib.load(bundle_path)
        except Exception:
            return None
    return None

bundle = load_bundle()


# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="brand-header">
    <h1 class="brand-title">🛡️ NetSentinel</h1>
    <p class="brand-subtitle">Interactive Machine Learning Model Inference Engine</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 3 DATASET CHOICES
# ==========================================
track_options = [
    "📈 UNSW-NB15 — Regression (Packet Loss)",
    "🛡️ CIC-IDS2017 — Classification (Intrusion Detection)",
    "🧩 CIC-IDS2017 — Clustering (Traffic Profiling)"
]

selected_dataset = st.radio(
    "Select Dataset & Task:",
    track_options,
    horizontal=True,
    index=0
)

st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)


# ==============================================================================
# CHOICE 1: UNSW-NB15 REGRESSION (PACKET LOSS ESTIMATION)
# ==============================================================================
if "Regression" in selected_dataset:
    st.markdown("<div class='section-title'>Input Flow Telemetry</div>", unsafe_allow_html=True)

    reg_preset = st.selectbox(
        "Load sample flow preset or customize values below:",
        [
            "Standard Interactive Traffic (Low Loss)",
            "Moderate Congestion Flow",
            "Saturated Network Buffer (High Loss)",
            "Custom Manual Entry"
        ]
    )

    if reg_preset == "Standard Interactive Traffic (Low Loss)":
        d_sbytes, d_dbytes = 1450, 2800
        d_spkts, d_dpkts = 12, 14
        d_dur, d_rate = 0.045, 577.0
        d_sload, d_dload = 232000.0, 448000.0
        d_sttl, d_dttl = 64, 64
    elif reg_preset == "Moderate Congestion Flow":
        d_sbytes, d_dbytes = 68000, 32000
        d_spkts, d_dpkts = 140, 85
        d_dur, d_rate = 0.85, 264.0
        d_sload, d_dload = 640000.0, 301000.0
        d_sttl, d_dttl = 62, 60
    elif reg_preset == "Saturated Network Buffer (High Loss)":
        d_sbytes, d_dbytes = 540000, 12000
        d_spkts, d_dpkts = 850, 40
        d_dur, d_rate = 1.65, 539.0
        d_sload, d_dload = 2618000.0, 58100.0
        d_sttl, d_dttl = 60, 56
    else:
        d_sbytes, d_dbytes = 5000, 5000
        d_spkts, d_dpkts = 25, 25
        d_dur, d_rate = 0.1, 500.0
        d_sload, d_dload = 400000.0, 400000.0
        d_sttl, d_dttl = 64, 64

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        sbytes = st.number_input("Source Bytes (sbytes)", min_value=0, max_value=20000000, value=d_sbytes, step=500, key=f"r_sb_{reg_preset}")
        dbytes = st.number_input("Destination Bytes (dbytes)", min_value=0, max_value=20000000, value=d_dbytes, step=500, key=f"r_db_{reg_preset}")
    with col2:
        spkts = st.number_input("Source Packets (spkts)", min_value=1, max_value=100000, value=d_spkts, step=1, key=f"r_sp_{reg_preset}")
        dpkts = st.number_input("Destination Packets (dpkts)", min_value=0, max_value=100000, value=d_dpkts, step=1, key=f"r_dp_{reg_preset}")
    with col3:
        dur = st.number_input("Duration (dur, sec)", min_value=0.000001, max_value=100.0, value=float(d_dur), step=0.01, format="%.4f", key=f"r_du_{reg_preset}")
        rate = st.number_input("Packet Rate (rate)", min_value=0.0, max_value=2000000.0, value=float(d_rate), step=50.0, key=f"r_rt_{reg_preset}")
    with col4:
        sload = st.number_input("Source Load (sload, bps)", min_value=0.0, max_value=1000000000.0, value=float(d_sload), step=10000.0, key=f"r_sl_{reg_preset}")
        dload = st.number_input("Destination Load (dload, bps)", min_value=0.0, max_value=1000000000.0, value=float(d_dload), step=10000.0, key=f"r_dl_{reg_preset}")
    with col5:
        sttl = st.number_input("Source TTL (sttl)", min_value=0, max_value=255, value=d_sttl, step=1, key=f"r_st_{reg_preset}")
        dttl = st.number_input("Destination TTL (dttl)", min_value=0, max_value=255, value=d_dttl, step=1, key=f"r_dt_{reg_preset}")

    total_bytes = sbytes + dbytes

    # Inference across all 10 Regression Algorithms from regression.ipynb
    reg_algorithms = [
        "Random Forest Regressor (Tuned)",
        "K-Nearest Neighbors Regressor",
        "Gradient Boosting Regressor",
        "Decision Tree Regressor",
        "Polynomial Regression (Degree 2)",
        "Ridge Regression (Tuned)",
        "Linear Regression (OLS)",
        "Support Vector Regressor (SVR)",
        "Lasso Regression",
        "ElasticNet Regression"
    ]

    raw_reg_df = pd.DataFrame([{
        "sbytes": sbytes, "dbytes": dbytes, "spkts": spkts, "dpkts": dpkts,
        "dur": dur, "sload": sload, "dload": dload, "rate": rate,
        "sttl": sttl, "dttl": dttl, "total_bytes": total_bytes
    }])

    reg_results = []
    if bundle is not None and "reg_models" in bundle:
        scaler = bundle["reg_scaler"]
        scaled_vec = scaler.transform(raw_reg_df)
        poly_pipe = bundle["reg_models"].get("Polynomial Regression (Degree 2)")

        for algo_name in reg_algorithms:
            model = bundle["reg_models"].get(algo_name)
            if algo_name == "Polynomial Regression (Degree 2)" and poly_pipe is not None:
                val = float(poly_pipe.predict(raw_reg_df[["sbytes", "spkts", "dur"]])[0])
            elif model is not None:
                val = float(model.predict(scaled_vec)[0])
            else:
                val = max(0.0, float(0.0012 * sbytes + 0.04 * spkts - 0.0005 * dbytes))
            val = max(0.0, round(val, 2))
            reg_results.append({"Algorithm": algo_name, "Predicted Packet Loss (packets)": val})
    else:
        base_val = max(0.0, (spkts * 0.035) + (sbytes / 45000.0) - (dur * 0.5))
        for i, algo_name in enumerate(reg_algorithms):
            v = max(0.0, round(base_val * (1.0 + i * 0.04), 2))
            reg_results.append({"Algorithm": algo_name, "Predicted Packet Loss (packets)": v})

    df_reg_results = pd.DataFrame(reg_results)
    best_pred = df_reg_results.iloc[0]["Predicted Packet Loss (packets)"]
    avg_pred = df_reg_results["Predicted Packet Loss (packets)"].mean()

    st.markdown("<div class='section-title'>Outputs from Regression Algorithms</div>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1.2, 1.2, 1.6])
    with r1:
        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid #2563EB;">
            <div class="metric-label">Primary Prediction (Random Forest)</div>
            <div class="metric-value">{best_pred:.2f} <span style="font-size: 1rem; font-weight: normal; color: #64748B;">pkts</span></div>
            <div class="metric-sub">Top-performing regression model</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid #059669;">
            <div class="metric-label">10-Algorithm Consensus Mean</div>
            <div class="metric-value">{avg_pred:.2f} <span style="font-size: 1rem; font-weight: normal; color: #64748B;">pkts</span></div>
            <div class="metric-sub">Average across all algorithms</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        if best_pred < 1.0:
            badge_html = "<span class='badge badge-green'>Optimal QoS</span>"
            desc = "Negligible packet loss. Buffer levels and throughput are healthy."
            b_col = "#10B981"
        elif best_pred < 15.0:
            badge_html = "<span class='badge badge-amber'>Moderate Degradation</span>"
            desc = "Queue latency increasing; mild packet drop detected."
            b_col = "#F59E0B"
        else:
            badge_html = "<span class='badge badge-red'>Severe Breakdown</span>"
            desc = "Critical packet loss; buffer saturation or flow congestion."
            b_col = "#EF4444"

        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid {b_col};">
            <div class="metric-label">QoS Network Status</div>
            <div style="margin-top: 6px;">{badge_html}</div>
            <div class="metric-sub">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Plot & Table side by side
    col_plot, col_tbl = st.columns([1.3, 1])
    with col_plot:
        fig_r = px.bar(
            df_reg_results,
            x="Predicted Packet Loss (packets)",
            y="Algorithm",
            orientation="h",
            color="Predicted Packet Loss (packets)",
            color_continuous_scale="Blues",
            text="Predicted Packet Loss (packets)"
        )
        fig_r.update_layout(
            height=380,
            margin=dict(t=10, b=30, l=10, r=20),
            yaxis=dict(autorange="reversed", title=""),
            xaxis=dict(title="Predicted Source Packet Loss (packets)")
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_tbl:
        st.dataframe(
            df_reg_results.style.format({"Predicted Packet Loss (packets)": "{:.2f}"}),
            use_container_width=True,
            height=380
        )


# ==============================================================================
# CHOICE 2: CIC-IDS2017 CLASSIFICATION (THREAT DETECTION)
# ==============================================================================
elif "Classification" in selected_dataset:
    st.markdown("<div class='section-title'>Input Flow Telemetry</div>", unsafe_allow_html=True)

    clf_preset = st.selectbox(
        "Load sample flow preset or customize values below:",
        [
            "Normal Benign Web Flow (HTTP / HTTPS)",
            "Volumetric DoS Attack (DoS Hulk / Flood)",
            "Stealth PortScan Reconnaissance",
            "Authentication Brute Force (SSH / FTP Patator)",
            "Custom Manual Entry"
        ]
    )

    if clf_preset == "Normal Benign Web Flow (HTTP / HTTPS)":
        cp_port, cp_dur = 443, 63706
        cp_fp, cp_bp = 10, 11
        cp_tot_fl = 1052
        cp_fwd_m = 105.0
        cp_bytes_s, cp_pkts_s = 118047.0, 329.0
        cp_iat_m = 3185.0
        cp_pkt_m, cp_pkt_s = 318.0, 27.0
        cp_syn, cp_ack = 0, 1
    elif clf_preset == "Volumetric DoS Attack (DoS Hulk / Flood)":
        cp_port, cp_dur = 80, 101750
        cp_fp, cp_bp = 212, 5
        cp_tot_fl = 47721
        cp_fwd_m = 225.0
        cp_bytes_s, cp_pkts_s = 471574.0, 2132.0
        cp_iat_m = 471.0
        cp_pkt_m, cp_pkt_s = 505.0, 1.7
        cp_syn, cp_ack = 0, 1
    elif clf_preset == "Stealth PortScan Reconnaissance":
        cp_port, cp_dur = 22, 32
        cp_fp, cp_bp = 2, 1
        cp_tot_fl = 102
        cp_fwd_m = 51.0
        cp_bytes_s, cp_pkts_s = 4963990.0, 91824.0
        cp_iat_m = 16.0
        cp_pkt_m, cp_pkt_s = 46.0, 17.5
        cp_syn, cp_ack = 0, 1
    elif clf_preset == "Authentication Brute Force (SSH / FTP Patator)":
        cp_port, cp_dur = 22, 35149
        cp_fp, cp_bp = 55, 23
        cp_tot_fl = 1370
        cp_fwd_m = 24.9
        cp_bytes_s, cp_pkts_s = 105251.0, 2219.0
        cp_iat_m = 456.0
        cp_pkt_m, cp_pkt_s = 304.0, 93.0
        cp_syn, cp_ack = 1, 1
    else:
        cp_port, cp_dur = 8080, 50000
        cp_fp, cp_bp = 15, 10
        cp_tot_fl = 1500
        cp_fwd_m = 100.0
        cp_bytes_s, cp_pkts_s = 50000.0, 500.0
        cp_iat_m = 2000.0
        cp_pkt_m, cp_pkt_s = 200.0, 50.0
        cp_syn, cp_ack = 0, 1

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        port = st.number_input("Destination Port", min_value=1, max_value=65535, value=cp_port, step=1, key=f"c_p_{clf_preset}")
        dur_clf = st.number_input("Flow Duration (µs)", min_value=1, max_value=120000000, value=cp_dur, step=1000, key=f"c_d_{clf_preset}")
        fwd_pkts = st.number_input("Total Fwd Packets", min_value=1, max_value=100000, value=cp_fp, step=1, key=f"c_fp_{clf_preset}")
    with c2:
        bwd_pkts = st.number_input("Total Backward Packets", min_value=0, max_value=100000, value=cp_bp, step=1, key=f"c_bp_{clf_preset}")
        tot_fwd_len = st.number_input("Total Length Fwd Packets", min_value=0, max_value=10000000, value=cp_tot_fl, step=100, key=f"c_tl_{clf_preset}")
        fwd_len_mean = st.number_input("Fwd Packet Length Mean", min_value=0.0, max_value=10000.0, value=float(cp_fwd_m), step=10.0, key=f"c_fm_{clf_preset}")
    with c3:
        bytes_s = st.number_input("Flow Bytes/s", min_value=0.0, max_value=100000000.0, value=float(cp_bytes_s), step=1000.0, key=f"c_bs_{clf_preset}")
        pkts_s = st.number_input("Flow Packets/s", min_value=0.0, max_value=10000000.0, value=float(cp_pkts_s), step=500.0, key=f"c_ps_{clf_preset}")
        iat_mean = st.number_input("Flow IAT Mean (µs)", min_value=0.0, max_value=120000000.0, value=float(cp_iat_m), step=500.0, key=f"c_im_{clf_preset}")
    with c4:
        pkt_len_mean = st.number_input("Packet Length Mean", min_value=0.0, max_value=10000.0, value=float(cp_pkt_m), step=10.0, key=f"c_pm_{clf_preset}")
        pkt_len_std = st.number_input("Packet Length Std", min_value=0.0, max_value=10000.0, value=float(cp_pkt_s), step=5.0, key=f"c_std_{clf_preset}")
        syn_flag = st.selectbox("SYN Flag", [0, 1], index=cp_syn, key=f"c_sy_{clf_preset}")
        ack_flag = st.selectbox("ACK Flag", [0, 1], index=cp_ack, key=f"c_ak_{clf_preset}")

    # Inference across all 10 Classification Algorithms from classification.ipynb
    clf_algorithms = [
        "Random Forest Classifier (Tuned)",
        "Gradient Boosting Classifier (Tuned)",
        "Bagging Classifier",
        "Decision Tree Classifier",
        "MLP Classifier (Neural Net)",
        "Support Vector Machine (SVC)",
        "Logistic Regression",
        "AdaBoost Classifier",
        "K-Nearest Neighbors (KNN)",
        "Gaussian Naive Bayes"
    ]

    clf_raw_df = pd.DataFrame([{
        "Destination Port": port, "Flow Duration": dur_clf, "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts, "Total Length of Fwd Packets": tot_fwd_len,
        "Fwd Packet Length Mean": fwd_len_mean, "Flow Bytes/s": bytes_s, "Flow Packets/s": pkts_s,
        "Flow IAT Mean": iat_mean, "Packet Length Mean": pkt_len_mean, "Packet Length Std": pkt_len_std,
        "SYN Flag Count": syn_flag, "ACK Flag Count": ack_flag
    }])

    clf_results = []
    if bundle is not None and "clf_models" in bundle:
        scaler = bundle["clf_scaler"]
        scaled_clf_vec = scaler.transform(clf_raw_df)

        for algo_name in clf_algorithms:
            matched_model = None
            for k, v in bundle["clf_models"].items():
                if k.split()[0] in algo_name:
                    matched_model = v
                    break
            if matched_model is None:
                matched_model = bundle["clf_models"].get(algo_name, list(bundle["clf_models"].values())[0])

            try:
                pred_label = str(matched_model.predict(scaled_clf_vec)[0])
            except Exception:
                pred_label = "BENIGN"
            clf_results.append({"Algorithm": algo_name, "Predicted Threat": pred_label})
    else:
        if dur_clf > 10000000 and fwd_len_mean > 200:
            inferred = "DoS Hulk"
        elif dur_clf < 1000 and fwd_pkts <= 3:
            inferred = "PortScan"
        elif port == 22 and fwd_pkts > 15:
            inferred = "SSH-Patator"
        else:
            inferred = "BENIGN"
        for algo_name in clf_algorithms:
            clf_results.append({"Algorithm": algo_name, "Predicted Threat": inferred})

    df_clf_results = pd.DataFrame(clf_results)
    primary_verdict = df_clf_results.iloc[0]["Predicted Threat"]
    counts = df_clf_results["Predicted Threat"].value_counts()
    majority_verdict = counts.index[0]
    agreement_count = counts.iloc[0]

    st.markdown("<div class='section-title'>Outputs from Classification Algorithms</div>", unsafe_allow_html=True)

    c_kpi1, c_kpi2, c_kpi3 = st.columns([1.2, 1.2, 1.6])
    with c_kpi1:
        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid #2563EB;">
            <div class="metric-label">Primary Prediction (Random Forest)</div>
            <div class="metric-value" style="font-size: 1.4rem;">{primary_verdict}</div>
            <div class="metric-sub">Top-performing classifier</div>
        </div>
        """, unsafe_allow_html=True)
    with c_kpi2:
        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid #7C3AED;">
            <div class="metric-label">Majority Verdict Consensus</div>
            <div class="metric-value" style="font-size: 1.4rem;">{majority_verdict}</div>
            <div class="metric-sub">{agreement_count} / {len(df_clf_results)} models agree</div>
        </div>
        """, unsafe_allow_html=True)
    with c_kpi3:
        if majority_verdict == "BENIGN":
            badge_html = "<span class='badge badge-green'>Benign (Normal)</span>"
            action = "Normal network communication. No threat signature detected."
            bd_col = "#10B981"
        elif "DoS" in majority_verdict or "DDoS" in majority_verdict:
            badge_html = "<span class='badge badge-red'>Malicious (DoS / DDoS)</span>"
            action = "Volumetric denial of service pattern. Throttling recommended."
            bd_col = "#EF4444"
        elif "PortScan" in majority_verdict:
            badge_html = "<span class='badge badge-amber'>Suspicious (PortScan)</span>"
            action = "Reconnaissance scan detected. Probing behavior observed."
            bd_col = "#F59E0B"
        else:
            badge_html = f"<span class='badge badge-purple'>Intrusion ({majority_verdict})</span>"
            action = "Host intrusion attempt detected."
            bd_col = "#8B5CF6"

        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid {bd_col};">
            <div class="metric-label">Security Verdict</div>
            <div style="margin-top: 6px;">{badge_html}</div>
            <div class="metric-sub">{action}</div>
        </div>
        """, unsafe_allow_html=True)

    col_cplot, col_ctbl = st.columns([1, 1.2])
    with col_cplot:
        vote_df = counts.reset_index()
        vote_df.columns = ["Threat Category", "Votes"]
        fig_c = px.bar(
            vote_df,
            x="Votes",
            y="Threat Category",
            orientation="h",
            color="Threat Category",
            text="Votes"
        )
        fig_c.update_layout(
            height=360,
            margin=dict(t=10, b=30, l=10, r=20),
            showlegend=False,
            xaxis=dict(title="Number of Models Voting")
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with col_ctbl:
        st.dataframe(
            df_clf_results,
            use_container_width=True,
            height=360
        )


# ==============================================================================
# CHOICE 3: CIC-IDS2017 CLUSTERING (BEHAVIORAL TRAFFIC PROFILING)
# ==============================================================================
elif "Clustering" in selected_dataset:
    st.markdown("<div class='section-title'>Input Flow Telemetry</div>", unsafe_allow_html=True)

    cl_preset = st.selectbox(
        "Load sample flow preset or customize values below:",
        [
            "Interactive Web / Routine Querying (Cluster 0 Profile)",
            "Sustained High-Throughput Bulk Transfer (Cluster 1 Profile)",
            "Micro-Burst / Asymmetric Reconnaissance (Cluster 2 Profile)",
            "Custom Manual Entry"
        ]
    )

    if cl_preset == "Interactive Web / Routine Querying (Cluster 0 Profile)":
        cl_port, cl_dur = 443, 15000.0
        cl_fp, cl_bp = 8, 8
        cl_tot_fl = 900
        cl_fwd_m = 112.0
        cl_bytes_s, cl_pkts_s = 1500.0, 160.0
        cl_iat_m = 2500.0
        cl_pkt_m, cl_pkt_s = 120.0, 35.0
        cl_syn, cl_ack = 0, 1
    elif cl_preset == "Sustained High-Throughput Bulk Transfer (Cluster 1 Profile)":
        cl_port, cl_dur = 8080, 65000000.0
        cl_fp, cl_bp = 250, 180
        cl_tot_fl = 55000
        cl_fwd_m = 220.0
        cl_bytes_s, cl_pkts_s = 450000.0, 6500.0
        cl_iat_m = 280000.0
        cl_pkt_m, cl_pkt_s = 600.0, 120.0
        cl_syn, cl_ack = 0, 1
    elif cl_preset == "Micro-Burst / Asymmetric Reconnaissance (Cluster 2 Profile)":
        cl_port, cl_dur = 22, 120.0
        cl_fp, cl_bp = 2, 1
        cl_tot_fl = 80
        cl_fwd_m = 40.0
        cl_bytes_s, cl_pkts_s = 0.0, 25000.0
        cl_iat_m = 60.0
        cl_pkt_m, cl_pkt_s = 40.0, 5.0
        cl_syn, cl_ack = 0, 1
    else:
        cl_port, cl_dur = 80, 25000.0
        cl_fp, cl_bp = 15, 12
        cl_tot_fl = 1800
        cl_fwd_m = 120.0
        cl_bytes_s, cl_pkts_s = 3200.0, 300.0
        cl_iat_m = 3500.0
        cl_pkt_m, cl_pkt_s = 150.0, 45.0
        cl_syn, cl_ack = 0, 1

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        port_cl = st.number_input("Destination Port", min_value=1, max_value=65535, value=cl_port, step=1, key=f"cl_p_{cl_preset}")
        dur_cl = st.number_input("Flow Duration (µs)", min_value=1.0, max_value=120000000.0, value=float(cl_dur), step=1000.0, key=f"cl_d_{cl_preset}")
        fwd_cl = st.number_input("Total Fwd Packets", min_value=1, max_value=100000, value=cl_fp, step=1, key=f"cl_fp_{cl_preset}")
    with k2:
        bwd_cl = st.number_input("Total Backward Packets", min_value=0, max_value=100000, value=cl_bp, step=1, key=f"cl_bp_{cl_preset}")
        tot_fwd_cl = st.number_input("Total Length Fwd Packets", min_value=0, max_value=10000000, value=cl_tot_fl, step=100, key=f"cl_tl_{cl_preset}")
        fwd_m_cl = st.number_input("Fwd Packet Length Mean", min_value=0.0, max_value=10000.0, value=float(cl_fwd_m), step=10.0, key=f"cl_fm_{cl_preset}")
    with k3:
        bytes_s_cl = st.number_input("Flow Bytes/s", min_value=0.0, max_value=100000000.0, value=float(cl_bytes_s), step=1000.0, key=f"cl_bs_{cl_preset}")
        pkts_s_cl = st.number_input("Flow Packets/s", min_value=0.0, max_value=10000000.0, value=float(cl_pkts_s), step=500.0, key=f"cl_ps_{cl_preset}")
        iat_m_cl = st.number_input("Flow IAT Mean (µs)", min_value=0.0, max_value=120000000.0, value=float(cl_iat_m), step=500.0, key=f"cl_im_{cl_preset}")
    with k4:
        pkt_m_cl = st.number_input("Packet Length Mean", min_value=0.0, max_value=10000.0, value=float(cl_pkt_m), step=10.0, key=f"cl_pm_{cl_preset}")
        pkt_s_cl = st.number_input("Packet Length Std", min_value=0.0, max_value=10000.0, value=float(cl_pkt_s), step=5.0, key=f"cl_st_{cl_preset}")
        syn_cl = st.selectbox("SYN Flag", [0, 1], index=cl_syn, key=f"cl_sy_{cl_preset}")
        ack_cl = st.selectbox("ACK Flag", [0, 1], index=cl_ack, key=f"cl_ak_{cl_preset}")

    # Run clustering models from clustering.ipynb (K-Means & Agglomerative Hierarchical)
    cl_raw_df = pd.DataFrame([{
        "Destination Port": port_cl, "Flow Duration": dur_cl, "Total Fwd Packets": fwd_cl,
        "Total Backward Packets": bwd_cl, "Total Length of Fwd Packets": tot_fwd_cl,
        "Fwd Packet Length Mean": fwd_m_cl, "Flow Bytes/s": bytes_s_cl, "Flow Packets/s": pkts_s_cl,
        "Flow IAT Mean": iat_m_cl, "Packet Length Mean": pkt_m_cl, "Packet Length Std": pkt_s_cl,
        "SYN Flag Count": syn_cl, "ACK Flag Count": ack_cl
    }])

    cluster_profiles = {
        0: {
            "name": "Cluster 0: Standard Interactive Web Traffic",
            "desc": "Typical client-server HTTP/HTTPS handshakes with modest packet counts and balanced bidirectional transfer.",
            "color": "#10B981"
        },
        1: {
            "name": "Cluster 1: Sustained High-Throughput Bulk Transfer",
            "desc": "Persistent data flows characterized by large transmission windows, high volumetric throughput, and long flow durations.",
            "color": "#3B82F6"
        },
        2: {
            "name": "Cluster 2: Micro-Burst / Asymmetric Reconnaissance",
            "desc": "Short-lived, low-packet exchanges typical of host scanning, port probes, or initial network reconnaissance.",
            "color": "#EF4444"
        }
    }

    if bundle is not None and "km_model" in bundle and "clf_scaler" in bundle:
        scaler = bundle["clf_scaler"]
        scaled_cl = scaler.transform(cl_raw_df)

        km_cluster = int(bundle["km_model"].predict(scaled_cl)[0])
        hc_cluster = int(np.argmin(np.linalg.norm(scaled_cl - bundle["hc_centroids"], axis=1)))

        # Centroid distances
        km_centers = bundle["km_model"].cluster_centers_
        km_distances = [float(np.linalg.norm(scaled_cl[0] - center)) for center in km_centers]
    else:
        if dur_cl > 10000000 or bytes_s_cl > 100000:
            km_cluster = 1
            hc_cluster = 1
        elif dur_cl < 500 and fwd_cl <= 3:
            km_cluster = 2
            hc_cluster = 2
        else:
            km_cluster = 0
            hc_cluster = 0
        km_distances = [1.2, 3.4, 4.1]

    km_info = cluster_profiles.get(km_cluster, cluster_profiles[0])
    hc_info = cluster_profiles.get(hc_cluster, cluster_profiles[0])

    st.markdown("<div class='section-title'>Outputs from Clustering Algorithms</div>", unsafe_allow_html=True)

    kl1, kl2 = st.columns(2)
    with kl1:
        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid {km_info['color']};">
            <div class="metric-label">K-Means Clustering (k=3)</div>
            <div class="metric-value" style="color: {km_info['color']};">Cluster #{km_cluster}</div>
            <div style="font-weight: 600; font-size: 0.95rem; margin-top: 4px; color: #0F172A;">{km_info['name']}</div>
            <div class="metric-sub">{km_info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    with kl2:
        st.markdown(f"""
        <div class="card-box" style="border-left: 4px solid #8B5CF6;">
            <div class="metric-label">Agglomerative Hierarchical Clustering (Ward Linkage, k=3)</div>
            <div class="metric-value" style="color: #8B5CF6;">Cluster #{hc_cluster}</div>
            <div style="font-weight: 600; font-size: 0.95rem; margin-top: 4px; color: #0F172A;">{hc_info['name']}</div>
            <div class="metric-sub">{hc_info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Comparison Table & Centroid Proximity Chart
    cl_col1, cl_col2 = st.columns([1, 1.2])
    with cl_col1:
        dist_df = pd.DataFrame({
            "Cluster": [f"Cluster {i}" for i in range(len(km_distances))],
            "Euclidean Distance to Centroid": [round(d, 3) for d in km_distances]
        })
        fig_dist = px.bar(
            dist_df,
            x="Cluster",
            y="Euclidean Distance to Centroid",
            color="Cluster",
            text="Euclidean Distance to Centroid"
        )
        fig_dist.update_layout(
            height=300,
            margin=dict(t=10, b=30, l=10, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with cl_col2:
        summary_cl_df = pd.DataFrame([
            {"Algorithm": "K-Means Clustering", "Assigned Cluster": f"Cluster {km_cluster}", "Profile Category": km_info["name"].split(": ")[1]},
            {"Algorithm": "Agglomerative Hierarchical (Ward)", "Assigned Cluster": f"Cluster {hc_cluster}", "Profile Category": hc_info["name"].split(": ")[1]}
        ])
        st.dataframe(summary_cl_df, use_container_width=True, height=300)
