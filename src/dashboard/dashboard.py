import os
import time
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = "/home/vanshaj/ai-sdn-nids"

ALERT_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs/logs/nids_alerts.csv"
)

FLOW_FILE = os.path.join(
    PROJECT_ROOT,
    "data/flows/flow_stats.csv"
)

st.set_page_config(
    page_title="AI-Driven SDN Intrusion Detection System for Real-Time Network Security",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------
# HOST INFORMATION
# More hosts will be added after we confirm their IPs
# ---------------------------------------------------------

HOST_NAMES = {
    "10.0.90.10": "atkext",
    "10.0.50.11": "web",
}

HOST_ICONS = {
    "atkext": "🧑‍💻",
    "web": "🌐",
    "unknown": "💻",
}


def host_name(ip):
    return HOST_NAMES.get(str(ip), "unknown")


def host_icon(ip):
    name = host_name(ip)
    return HOST_ICONS.get(name, "💻")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_alerts():
    try:
        if os.path.exists(ALERT_FILE):
            return pd.read_csv(ALERT_FILE)
    except Exception:
        pass

    return pd.DataFrame()


def load_flows():
    try:
        if os.path.exists(FLOW_FILE):
            return pd.read_csv(FLOW_FILE)
    except Exception:
        pass

    return pd.DataFrame()


# ---------------------------------------------------------
# PAGE DESIGN
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 800;
        text-align: center;
        padding-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .live-indicator {
        background-color: #123d2a;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 8px 14px;
        display: inline-block;
        font-weight: bold;
    }

    .attack-box {
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 18px;
        background-color: rgba(239,68,68,0.08);
        margin-bottom: 12px;
    }

    .safe-box {
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 18px;
        background-color: rgba(34,197,94,0.08);
    }

    .host-box {
        border: 1px solid #475569;
        border-radius: 12px;
        text-align: center;
        padding: 18px;
        margin: 5px;
        background-color: rgba(30,41,59,0.35);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LIVE DASHBOARD
# ---------------------------------------------------------

@st.fragment(run_every="2s")
def live_dashboard():

    alerts = load_alerts()
    flows = load_flows()

    st.markdown(
        """
        <div class="main-title">
        🛡️ AI-Driven SDN Intrusion Detection System for Real-Time Network Security
        </div>

        <div class="sub-title">
        AI/ML-Assisted SDN Network Monitoring, Intrusion Detection and Automated Response
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="live-indicator">● LIVE MONITORING</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # -----------------------------------------------------
    # SUMMARY COUNTERS
    # -----------------------------------------------------

    total_events = len(alerts)

    if not alerts.empty and "prediction" in alerts.columns:
        attack_count = len(
            alerts[alerts["prediction"].astype(str).str.upper() == "ATTACK"]
        )

        benign_count = len(
            alerts[alerts["prediction"].astype(str).str.upper() == "BENIGN"]
        )
    else:
        attack_count = 0
        benign_count = 0

    if not flows.empty:
        ips = set()

        if "src_ip" in flows.columns:
            ips.update(flows["src_ip"].dropna().astype(str))

        if "dst_ip" in flows.columns:
            ips.update(flows["dst_ip"].dropna().astype(str))

        active_hosts = len(ips)
        total_flows = len(flows)

    else:
        active_hosts = 0
        total_flows = 0

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("📡 Total Flows", total_flows)
    c2.metric("📝 NIDS Events", total_events)
    c3.metric("✅ Benign", benign_count)
    c4.metric("🚨 Attacks", attack_count)
    c5.metric("🖥️ Active Hosts", active_hosts)

    st.divider()
    # -----------------------------------------------------
    # SDN NETWORK TOPOLOGY
    # -----------------------------------------------------

    st.subheader("🌐 SDN Network Topology")

    st.markdown(
        """
        <style>
        .topology-controller {
            text-align:center;
            border:2px solid #3b82f6;
            border-radius:12px;
            padding:14px;
            margin:10px auto;
            max-width:340px;
            background:rgba(59,130,246,0.10);
        }

        .topology-core {
            text-align:center;
            border:2px solid #8b5cf6;
            border-radius:12px;
            padding:14px;
            margin:10px auto;
            max-width:240px;
            background:rgba(139,92,246,0.10);
        }

        .topology-switch {
            text-align:center;
            border:1px solid #64748b;
            border-radius:10px;
            padding:10px;
            margin-bottom:8px;
            font-weight:bold;
        }

        .topology-host {
            border:1px solid #475569;
            border-radius:8px;
            padding:8px;
            margin:5px 0;
            text-align:center;
            font-size:14px;
        }

        .topology-attacker {
            border:2px solid #ef4444;
            background:rgba(239,68,68,0.08);
        }

        .topology-link {
            text-align:center;
            font-size:26px;
            margin:2px;
        }
        </style>

        <div class="topology-controller">
            <b>🧠 POX Controller</b><br>
            c0 — 127.0.0.1:6633<br>
            <small>OpenFlow Controller</small>
        </div>

        <div class="topology-link">↓</div>

        <div class="topology-core">
            <b>🔀 Core Switch s1</b><br>
            Open vSwitch
        </div>

        <div class="topology-link">↓</div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            """
            <div class="topology-switch">🔀 s2<br>HR Network</div>

            <div class="topology-host">
            👤 <b>hr1</b><br>
            10.0.10.11
            </div>

            <div class="topology-host">
            👤 <b>hr2</b><br>
            10.0.10.12
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="topology-switch">🔀 s3<br>Finance Network</div>

            <div class="topology-host">
            💼 <b>fin1</b><br>
            10.0.20.11
            </div>

            <div class="topology-host">
            💼 <b>fin2</b><br>
            10.0.20.12
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="topology-switch">🔀 s4<br>IT Network</div>

            <div class="topology-host">
            💻 <b>it1</b><br>
            10.0.30.11
            </div>

            <div class="topology-host">
            💻 <b>it2</b><br>
            10.0.30.12
            </div>

            <div class="topology-host topology-attacker">
            ⚠️ <b>atkint</b><br>
            10.0.90.20<br>
            <small>Internal Test Attacker</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="topology-switch">🔀 s5<br>IoT Network</div>

            <div class="topology-host">
            📷 <b>camera</b><br>
            10.0.40.11
            </div>

            <div class="topology-host">
            🌡️ <b>sensor</b><br>
            10.0.40.12
            </div>

            <div class="topology-host">
            🚪 <b>doorlock</b><br>
            10.0.40.13
            </div>

            <div class="topology-host">
            🍓 <b>rpi</b><br>
            10.0.40.14
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
            <div class="topology-switch">🔀 s6<br>Server Network</div>

            <div class="topology-host">
            🌐 <b>web</b><br>
            10.0.50.11
            </div>

            <div class="topology-host">
            📡 <b>dns</b><br>
            10.0.50.12
            </div>

            <div class="topology-host">
            📁 <b>ftp</b><br>
            10.0.50.13
            </div>

            <div class="topology-host">
            🗄️ <b>database</b><br>
            10.0.50.14
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="topology-link">↑</div>

        <div class="topology-controller topology-attacker">
            🧑‍💻 <b>atkext</b><br>
            10.0.90.10<br>
            <small>External Test Attacker — connected to s1</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Topology reflects the active Mininet SDN environment controlled by POX through OpenFlow."
    )

    st.divider()
    # -----------------------------------------------------
    # CURRENT SECURITY STATUS
    # -----------------------------------------------------
    
    st.subheader("🚨 Live Security Status")

    attacks = pd.DataFrame()

    if not alerts.empty and "prediction" in alerts.columns:
        attacks = alerts[
            alerts["prediction"].astype(str).str.upper() == "ATTACK"
        ]

    if not attacks.empty:

        latest = attacks.iloc[-1]

        src_ip = str(latest.get("src_ip", "Unknown"))
        dst_ip = str(latest.get("dst_ip", "Unknown"))

        src_host = str(latest.get("src_host", host_name(src_ip)))
        dst_host = str(latest.get("dst_host", host_name(dst_ip)))

        attack_type = str(latest.get("attack_type", "UNKNOWN"))
        severity = str(latest.get("severity", "HIGH"))
        status = str(latest.get("status", "DETECTED"))
        action = str(latest.get("action", "NONE"))
        reason = str(latest.get("reason", ""))

        st.markdown(
            f"""
            <div class="attack-box">

            <h3>🚨 {severity} SECURITY ALERT</h3>

            <b>Threat:</b> {attack_type}<br><br>

            <b>Source:</b>
            🧑‍💻 {src_host} ({src_ip})

            <br>

            <b>Target:</b>
            🌐 {dst_host} ({dst_ip})

            <br><br>

            <b>Detection Status:</b> {status}<br>
            <b>Automatic Response:</b> {action}<br>
            <b>Reason:</b> {reason}

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="safe-box">
            <h3>✅ NETWORK STATUS: NORMAL</h3>
            No active attack has been recorded.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -----------------------------------------------------
    # LIVE SOURCE → DESTINATION
    # -----------------------------------------------------

    st.subheader("🎯 Live Incident Path")

    if not attacks.empty:

        latest = attacks.iloc[-1]

        src_ip = str(latest.get("src_ip", "Unknown"))
        dst_ip = str(latest.get("dst_ip", "Unknown"))

        src_host = str(latest.get("src_host", host_name(src_ip)))
        dst_host = str(latest.get("dst_host", host_name(dst_ip)))

        attack_type = str(latest.get("attack_type", "ATTACK"))

        left, middle, right = st.columns([2, 1, 2])

        with left:
            st.markdown(
                f"""
                <div class="host-box">
                <h1>🧑‍💻</h1>
                <h3>{src_host}</h3>
                <b>{src_ip}</b>
                <br>
                Source Host
                </div>
                """,
                unsafe_allow_html=True,
            )

        with middle:
            st.markdown(
                f"""
                <div style="text-align:center;padding-top:45px">
                <h2>➡️</h2>
                <b>🚨 {attack_type}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                f"""
                <div class="host-box">
                <h1>🌐</h1>
                <h3>{dst_host}</h3>
                <b>{dst_ip}</b>
                <br>
                Target Host
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.info("Waiting for security incidents...")

    st.divider()

    # -----------------------------------------------------
    # TRAFFIC BETWEEN HOSTS
    # -----------------------------------------------------

    st.subheader("🌐 Live Traffic Between Hosts")

    if not flows.empty:

        wanted = [
            "src_ip",
            "dst_ip",
            "protocol",
            "src_port",
            "dst_port",
            "packet_count",
            "byte_count",
        ]

        existing = [c for c in wanted if c in flows.columns]

        traffic = flows[existing].tail(25).copy()

        if "src_ip" in traffic.columns:
            traffic.insert(
                0,
                "Source Host",
                traffic["src_ip"].astype(str).map(
                    lambda x: f"{host_icon(x)} {host_name(x)}"
                ),
            )

        if "dst_ip" in traffic.columns:
            traffic.insert(
                min(2, len(traffic.columns)),
                "Destination Host",
                traffic["dst_ip"].astype(str).map(
                    lambda x: f"{host_icon(x)} {host_name(x)}"
                ),
            )

        st.dataframe(
            traffic.iloc[::-1],
            width="stretch",
            hide_index=True,
        )

    else:
        st.warning("Waiting for POX flow statistics...")

    st.divider()

    # -----------------------------------------------------
    # TRAFFIC GRAPH
    # -----------------------------------------------------

    st.subheader("📈 Live Network Traffic")

    if not flows.empty and "byte_count" in flows.columns:

        graph_data = flows.tail(100).copy()

        graph_data["byte_count"] = pd.to_numeric(
            graph_data["byte_count"],
            errors="coerce",
        ).fillna(0)

        graph_data = graph_data.reset_index(drop=True)

        st.line_chart(
            graph_data["byte_count"],
            height=300,
        )

    else:
        st.info("Traffic graph will appear when flow data is available.")

    st.divider()

    # -----------------------------------------------------
    # ATTACK TIMELINE
    # -----------------------------------------------------

    st.subheader("⚠️ Attack Timeline")

    if not attacks.empty:

        timeline = attacks.copy()

        timeline["timestamp"] = pd.to_datetime(
            timeline["timestamp"],
            errors="coerce",
        )

        attack_timeline = (
            timeline.dropna(subset=["timestamp"])
            .set_index("timestamp")
            .resample("10s")
            .size()
        )

        st.bar_chart(
            attack_timeline,
            height=280,
        )

    else:
        st.info("No attack events available for timeline.")

    st.divider()

    # -----------------------------------------------------
    # RECENT SECURITY INCIDENTS
    # -----------------------------------------------------

    st.subheader("📋 Recent Security Incidents")

    if not attacks.empty:

        incident_columns = [
            "timestamp",
            "src_host",
            "src_ip",
            "dst_host",
            "dst_ip",
            "attack_type",
            "severity",
            "unique_ports",
            "detection_method",
            "action",
            "status",
            "reason",
        ]

        existing = [
            col for col in incident_columns
            if col in attacks.columns
        ]

        st.dataframe(
            attacks[existing].tail(20).iloc[::-1],
            width="stretch",
            hide_index=True,
        )

    else:
        st.success("No security incidents recorded.")

    st.divider()

    # -----------------------------------------------------
    # RECENT ALL NIDS EVENTS
    # -----------------------------------------------------

    st.subheader("🔎 Recent NIDS Detections")

    if not alerts.empty:

        display_columns = [
            "timestamp",
            "src_host",
            "src_ip",
            "dst_host",
            "dst_ip",
            "prediction",
            "attack_type",
            "severity",
            "confidence",
            "status",
        ]

        existing = [
            c for c in display_columns
            if c in alerts.columns
        ]

        st.dataframe(
            alerts[existing].tail(30).iloc[::-1],
            width="stretch",
            hide_index=True,
        )

    else:
        st.info("Waiting for NIDS events...")


live_dashboard()
