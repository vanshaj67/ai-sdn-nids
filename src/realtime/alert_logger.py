import csv
import os
from datetime import datetime

ALERT_LOG = "/home/vanshaj/ai-sdn-nids/outputs/logs/nids_alerts.csv"

HOST_NAMES = {
    "10.0.90.10": "atkext",
    "10.0.50.11": "web",
}

FIELDS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_host",
    "dst_host",
    "src_port",
    "dst_port",
    "prediction",
    "attack_type",
    "severity",
    "confidence",
    "unique_ports",
    "detection_method",
    "action",
    "status",
    "reason",
]


def log_event(
    src_ip,
    dst_ip,
    src_port,
    dst_port,
    prediction,
    attack_type="NONE",
    severity="LOW",
    confidence="",
    unique_ports=0,
    detection_method="Extra Trees",
    action="NONE",
    status="OBSERVED",
    reason="",
):
    os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)

    new_file = not os.path.exists(ALERT_LOG)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_host": HOST_NAMES.get(src_ip, "unknown"),
        "dst_host": HOST_NAMES.get(dst_ip, "unknown"),
        "src_port": src_port,
        "dst_port": dst_port,
        "prediction": prediction,
        "attack_type": attack_type,
        "severity": severity,
        "confidence": confidence,
        "unique_ports": unique_ports,
        "detection_method": detection_method,
        "action": action,
        "status": status,
        "reason": reason,
    }

    with open(ALERT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if new_file:
            writer.writeheader()

        writer.writerow(row)
