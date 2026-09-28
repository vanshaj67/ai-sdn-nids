import pandas as pd
import numpy as np

FLOW_CSV = "/home/vanshaj/ai-sdn-nids/data/flows/flow_stats.csv"

def safe(v, default=0.0):
    if pd.isna(v):
        return default
    return float(v)

def convert_flow(row):
    packets = safe(row["packet_count"])
    bytes_ = safe(row["byte_count"])

    duration_us = (
        safe(row["duration_sec"]) * 1_000_000
        + safe(row["duration_nsec"]) / 1_000
    )

    duration_s = duration_us / 1_000_000.0

    packet_rate = packets / duration_s if duration_s > 0 else 0.0
    byte_rate = bytes_ / duration_s if duration_s > 0 else 0.0
    avg_packet = bytes_ / packets if packets > 0 else 0.0

    features = {
        "Destination Port": safe(row["dst_port"]),
        "Flow Duration": duration_us,
        "Total Fwd Packets": packets,
        "Total Length of Fwd Packets": bytes_,
        "Flow Bytes/s": byte_rate,
        "Flow Packets/s": packet_rate,
        "Fwd Packets/s": packet_rate,
        "Fwd Packet Length Mean": avg_packet,
        "Average Packet Size": avg_packet,
        "Avg Fwd Segment Size": avg_packet,
        "Subflow Fwd Packets": packets,
        "Subflow Fwd Bytes": bytes_
    }

    return features

if __name__ == "__main__":
    df = pd.read_csv(FLOW_CSV)

    row = df.iloc[-1]
    features = convert_flow(row)

    print("\nLatest POX flow")
    print("----------------")
    print("Source:", row["src_ip"])
    print("Destination:", row["dst_ip"])
    print("Protocol:", row["protocol"])

    print("\nCICIDS-compatible features")
    print("--------------------------")

    for name, value in features.items():
        print(f"{name:30s}: {value}")

    print("\nMapped features:", len(features))
    print("Model requires: 70")
