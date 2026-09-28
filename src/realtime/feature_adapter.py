import pandas as pd
import numpy as np

FLOW_CSV = "/home/vanshaj/ai-sdn-nids/data/flows/flow_stats.csv"


def safe(v, default=0.0):
    if pd.isna(v):
        return default
    return float(v)


def duration_us(row):
    return (
        safe(row["duration_sec"]) * 1_000_000
        + safe(row["duration_nsec"]) / 1_000
    )


def find_reverse_flow(df, row):
    if row["protocol"] not in [6, 17]:
        return None

    if pd.isna(row["src_port"]) or pd.isna(row["dst_port"]):
        return None

    reverse = df[
        (df["src_ip"] == row["dst_ip"]) &
        (df["dst_ip"] == row["src_ip"]) &
        (df["protocol"] == row["protocol"]) &
        (df["src_port"] == row["dst_port"]) &
        (df["dst_port"] == row["src_port"])
    ]

    if reverse.empty:
        return None

    return reverse.iloc[-1]


def convert_bidirectional_flow(row, reverse=None):
    fwd_packets = safe(row["packet_count"])
    fwd_bytes = safe(row["byte_count"])

    if reverse is not None:
        bwd_packets = safe(reverse["packet_count"])
        bwd_bytes = safe(reverse["byte_count"])
        reverse_duration = duration_us(reverse)
    else:
        bwd_packets = 0.0
        bwd_bytes = 0.0
        reverse_duration = 0.0

    fwd_duration = duration_us(row)
    flow_duration = max(fwd_duration, reverse_duration)

    duration_seconds = flow_duration / 1_000_000.0

    total_packets = fwd_packets + bwd_packets
    total_bytes = fwd_bytes + bwd_bytes

    fwd_mean = fwd_bytes / fwd_packets if fwd_packets > 0 else 0.0
    bwd_mean = bwd_bytes / bwd_packets if bwd_packets > 0 else 0.0
    avg_packet = total_bytes / total_packets if total_packets > 0 else 0.0

    flow_packets_s = total_packets / duration_seconds if duration_seconds > 0 else 0.0
    flow_bytes_s = total_bytes / duration_seconds if duration_seconds > 0 else 0.0
    fwd_packets_s = fwd_packets / duration_seconds if duration_seconds > 0 else 0.0
    bwd_packets_s = bwd_packets / duration_seconds if duration_seconds > 0 else 0.0

    features = {
        "Destination Port": safe(row["dst_port"]),
        "Flow Duration": flow_duration,

        "Total Fwd Packets": fwd_packets,
        "Total Backward Packets": bwd_packets,

        "Total Length of Fwd Packets": fwd_bytes,
        "Total Length of Bwd Packets": bwd_bytes,

        "Fwd Packet Length Max": fwd_mean,
        "Fwd Packet Length Min": fwd_mean,
        "Fwd Packet Length Mean": fwd_mean,
        "Fwd Packet Length Std": 0.0,

        "Bwd Packet Length Max": bwd_mean,
        "Bwd Packet Length Min": bwd_mean,
        "Bwd Packet Length Mean": bwd_mean,
        "Bwd Packet Length Std": 0.0,

        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_packets_s,

        "Fwd Packets/s": fwd_packets_s,
        "Bwd Packets/s": bwd_packets_s,

        "Min Packet Length": min(
            [x for x in [fwd_mean, bwd_mean] if x > 0],
            default=0.0
        ),

        "Max Packet Length": max(fwd_mean, bwd_mean),

        "Packet Length Mean": avg_packet,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,

        "Down/Up Ratio": (
            bwd_packets / fwd_packets
            if fwd_packets > 0 else 0.0
        ),

        "Average Packet Size": avg_packet,

        "Avg Fwd Segment Size": fwd_mean,
        "Avg Bwd Segment Size": bwd_mean,

        "Subflow Fwd Packets": fwd_packets,
        "Subflow Fwd Bytes": fwd_bytes,
        "Subflow Bwd Packets": bwd_packets,
        "Subflow Bwd Bytes": bwd_bytes,
    }

    return features


if __name__ == "__main__":
    df = pd.read_csv(FLOW_CSV)

    row = df.iloc[-1]
    reverse = find_reverse_flow(df, row)

    features = convert_bidirectional_flow(row, reverse)

    print("\nLatest POX flow")
    print("----------------")
    print("Source:", row["src_ip"])
    print("Destination:", row["dst_ip"])
    print("Protocol:", row["protocol"])

    if reverse is not None:
        print("Reverse flow: FOUND")
        print(
            f'{reverse["src_ip"]}:{reverse["src_port"]} -> '
            f'{reverse["dst_ip"]}:{reverse["dst_port"]}'
        )
    else:
        print("Reverse flow: NOT FOUND")

    print("\nCICIDS-compatible bidirectional features")
    print("----------------------------------------")

    for name, value in features.items():
        print(f"{name:30}: {value}")

    print("\nMapped features:", len(features))
