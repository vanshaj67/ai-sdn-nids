import sys
import time
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, "/home/vanshaj/ai-sdn-nids/src/realtime")
from feature_adapter import find_reverse_flow, convert_bidirectional_flow
from alert_logger import log_event

FLOW = "/home/vanshaj/ai-sdn-nids/data/flows/flow_stats.csv"
MODEL = "/home/vanshaj/ai-sdn-nids/outputs/models/improved/improved_pipeline.joblib"

saved = joblib.load(MODEL)
model = saved["model"]
encoder = saved["label_encoder"]
columns = saved["feature_columns"]

print("=" * 60)
print(" AI-DRIVEN SDN NETWORK INTRUSION DETECTION SYSTEM")
print("=" * 60)
print("Model: Extra Trees")
print("Classes:", len(model.classes_))
print("Expected features:", len(columns))
print("Waiting for POX flows...")
print("=" * 60)

last_rows = len(pd.read_csv(FLOW))
scan_tracker = {}
SCAN_WINDOW = 10
SCAN_THRESHOLD = 20
while True:
    try:
        df = pd.read_csv(FLOW)

        if len(df) <= last_rows:
            time.sleep(2)
            continue

        new = df.iloc[last_rows:]
        last_rows = len(df)

        for _, row in new.iterrows():

            # Use TCP/UDP flows with valid ports
            if row["protocol"] not in [6, 17]:
                continue

            if pd.isna(row["src_port"]) or pd.isna(row["dst_port"]):
                continue
            # Behavioural port-scan tracking
            now = time.time()
            src_ip = str(row["src_ip"])
            dst_ip = str(row["dst_ip"])
            dst_port = int(row["dst_port"])
    
            src_port = int(row["src_port"])
            key = (src_ip, dst_ip)

            unique_ports = 0
            is_port_scan = False

            # Track likely connection-initiating flows only
            if src_port >= 1024:
                if key not in scan_tracker:
                    scan_tracker[key] = []

                scan_tracker[key] = [
                    (t, p) for t, p in scan_tracker[key]
                    if now - t <= SCAN_WINDOW
                ]

                scan_tracker[key].append((now, dst_port))

                unique_ports = len(set(p for t, p in scan_tracker[key]))
                is_port_scan = unique_ports >= SCAN_THRESHOLD    

            reverse = find_reverse_flow(df, row)
            features = convert_bidirectional_flow(row, reverse)
    
            X = np.array(
                [[features.get(c, 0.0) for c in columns]],
                dtype=float
            )
    
            X = np.nan_to_num(
                X,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
    
            prediction = model.predict(X)[0]
            label = encoder.inverse_transform([prediction])[0]
    
            probabilities = model.predict_proba(X)[0]
            confidence = float(np.max(probabilities)) * 100
    
            if is_port_scan:
                print(
                    f'{row["src_ip"]}:{int(row["src_port"])} -> '
                    f'{row["dst_ip"]}:{int(row["dst_port"])} | '
                    f'ATTACK | PORT_SCAN | Unique ports: {unique_ports}'
                )
                log_event(
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    prediction="ATTACK",
                    attack_type="PORT_SCAN",
                    severity="HIGH",
                    confidence="",
                    unique_ports=unique_ports,
                    detection_method="Behavioural",
                    action="NONE",
                    status="DETECTED",
                    reason=f"{unique_ports} unique destination ports detected within {SCAN_WINDOW} seconds",
                )

                with open(
                    "/home/vanshaj/ai-sdn-nids/outputs/logs/block_request.txt",
                    "w"
                ) as f:
                    f.write(src_ip)

                print(
                    f"AUTO-MITIGATION REQUESTED | "
                    f"Block source {src_ip} for 600 seconds"
                )

            else:
               print(
                    f'{row["src_ip"]}:{int(row["src_port"])} -> '
                    f'{row["dst_ip"]}:{int(row["dst_port"])} | '
                    f'{label} | Confidence: {confidence:.2f}%'
                )
               log_event(
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    prediction=str(label),
                    attack_type="NONE",
                    severity="LOW",
                    confidence=f"{confidence:.2f}",
                    unique_ports=0,
                    detection_method="Extra Trees",
                    action="NONE",
                    status="OBSERVED",
                    reason="ML flow classification",
                )

    except Exception as e:
        import traceback; traceback.print_exc()

        time.sleep(2)
