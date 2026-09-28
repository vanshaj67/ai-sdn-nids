import pandas as pd

FLOW_CSV = "/home/vanshaj/ai-sdn-nids/data/flows/flow_stats.csv"


def pox_duration_to_microseconds(duration_sec, duration_nsec):
    return (duration_sec * 1_000_000) + (duration_nsec / 1_000)
if __name__ == "__main__":
    df = pd.read_csv(FLOW_CSV)

    row = df.iloc[-1]

    flow_duration = pox_duration_to_microseconds(
        row["duration_sec"],
        row["duration_nsec"]
    )

    print("POX duration_sec:", row["duration_sec"])
    print("POX duration_nsec:", row["duration_nsec"])
    print("CICIDS Flow Duration:", flow_duration)
