# data/generate_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_transactions(n=1000):
    # Create a start time for transactions
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    rows = []
    for i in range(1, n + 1):
        # Increment time by 1 second for each transaction (or randomize)
        current_time = start_time + timedelta(seconds=i)
        # Generate synthetic values
        amount = round(random.uniform(10, 500), 2)
        feature1 = round(random.uniform(0, 1), 2)
        feature2 = round(random.uniform(0, 5), 2)
        # Label as fraud (1) or not fraud (0); for simplicity, mark ~10% as fraud randomly
        label = 1 if random.random() < 0.1 else 0
        rows.append({
            "transaction_id": i,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "feature1": feature1,
            "feature2": feature2,
            "label": label
        })
    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    df = generate_transactions(n=10000)  # generate 10K records; adjust n as needed
    df.to_csv("data/transactions.csv", index=False)
    print("Synthetic dataset generated and saved to data/transactions.csv")
