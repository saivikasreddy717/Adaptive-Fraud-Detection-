import json
import time
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def produce_messages(csv_file, topic='fraud_topic'):
    data = pd.read_csv(csv_file)
    total = len(data)
    print(f"Total messages to send: {total}")
    for i, (_, row) in enumerate(data.iterrows(), start=1):
        message = row.to_dict()
        try:
            producer.send(topic, message)
        except Exception as e:
            print(f"Error on message {i}: {e}")
        if i % 100 == 0:
            print(f"Sent {i} messages out of {total}")
        time.sleep(0.01)  # adjust or remove for testing
    producer.flush()
    print("All messages produced.")

if __name__ == '__main__':
    produce_messages("data/transactions.csv")
