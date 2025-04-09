# kafka_producer.py
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
    for _, row in data.iterrows():
        message = row.to_dict()
        producer.send(topic, message)
        time.sleep(0.1)  # simulate a data stream
    producer.flush()

if __name__ == '__main__':
    produce_messages('data/transactions.csv')
