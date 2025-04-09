# kafka_consumer.py
import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'fraud_topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

def consume_messages():
    for message in consumer:
        data = message.value
        # Here, you might ingest data into Spark/Delta Lake.
        print("Received message:", data)
        # (For demonstration, we just print out the message.)

if __name__ == '__main__':
    consume_messages()
