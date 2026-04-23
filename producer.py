import cv2
import json
from kafka import KafkaProducer
import base64

# -------------------------
# KAFKA PRODUCER
# -------------------------
producer = KafkaProducer(
    bootstrap_servers='localhost:9092'
)

cap = cv2.VideoCapture("yolo.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = base64.b64encode(buffer).decode('utf-8')

    producer.send(
        'video-stream',
        json.dumps({"frame": frame_bytes}).encode('utf-8')
    )

    producer.flush()  # 🔥 ensures message is sent

    print("Frame sent")

cap.release()