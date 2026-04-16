import cv2
import numpy as np
import json
import base64
from kafka import KafkaConsumer
from ultralytics import YOLO
import easyocr
import re
import os
import time
import boto3
from dotenv import load_dotenv

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# -------------------------
# S3 CLIENT
# -------------------------
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# -------------------------
# MODELS
# -------------------------
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'])

# -------------------------
# KAFKA CONSUMER
# -------------------------
consumer = KafkaConsumer(
    'video-stream',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    enable_auto_commit=True
)

# -------------------------
# MAIN LOOP
# -------------------------
for msg in consumer:
    data = json.loads(msg.value)
    
    # decode frame
    frame_bytes = base64.b64decode(data["frame"])
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)

    for box in results[0].boxes:
        if box.conf[0] < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = frame[y1:y2, x1:x2]

        # -------------------------
        # OCR
        # -------------------------
        plate = cv2.resize(plate, None, fx=4, fy=4)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        ocr_result = reader.readtext(gray)

        for res in ocr_result:
            text = re.sub('[^A-Z0-9]', '', res[1])

            if len(text) >= 6:
                print("Detected Plate:", text)

                # -------------------------
                # SAVE + UPLOAD TO S3
                # -------------------------
                timestamp = int(time.time())
                file_name = f"{text}_{timestamp}.jpg"

                cv2.imwrite(file_name, plate)

                try:
                    s3.upload_file(file_name, S3_BUCKET_NAME, file_name)
                    print(f"✅ Uploaded to S3: {file_name}")

                    # delete local file
                    os.remove(file_name)

                except Exception as e:
                    print("❌ S3 Upload Error:", e)

                # -------------------------
                # DRAW ON FRAME
                # -------------------------
                cv2.putText(frame, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Kafka YOLO Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()