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
import csv
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
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='plate-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# -------------------------
# STORE ALL RECORDS
# -------------------------
all_records = []

# -------------------------
# MAIN LOOP
# -------------------------
for msg in consumer:
    print("📥 Message received")

    data = msg.value

    # Decode frame
    frame_bytes = base64.b64decode(data["frame"])
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)

    for box in results[0].boxes:
        if box.conf[0] < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = frame[y1:y2, x1:x2]

        # OCR
        plate_resized = cv2.resize(plate, None, fx=4, fy=4)
        gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)

        ocr_result = reader.readtext(gray)

        for res in ocr_result:
            text = re.sub('[^A-Z0-9]', '', res[1])

            if len(text) >= 6:
                print("🚗 Detected Plate:", text)

                timestamp = int(time.time())
                date_str = time.strftime("%Y-%m-%d")
                hour_str = time.strftime("%H")

                image_file = f"{text}_{timestamp}.jpg"

                # SAVE IMAGE
                cv2.imwrite(image_file, plate)

                image_s3_key = f"bronze/{image_file}"
                image_s3_path = f"s3://{S3_BUCKET_NAME}/{image_s3_key}"

                try:
                    s3.upload_file(image_file, S3_BUCKET_NAME, image_s3_key)
                    os.remove(image_file)
                except Exception as e:
                    print("❌ Image Upload Error:", e)
                    continue

                # STORE METADATA (NOT uploading yet)
                metadata = {
                    "plate_number": text,
                    "timestamp": timestamp,
                    "image_path": image_s3_path,
                    "confidence": float(box.conf[0]),
                    "camera_id": "cam_01",
                    "date": date_str,
                    "hour": hour_str
                }

                all_records.append(metadata)

    cv2.imshow("Kafka YOLO Stream", frame)

    # STOP CONDITION (when video ends OR press q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------
# AFTER LOOP → CREATE CSV
# -------------------------
if all_records:
    csv_file = f"plates_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
        writer.writeheader()
        writer.writerows(all_records)

    # -------------------------
    # UPLOAD ONCE
    # -------------------------
    s3_key = f"bronze/{csv_file}"

    try:
        s3.upload_file(csv_file, S3_BUCKET_NAME, s3_key)
        print(f"📄 Uploaded FINAL CSV: {csv_file}")
        os.remove(csv_file)
    except Exception as e:
        print("❌ Final CSV Upload Error:", e)

cv2.destroyAllWindows()