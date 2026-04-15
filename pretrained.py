import cv2
import numpy as np
import re
from ultralytics import YOLO
import easyocr

# load models
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture("video1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for box in results[0].boxes:

        # confidence filter
        if box.conf[0] < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        plate = frame[y1:y2, x1:x2]

        # -------------------------
        # 🔥 IMAGE PREPROCESSING
        # -------------------------
        plate = cv2.resize(plate, None, fx=4, fy=4)

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # sharpen
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpen = cv2.filter2D(gray, -1, kernel)

        # adaptive threshold
        thresh = cv2.adaptiveThreshold(
            sharpen, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # -------------------------
        # 🔤 OCR
        # -------------------------
        ocr_result = reader.readtext(
            thresh,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        for res in ocr_result:
            text = res[1]

            # clean text
            text = re.sub('[^A-Z0-9]', '', text)

            # fix common OCR mistakes
            text = text.replace('O', '0')
            text = text.replace('I', '1')
            text = text.replace('S', '5')

            if len(text) >= 6:
                print("Plate:", text)

                # draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # draw text
                cv2.putText(frame, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Final Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()