import cv2
from ultralytics import YOLO
import requests
import time
from datetime import datetime

# 🔗 Your Discord Webhook URL
WEBHOOK_URL = "https://discord.com/api/webhooks/1395241478153965729/kRaQfdd6aWHKzWZxGpsS0JCfi0GNUQ606VIIyx5bcM2jbVncnFnq-v23njqsaMxe3Owu"

# 📦 Load YOLOv8 model
model = YOLO("yolov8n.pt")  # nano version (lightweight and fast)

# 🚀 Send message to Discord
def send_discord_message(message):
    payload = {
        "content": message,
        "username": "Jetson Person Detector"
    }
    response = requests.post(WEBHOOK_URL, json=payload)
    if response.status_code == 204:
        print("✅ Discord notified")
    else:
        print(f"❌ Failed to send: {response.status_code}, {response.text}")

# 🎥 Start video capture
def main():
    cap = cv2.VideoCapture(0)
    last_sent = 0

    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("📸 Camera is live. Looking for people...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[0]
        for cls in results.boxes.cls:
            label = results.names[int(cls)]
            if label == "person":
                now = time.time()
                if now - last_sent > 10:  # send ping at most once every 10 seconds
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    send_discord_message(f"👤 Person detected at {timestamp}")
                    last_sent = now

        # Optional: Show live feed
        cv2.imshow("Live Feed - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()