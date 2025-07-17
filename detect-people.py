import cv2
from ultralytics import YOLO
import requests
import time
from datetime import datetime

# 🔗 Discord webhook
WEBHOOK_URL = "https://discord.com/api/webhooks/1395241478153965729/kRaQfdd6aWHKzWZxGpsS0JCfi0GNUQ606VIIyx5bcM2jbVncnFnq-v23njqsaMxe3Owu"

# 📦 Load YOLO model
model = YOLO("yolov8n.pt")  # Nano version

# 🚀 Send message + image to Discord
def send_discord_message_with_image(message, image_path):
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "content": message,
            "username": "Jetson Person Detector"
        }
        response = requests.post(WEBHOOK_URL, data=data, files=files)
    if response.status_code in (200, 204):
        print("✅ Discord notified with image")
    else:
        print(f"❌ Error sending to Discord: {response.status_code}, {response.text}")

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
                if now - last_sent > 10:  # Rate limit
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    image_path = f"person_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame)

                    message = f"👤 Person detected at {timestamp}"
                    send_discord_message_with_image(message, image_path)

                    last_sent = now

        time.sleep(0.01)

    cap.release()

if __name__ == "__main__":
    main()