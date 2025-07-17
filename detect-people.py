import cv2
import os
import time
import requests
from datetime import datetime
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import TreePredictor, Tree
from nanoowl.tree_drawing import draw_tree_output

# 🔗 Discord webhook
WEBHOOK_URL = "https://discord.com/api/webhooks/1395241478153965729/kRaQfdd6aWHKzWZxGpsS0JCfi0GNUQ606VIIyx5bcM2jbVncnFnq-v23njqsaMxe3Owu"

# 🧠 Person detection prompt
PROMPT = "a person"
THRESHOLD = 0.2  # adjust as needed

# 🧠 NanoOWL setup
predictor = TreePredictor(
    owl_predictor=OwlPredictor(
        model_id="google/owlvit-base-patch32",
        image_encoder_engine="../data/owl_image_encoder_patch32.engine"
    )
)
tree = Tree.from_prompt(PROMPT)
clip_text_encodings = predictor.encode_clip_text(tree)
owl_text_encodings = predictor.encode_owl_text(tree)

def send_discord_message_with_image(message, image_path):
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"content": message, "username": "Jetson NanoOWL Detector"}
        response = requests.post(WEBHOOK_URL, data=data, files=files)
    if response.status_code in (200, 204):
        print("✅ Discord notified with image")
    else:
        print(f"❌ Failed to send: {response.status_code}, {response.text}")

def main():
    cap = cv2.VideoCapture(0)
    last_sent = 0

    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("📸 Camera is live. Looking for people using NanoOWL...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR (cv2) → RGB (Pillow)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        output = predictor.predict(
            image=pil_image,
            tree=tree,
            clip_text_encodings=clip_text_encodings,
            owl_text_encodings=owl_text_encodings,
            threshold=THRESHOLD
        )

        if output.boxes:  # If NanoOWL found anything
            now = time.time()
            if now - last_sent > 10:  # 10 second cooldown
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_path = f"nanoowl_person_{timestamp}.jpg"

                result_image = draw_tree_output(pil_image, output, tree=tree, draw_text=True)
                result_image.save(image_path)

                msg = f"👤 Person detected at {timestamp} (NanoOWL)"
                send_discord_message_with_image(msg, image_path)

                os.remove(image_path)
                last_sent = now

        time.sleep(0.01)

    cap.release()

if __name__ == "__main__":
    main()