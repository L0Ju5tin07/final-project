#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import time
import cv2
from PIL import Image
import requests

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_drawing import draw_tree_output

def cv2_to_pil(image):
    """
    Convert an OpenCV BGR image to a PIL Image in RGB format.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def notify_discord(webhook_url: str, image_bytes: bytes):
    """
    Send a notification to the given Discord webhook URL with an image attachment.
    """
    data = {"content": "🚨 Person detected! 📷"}
    files = {"file": ("detection.jpg", image_bytes, "image/jpeg")}
    try:
        resp = requests.post(webhook_url, data=data, files=files)
        resp.raise_for_status()
        logging.info("Discord notification sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send Discord notification: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Jetson Nano/Orin live person detector with Discord alerts"
    )
    parser.add_argument(
        "image_encode_engine", type=str,
        help="Path or name of the ONNX/TensorRT engine for image encoding"
    )
    parser.add_argument(
        "--discord_webhook_url", type=str, required=True,
        help="Discord webhook URL to send alerts to"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera device index (default: 0)"
    )
    parser.add_argument(
        "--resolution", type=str, default="640x480",
        help="Camera resolution WIDTHxHEIGHT (default: 640x480)"
    )
    parser.add_argument(
        "--image_quality", type=int, default=50,
        help="JPEG quality for sent images (0-100, default: 50)"
    )
    parser.add_argument(
        "--notify_interval", type=int, default=60,
        help="Minimum seconds between successive Discord alerts (default: 60s)"
    )
    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split('x'))

    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing predictor and person prompt...")

    # Initialize the model pipeline
    predictor = TreePredictor(
        owl_predictor=OwlPredictor(image_encoder_engine=args.image_encode_engine)
    )

    # Prepare a single "person" prompt for continuous detection
    tree = Tree.from_prompt("person")
    clip_encodings = predictor.encode_clip_text(tree)
    owl_encodings = predictor.encode_owl_text(tree)

    # Open camera
    logging.info(f"Opening camera device {args.camera} at {width}x{height}...")
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    last_notification = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logging.warning("Failed to read frame from camera; exiting.")
                break

            # Perform prediction
            result = predictor.predict(
                cv2_to_pil(frame),
                tree=tree,
                clip_text_encodings=clip_encodings,
                owl_text_encodings=owl_encodings
            )

            # Collect all detections (result.detections is already a dict_values of TreeDetection)
            all_detections = list(result.detections)

            # If any detection and interval elapsed, notify
            if all_detections and (time.time() - last_notification) > args.notify_interval:
                # Draw bounding boxes for detections
                annotated = draw_tree_output(frame.copy(), result, tree)
                success, buf = cv2.imencode(
                    '.jpg', annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, args.image_quality]
                )
                if success:
                    notify_discord(args.discord_webhook_url, buf.tobytes())
                    last_notification = time.time()
                else:
                    logging.error("Failed to JPEG-encode annotated image.")
            else:
                annotated = frame

            # Optional: display locally; quit on 'q'
            cv2.imshow("Person Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User requested exit.")
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user; shutting down.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
