# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import argparse
import logging
import weakref
import time
import json

from aiohttp import web, WSCloseCode, ClientSession, FormData
import cv2
import PIL.Image
import matplotlib.pyplot as plt

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor

# --------------------
# Helper: send Discord alert
# --------------------
async def send_discord_alert(webhook_url: str, image_bytes: bytes, content: str):
    """
    Posts a message with an image to the given Discord webhook URL.
    """
    form = FormData()
    # message payload
    payload = {"content": content}
    form.add_field('payload_json', json.dumps(payload), content_type='application/json')
    form.add_field('file', image_bytes, filename='capture.jpg', content_type='image/jpeg')

    async with ClientSession() as session:
        async with session.post(webhook_url, data=form) as resp:
            logging.info(f"Discord webhook responded with status: {resp.status}")

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    # --- CLI arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("image_encode_engine", type=str,
                        help="Name/path of the image encoder engine (e.g., ONNX or TensorRT file)")
    parser.add_argument("--discord_webhook_url", type=str, required=True,
                        help="Discord webhook URL for alerts when people are detected")
    parser.add_argument("--image_quality", type=int, default=50,
                        help="JPEG quality for streamed frames (0–100)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to serve the web app on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host/interface to bind the server")
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera device index")
    parser.add_argument("--resolution", type=str, default="640x480",
                        help="Camera resolution WIDTHxHEIGHT")
    args = parser.parse_args()

    WIDTH, HEIGHT = map(int, args.resolution.split("x"))
    CAMERA_DEVICE = args.camera
    IMAGE_QUALITY = args.image_quality
    DISCORD_WEBHOOK_URL = args.discord_webhook_url

    # --- Predictor setup ---
    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            image_encoder_engine=args.image_encode_engine
        )
    )
    prompt_data = None

    # --- Utility functions ---
    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)

    # --- HTTP handlers ---
    async def handle_index_get(request: web.Request):
        return web.FileResponse("./index.html")

    async def websocket_handler(request):
        nonlocal prompt_data
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        request.app['websockets'].add(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT and msg.data.startswith("prompt:"):
                    _, prompt = msg.data.split(":", 1)
                    tree = Tree.from_prompt(prompt)
                    clip_enc = predictor.encode_clip_text(tree)
                    owl_enc = predictor.encode_owl_text(tree)
                    prompt_data = {"tree": tree,
                                   "clip_encodings": clip_enc,
                                   "owl_encodings": owl_enc}
                    logging.info(f"Prompt set: {prompt}")
        finally:
            request.app['websockets'].discard(ws)
        return ws

    async def on_shutdown(app: web.Application):
        for ws in set(app['websockets']):
            await ws.close(code=WSCloseCode.GOING_AWAY,
                           message='Server shutdown')

    # --- Detection loop ---
    async def detection_loop(app: web.Application):
        loop = asyncio.get_running_loop()
        camera = cv2.VideoCapture(CAMERA_DEVICE)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        last_alert_time = 0
        ALERT_COOLDOWN = 30.0  # seconds between alerts

        def _read_and_encode():
            ok, frame = camera.read()
            if not ok:
                return ok, None, []

            image_pil = cv2_to_pil(frame)
            detections = []

            if prompt_data is not None:
                detections = predictor.predict(
                    image_pil,
                    tree=prompt_data['tree'],
                    clip_text_encodings=prompt_data['clip_encodings'],
                    owl_text_encodings=prompt_data['owl_encodings']
                )
                frame = draw_tree_output(frame, detections, prompt_data['tree'])

            # encode to JPEG
            jpeg = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])[1]
            return ok, bytes(jpeg), detections

        while True:
            ok, jpeg_bytes, detections = await loop.run_in_executor(None, _read_and_encode)
            if not ok:
                break

            # Check for any person detections
            now = time.time()
            if DISCORD_WEBHOOK_URL and any(d.label.lower() == 'person' for d in detections):
                if now - last_alert_time > ALERT_COOLDOWN:
                    last_alert_time = now
                    # create an async task so notification does not block streaming
                    asyncio.create_task(
                        send_discord_alert(
                            DISCORD_WEBHOOK_URL,
                            jpeg_bytes,
                            "@here Person detected!"
                        )
                    )

            # Broadcast frame to clients
            for ws in app['websockets']:
                await ws.send_bytes(jpeg_bytes)

        camera.release()

    async def run_detection_loop(app: web.Application):
        task = asyncio.create_task(detection_loop(app))
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # --- App setup & run ---
    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app['websockets'] = weakref.WeakSet()
    app.router.add_get('/', handle_index_get)
    app.router.add_route('GET', '/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)
    web.run_app(app, host=args.host, port=args.port)