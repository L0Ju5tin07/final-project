# Jetson NanoOWL Person Detector with Discord Alerts
![Jetson Powered](https://img.shields.io/badge/Jetson-Orin%20Nano-blue?logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
[Link to demo]https://youtu.be/042-LH_Pyd8 

## Overview

This script leverages NanoOWL on a Jetson Nano/Orin to perform continuous, real-time person detection via a connected camera. When a person is detected with confidence above a configurable threshold, the script:

1. Draws bounding boxes around detected persons.
2. Sends an image alert to a specified Discord webhook (once per notification interval).
3. (Optionally) Displays the live annotated video on a local GUI window.

## Features

* **Continuous Operation**: Runs until manually stopped (Ctrl+C or pressing `q` in display mode).
* **Configurable Threshold**: Only notifies if detection confidence ≥ `--score_threshold`.
* **Cooldown Interval**: Avoids spamming Discord by waiting `--notify_interval` seconds between alerts.
* **Flexible Output**: Optional GUI display via `--display` or headless operation.

## Prerequisites

* **Hardware**: NVIDIA Jetson Nano or Orin with a USB or CSI camera attached.
* **OS**: Linux (Ubuntu 18.04 or later recommended).
* **CUDA**: JetPack with TensorRT support for optimized inference.

## Software Dependencies

Install the following Python packages (preferably in a `venv`):

```
pip3 install nanoowl torch torchvision
pip3 install opencv-python pillow requests
```

Ensure you have your trained or downloaded NanoOWL engine file (e.g., `owl_image_encoder.engine`) accessible.

## Setup Instructions

1. **Clone this repository** to your Jetson device:

   ```bash
   git clone <your-repo-url>
   cd final-project
   ```

2. **Prepare your model engine** (TensorRT or ONNX):

   * Place the `.engine` or `.onnx` file in the project folder or note its path.

3. **Obtain a Discord Webhook URL**:

   * In your Discord server, go to **Server Settings → Integrations → Webhooks → New Webhook**.
   * Copy the webhook URL for use in the next step.

4. **Install Python dependencies**:

   ```bash
   pip3 install -r requirements.txt
   ```

   (Or install packages individually as above.)

## Usage

Run the detector script with:

```bash
python3 detect-people.py /path/to/owl_image_encoder.engine \
  --discord_webhook_url YOUR_WEBHOOK_URL \
  [--camera 0] \
  [--resolution WIDTHxHEIGHT] \
  [--image_quality 0-100] \
  [--score_threshold 0.0-1.0] \
  [--notify_interval seconds] \
  [--display]
```

* **`/path/to/owl_image_encoder.engine`**: Path to your NanoOWL engine file.
* **`--discord_webhook_url`**: (Required) Your Discord webhook endpoint.
* **`--camera`**: (Default `0`) Device index of your camera.
* **`--resolution`**: (Default `640x480`) Capture size, e.g., `1280x720`.
* **`--image_quality`**: (Default `50`) JPEG quality for the sent snapshot.
* **`--score_threshold`**: (Default `0.5`) Minimum detection confidence for alerts.
* **`--notify_interval`**: (Default `60`) Seconds to wait between notifications.
* **`--display`**: (Flag) Enable GUI window (requires X11/Wayland).

## Example

```bash
python3 detect-people.py owl_image_encoder_patch32.engine \
  --discord_webhook_url https://discord.com/api/webhooks/... \
  --resolution 1280x720 \
  --score_threshold 0.75 \
  --notify_interval 30 \
  --display
```

This will run detection at 1280×720, notify only if confidence ≥ 75%, send alerts at most every 30 s, and show the GUI window.

## Stopping

* To **exit** headless mode: Press **Ctrl+C** in the terminal.
* To **exit** GUI mode: Click the window and press **q**.

## Customization

* Modify `TREE.from_prompt("person")` in the code to detect other objects (e.g., "car", "cat").
* Tweak the confidence threshold and interval to suit your environment.

## Troubleshooting

* **No detections**: Lower `--score_threshold` or test with a clear view of a person.
* **Display errors**: Remove `--display` in headless (non-GUI) setups.
* **Slow performance**: Use a lower resolution or optimize your TensorRT engine.

## License

Apache License 2.0 (see header in `detect-people.py`)
