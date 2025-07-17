import time
import requests
from datetime import datetime

WEBHOOK_URL = https://discord.com/api/webhooks/1395241478153965729/kRaQfdd6aWHKzWZxGpsS0JCfi0GNUQ606VIIyx5bcM2jbVncnFnq-v23njqsaMxe3Owu

def send_discord_message():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payload = {
        "content": f"🔔 LLaVA Ping from Jetson Orin Nano at {now}",
        "username": "Jetson LLaVA Notifier"
    }
    try:
        response = requests.post(WEBHOOK_URL, json=payload)
        if response.status_code == 204:
            print(f"✅ Sent at {now}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Exception occurred: {e}")

if __name__ == "__main__":
    while True:
        send_discord_message()
        time.sleep(60)