from flask import Flask, request
import requests, os, threading
from main import ask_user  
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
VERIFY_TOKEN = "my_verify_token"

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Webhook verification (done once on Meta dashboard)"""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print(" Webhook verified successfully!")
        return challenge, 200
    else:
        return "Verification failed", 403

def process_webhook(data):
    print("Received webhook:", data)

    if not data or "entry" not in data:
        return

    for entry in data["entry"]:
        for change in entry.get("changes", []):
            value = change.get("value", {})

            # 🔥 Ignore delivery/read receipts (prevents duplicates)
            if "messages" not in value:
                continue

            messages = value.get("messages", [])
            for message in messages:

                # Only process text messages
                if message.get("type") != "text":
                    continue  

                phone_number_id = value["metadata"]["phone_number_id"]
                from_number = message["from"]
                msg_text = message["text"]["body"]

                print(f"Message from {from_number}: {msg_text}")

                # Generate bot response
                try:
                    reply_text = ask_user(msg_text)
                except Exception as e:
                    print("Chatbot error:", e)
                    reply_text = "Sorry, something went wrong."

                send_whatsapp_message(phone_number_id, from_number, reply_text)

@app.route('/webhook', methods=['POST'])
def receive_message():
    """Receive WhatsApp messages (asynchronous, avoids double-processing)"""
    data = request.get_json()

    # Return 200 OK immediately (prevents WhatsApp retry duplicates)
    threading.Thread(target=process_webhook, args=(data,)).start()
    return "EVENT_RECEIVED", 200

def send_whatsapp_message(phone_number_id, to, message):
    """Send reply back to WhatsApp user"""
    url = f"https://graph.facebook.com/v24.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }
    response = requests.post(url, headers=headers, json=data)
    print(" Sent message:", response.text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)