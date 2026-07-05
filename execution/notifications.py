import os
import json
import firebase_admin
from firebase_admin import credentials, messaging

def send_push_notification(title: str, body: str) -> None:
    device_token = os.environ.get("FCM_DEVICE_TOKEN")
    service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    local_creds_path = "firebase-service-account.json"
    
    if not device_token:
        print("ℹ️ FCM_DEVICE_TOKEN not set. Skipping push notification.")
        return

    try:
        if not firebase_admin._apps:
            if service_account_json:
                cred_info = json.loads(service_account_json)
                cred = credentials.Certificate(cred_info)
                firebase_admin.initialize_app(cred)
            elif os.path.exists(local_creds_path):
                cred = credentials.Certificate(local_creds_path)
                firebase_admin.initialize_app(cred)
            else:
                print("ℹ️ No Firebase credentials found (FIREBASE_SERVICE_ACCOUNT or firebase-service-account.json). Skipping push.")
                return

        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            token=device_token,
        )

        response = messaging.send(message)
        print(f"🚀 Push notification sent successfully! Message ID: {response}")

    except Exception as e:
        print(f"⚠️ Failed to send Firebase push notification: {e}")
