import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
api_key_sid = os.getenv("TWILIO_API_KEY_SID")
api_key_secret = os.getenv("TWILIO_API_KEY_SECRET")
from_number = os.getenv("TWILIO_PHONE_NUMBER")

to_number = "+918889818291"

client = Client(
    api_key_sid,
    api_key_secret,
    account_sid=account_sid,
)

message = client.messages.create(
    body="sms_appointment_reminders",
    from_=from_number,
    to=to_number,
)

print("SMS sent successfully!")
print("Message SID:", message.sid)
print("Status:", message.status)
