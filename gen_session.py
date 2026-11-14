# Run ONCE on your own PC:  python gen_session.py   (log in with the 2nd account)
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
api_id = int(input("API_ID: ")); api_hash = input("API_HASH: ").strip()
with TelegramClient(StringSession(), api_id, api_hash) as c:
    print("\nSESSION_STRING (keep secret):\n" + c.session.save())
