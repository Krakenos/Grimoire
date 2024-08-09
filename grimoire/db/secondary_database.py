from collections import defaultdict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import create_engine, text

from grimoire.core.settings import settings

secondary_db_engine = None

if settings["secondary_db"]["enabled"]:
    secondary_db_engine = create_engine(settings["secondary_db"]["db_engine"])


def get_messages_from_external_db(message_ids: list[str]) -> list[str]:
    if secondary_db_engine is None:
        raise Exception("Secondary database engine not initialized")

    messages_content = defaultdict(lambda: [])
    swipe_indices = {}

    message_query = text("SELECT id, messages, swipes_index FROM chat_messages WHERE id in :message_ids")
    swipes_query = text(
        "SELECT id, chat_message_id, content "
        "FROM message_swipes "
        "WHERE chat_message_id in :message_ids ORDER BY created_at"
    )

    with secondary_db_engine.connect() as conn:
        message_rows = conn.execute(message_query, {"message_ids": message_ids})
        swipe_rows = conn.execute(swipes_query, {"message_ids": message_ids})

        for row in swipe_rows:
            messages_content[row[1]].append(row[2])

        for row in message_rows:
            swipe_indices[row[0]] = row[2]

    encrypted_messages = [messages_content[message_id][swipe_indices[message_id]] for message_id in message_ids]
    messages = decrypt_messages(encrypted_messages)
    return messages


def decrypt_messages(encrypted_texts: list[str]) -> list[str]:
    key = settings["secondary_database"]["encryption_key"]
    nonce_size = 12
    aesgcm = AESGCM(key)
    decrypted_texts = []
    for encrypted_text in encrypted_texts:
        bytes_text = str.encode(encrypted_text)
        nonce = bytes_text[:nonce_size]
        encrypted_message = bytes_text[nonce_size:]
        decrypted_message = aesgcm.decrypt(nonce, encrypted_message, None)
        message = decrypted_message.decode()
        decrypted_texts.append(message)
    return decrypted_texts
