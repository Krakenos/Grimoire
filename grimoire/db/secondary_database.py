import base64
from collections import defaultdict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import NullPool, create_engine, text

from grimoire.core.settings import settings

if settings.secondary_database.enabled:
    secondary_db_engine = create_engine(settings.secondary_database.db_engine, poolclass=NullPool)
else:
    secondary_db_engine = None


def get_messages_from_external_db(
    message_ids: list[str], encryption_method: str, encryption_key: str
) -> list[str | None]:
    if secondary_db_engine is None:
        raise ValueError("Secondary database is None.")

    messages_content = defaultdict(lambda: [])
    swipe_indices = {}

    message_query = text("SELECT id, swipes_index FROM chat_messages WHERE id in :message_ids")
    swipes_query = text(
        "SELECT id, chat_message_id, content "
        "FROM message_swipes "
        "WHERE chat_message_id in :message_ids ORDER BY created_at"
    )

    with secondary_db_engine.connect() as conn:
        message_rows = conn.execute(message_query, {"message_ids": tuple(message_ids)})
        swipe_rows = conn.execute(swipes_query, {"message_ids": tuple(message_ids)})

        for row in swipe_rows:
            messages_content[str(row[1])].append(str(row[2]))

        for row in message_rows:
            swipe_indices[str(row[0])] = int(row[1])

    encrypted_messages = [
        messages_content[message_id][swipe_indices[message_id]] if messages_content[message_id] != [] else None
        for message_id in message_ids
    ]
    messages = decrypt_messages(encrypted_messages, encryption_key, encryption_method)
    return messages


def decrypt_messages(
    encrypted_texts: list[str | None], encryption_key: str = "", encryption_method: str = "aesgcm"
) -> list[str | None]:
    key = encryption_key.encode()
    nonce_size = 12
    aesgcm = AESGCM(key)
    decrypted_texts = []
    if encryption_method == "aesgcm":
        for encrypted_text in encrypted_texts:
            if encrypted_text is not None:
                bytes_text = base64.b64decode(encrypted_text)
                nonce = bytes_text[:nonce_size]
                encrypted_message = bytes_text[nonce_size:]
                decrypted_message = aesgcm.decrypt(nonce, encrypted_message, None)
                message = decrypted_message.decode()
                decrypted_texts.append(message)
            else:
                decrypted_texts.append(None)
    else:
        raise NotImplementedError
    return decrypted_texts
