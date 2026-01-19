import jwt
import random
import string
from datetime import datetime, timedelta, timezone

# pip install PyJWT
SECRET_KEY = "V3fryS3cr3tK3y,7h47Y0uC4n7F1nd:)!"
ALGORITHM = "HS256"

def random_chip_id(length: int = 15) -> str:
    return "".join(random.choices(string.digits, k=length))

def generate_jwt() -> str:
    chip_id = random_chip_id()
    # valability as ISO 8601 date (e.g. "2026-01-10")
    valability_date = datetime.now(timezone.utc).isoformat()
    payload = {
        "chip_id": chip_id,
        "valability": valability_date,  # string date, not timestamp
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    if isinstance(token, bytes):  # PyJWT < 2
        token = token.decode("utf-8")
    return token


if __name__ == "__main__":
    print(generate_jwt())
