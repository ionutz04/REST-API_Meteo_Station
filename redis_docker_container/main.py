import os 
import redis

r = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", "6379")),
    password=os.environ.get("REDIS_PASSWORD"),  # or None if no auth
    db=0,
)

value = r.get("timestamp")
print(value)
