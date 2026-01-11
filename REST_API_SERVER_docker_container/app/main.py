from flask import Flask, jsonify, request, redirect, g
import ssl
import os
import mysql.connector
from mysql.connector import pooling
import random
import psutil
import redis
import jwt
from datetime import datetime, timedelta, timezone

SECRET_KEY = "V3fryS3cr3tK3y,7h47Y0uC4n7F1nd:)!" 

def generate_chip_token(chip_id: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "chip_id": chip_id,
        "valability": now,
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

# Redis connection pool (thread-safe)
redis_pool = redis.ConnectionPool(
    host=os.environ.get("REDIS_HOST", "192.168.0.177"),
    port=int(os.environ.get("REDIS_PORT", "6379")),
    password=None,
    db=0,
    max_connections=10
)

def get_redis():
    return redis.Redis(connection_pool=redis_pool)

def send_data_redis(chip_id: str, temperature: float, humidity: float, wind_speed: float, rainfall: float, wind_direction_degrees: float, wind_direction_voltages: float):
    r = get_redis()
    ts = r.ts()
    timestamp = int(psutil.time.time() * 1000)
    # Use pipeline for atomic batch insert (faster)
    pipe = ts.pipeline()
    pipe.add(f"sensor:{chip_id}:temperature", timestamp, temperature)
    pipe.add(f"sensor:{chip_id}:humidity", timestamp, humidity)
    pipe.add(f"sensor:{chip_id}:wind_speed", timestamp, wind_speed)
    pipe.add(f"sensor:{chip_id}:rainfall", timestamp, rainfall)
    pipe.add(f"sensor:{chip_id}:wind_direction_degrees", timestamp, wind_direction_degrees)
    pipe.add(f"sensor:{chip_id}:wind_direction_voltages", timestamp, wind_direction_voltages)
    pipe.execute()

def init_timeseries(chip_id: str):
    r = get_redis()
    ts = r.ts()
    keys = ["temperature", "humidity", "wind_speed", "rainfall", "wind_direction_degrees", "wind_direction_voltages"]
    for key in keys:
        if not r.exists(f"sensor:{chip_id}:{key}"):
            ts.create(f"sensor:{chip_id}:{key}")

producers_database_config = {
    'user': 'admin',
    'password': 'ionutqwerty',
    'host': '192.168.0.177',
    'port': '3306',
    'database': 'PRODUCERS'
}

# MySQL connection pool (thread-safe, handles reconnection)
db_pool = pooling.MySQLConnectionPool(
    pool_name="producers_pool",
    pool_size=5,
    pool_reset_session=True,
    **producers_database_config
)

def get_db():
    """Get a database connection from the pool for this request."""
    if 'db' not in g:
        g.db = db_pool.get_connection()
    return g.db

def get_cursor():
    """Get a cursor for the current request's connection."""
    db = get_db()
    return db.cursor(dictionary=True)

app = Flask(__name__)
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('certs/server.crt', 'certs/server.key')

@app.teardown_appcontext
def close_db(error):
    """Return connection to pool after each request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()  # Returns to pool, doesn't actually close
        
@app.route('/request', methods=['POST'])
def request_access():
    try:
        decoded = jwt.decode(request.args.get('jwt'), SECRET_KEY, algorithms=["HS256"])
        chip_id = decoded.get('chip_id')
    except Exception:
        return jsonify({'error': 'Invalid JWT', 'code': 400}), 400
    
    cursor = get_cursor()
    
    # Check blacklist
    cursor.execute("SELECT chip_id FROM BLACK_LIST_PRODUCERS WHERE chip_id = %s", (chip_id,))
    if cursor.fetchone():
        return jsonify({'error': 'Chip ID is blacklisted', 'code': 403}), 403
    
    if chip_id and len(chip_id) == 15 and chip_id.isdigit():
        cursor.execute("INSERT INTO PRODUCERS (chip_id) VALUES (%s)", (chip_id,))
        get_db().commit()
        init_timeseries(chip_id)
        return jsonify({'message': 'Access request submitted'}), 200
    else:
        cursor.execute("INSERT INTO BLACK_LIST_PRODUCERS (chip_id) VALUES (%s)", (chip_id,))
        get_db().commit()
        return jsonify({'error': 'Invalid chip ID sent to black list', 'code': 405}), 405

@app.route('/generate_token', methods=['POST'])
def generate_token():
    try:
        decoded = jwt.decode(request.args.get('jwt'), SECRET_KEY, algorithms=["HS256"])
        chip_id = decoded['chip_id']
    except Exception:
        return jsonify({'error': 'Invalid JWT token', 'code': 400}), 400
    
    cursor = get_cursor()
    
    # Check if chip_id exists in producers
    cursor.execute("SELECT chip_id FROM PRODUCERS WHERE chip_id = %s", (chip_id,))
    if chip_id is None or chip_id == '' or not cursor.fetchone():
        return jsonify({'error': 'Chip ID not allowed', 'code': 403}), 403
    
    # Check blacklist
    cursor.execute("SELECT chip_id FROM BLACK_LIST_PRODUCERS WHERE chip_id = %s", (chip_id,))
    if cursor.fetchone():
        return jsonify({'error': 'Chip ID is blacklisted', 'code': 405}), 405
    
    token = generate_chip_token(chip_id)
    last_generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("UPDATE PRODUCERS SET last_generated_token = %s WHERE chip_id = %s", (last_generated, chip_id))
    get_db().commit()
    return jsonify(token), 200

@app.route('/get_data', methods=['POST'])
def retrieve_data():
    try:
        decoded = jwt.decode(request.args.get('jwt'), SECRET_KEY, algorithms=["HS256"])
        chip_id = decoded['chip_id']
    except Exception:
        return jsonify({'error': 'Invalid JWT token', 'code': 400}), 400
    
    cursor = get_cursor()
    
    # Check blacklist
    cursor.execute("SELECT chip_id FROM BLACK_LIST_PRODUCERS WHERE chip_id = %s", (chip_id,))
    if cursor.fetchone():
        return jsonify({'error': 'Chip ID is blacklisted', 'code': 403}), 403
    
    # Check token validity and producer existence
    cursor.execute("SELECT chip_id FROM PRODUCERS WHERE chip_id = %s", (chip_id,))
    is_valid_producer = cursor.fetchone() is not None
    is_token_valid = decoded['valability'] > (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    
    if is_token_valid and is_valid_producer:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({'error': 'JSON body is required', 'code': 400}), 400
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        wind_speed = data.get('wind_speed')
        rainfall = data.get('rainfall')
        wind_direction_degrees = data.get('wind_direction_degrees')
        wind_direction_voltages = data.get('wind_direction_voltage')
        send_data_redis(chip_id, temperature, humidity, wind_speed, rainfall, wind_direction_degrees, wind_direction_voltages)
        ssid = data.get('ssid')
        cursor.execute("UPDATE PRODUCERS SET ssid = %s WHERE chip_id = %s", (ssid, chip_id))
        get_db().commit()
        return jsonify(data), 200
    else:
        return jsonify({'error': 'Invalid or expired token', 'code': 405}), 405
    
if __name__ == '__main__':
    # Use threaded=True to handle multiple concurrent requests
    # This allows the server to handle new requests while others are waiting for DB/Redis
    app.run(host='0.0.0.0', port=5500, ssl_context=ssl_context, threaded=True)