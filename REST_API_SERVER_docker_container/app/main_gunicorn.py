from flask import Flask, jsonify, request, g
import ssl
import os
import mysql.connector
from mysql.connector import pooling
import psutil
import redis
import jwt
from datetime import datetime, timedelta, timezone
import multiprocessing

from gunicorn.app.base import BaseApplication


SECRET_KEY = "V3fryS3cr3tK3y,7h47Y0uC4n7F1nd:)!"

REDIS_CONFIG = {
    'host': os.environ.get("REDIS_HOST", "192.168.50.1"),
    'port': int(os.environ.get("REDIS_PORT", "6379")),
    'password': None,
    'db': 0,
    'max_connections': 20
}

MYSQL_CONFIG = {
    'user': 'admin',
    'password': 'ionutqwerty',
    'host': '192.168.50.1',
    'port': '3306',
    'database': 'PRODUCERS'
}

SSL_CERT = 'certs/server.crt'
SSL_KEY = 'certs/server.key'


redis_pool = redis.ConnectionPool(**REDIS_CONFIG)

def get_redis():
    return redis.Redis(connection_pool=redis_pool)

def send_data_redis(chip_id: str, temperature: float, humidity: float, 
                    wind_speed: float, rainfall: float, 
                    wind_direction_degrees: float,
                    dust: float, pressure: float = 0.0, altitude: float = 0.0):
    r = get_redis()
    ts = r.ts()
    timestamp = int(psutil.time.time() * 1000)
    
    pipe = ts.pipeline()
    pipe.add(f"sensor:{chip_id}:temperature", timestamp, temperature)
    pipe.add(f"sensor:{chip_id}:humidity", timestamp, humidity)
    pipe.add(f"sensor:{chip_id}:wind_speed", timestamp, wind_speed)
    pipe.add(f"sensor:{chip_id}:rainfall", timestamp, rainfall)
    pipe.add(f"sensor:{chip_id}:wind_direction_degrees", timestamp, wind_direction_degrees)
    pipe.add(f"sensor:{chip_id}:dust", timestamp, dust)  # Assuming dust value is 0 for now
    pipe.add(f"sensor:{chip_id}:pressure", timestamp, pressure)
    pipe.add(f"sensor:{chip_id}:altitude", timestamp, altitude)
    pipe.execute()

def init_timeseries(chip_id: str):
    r = get_redis()
    ts = r.ts()
    keys = ["temperature", "humidity", "wind_speed", "rainfall", 
            "wind_direction_degrees", "wind_direction_voltages", "dust", "pressure", "altitude"]
    for key in keys:
        if not r.exists(f"sensor:{chip_id}:{key}"):
            ts.create(f"sensor:{chip_id}:{key}")


db_pool = None

def init_db_pool():
    global db_pool
    db_pool = pooling.MySQLConnectionPool(
        pool_name="producers_pool",
        pool_size=5,
        pool_reset_session=True,
        **MYSQL_CONFIG
    )

def get_db():
    if 'db' not in g:
        g.db = db_pool.get_connection()
    return g.db

def get_cursor():
    return get_db().cursor(dictionary=True)


def generate_chip_token(chip_id: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "chip_id": chip_id,
        "valability": now,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def create_app():
    app = Flask(__name__)
    
    @app.teardown_appcontext
    def close_db(error):
        db = g.pop('db', None)
        if db is not None:
            db.close()

    @app.route('/request', methods=['POST'])
    def request_access():
        try:
            decoded = jwt.decode(request.args.get('jwt'), SECRET_KEY, algorithms=["HS256"])
            chip_id = decoded.get('chip_id')
        except Exception:
            return jsonify({'error': 'Invalid JWT', 'code': 400}), 400
        
        cursor = get_cursor()
        
        cursor.execute("SELECT chip_id FROM BLACK_LIST_PRODUCERS WHERE chip_id = %s", (chip_id,))
        if cursor.fetchone():
            return jsonify({'error': 'Chip ID is blacklisted', 'code': 403}), 403
        
        if chip_id and len(chip_id) > 10 and chip_id.isdigit():
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
        
        cursor.execute("SELECT chip_id FROM PRODUCERS WHERE chip_id = %s", (chip_id,))
        if chip_id is None or chip_id == '' or not cursor.fetchone():
            return jsonify({'error': 'Chip ID not allowed', 'code': 403}), 403
        
        cursor.execute("SELECT chip_id FROM BLACK_LIST_PRODUCERS WHERE chip_id = %s", (chip_id,))
        if cursor.fetchone():
            return jsonify({'error': 'Chip ID is blacklisted', 'code': 405}), 405
        
        token = generate_chip_token(chip_id)
        last_generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("UPDATE PRODUCERS SET last_generated_token = %s WHERE chip_id = %s", 
                      (last_generated, chip_id))
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
        
        cursor.execute("SELECT chip_id FROM BLACK_LIST_PRODUCERS WHERE chip_id = %s", (chip_id,))
        if cursor.fetchone():
            return jsonify({'error': 'Chip ID is blacklisted', 'code': 403}), 403
        
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
            dust = data.get('dust')
            pressure = data.get('pressure')
            altitude = data.get('altitude')
            
            send_data_redis(chip_id, temperature, humidity, wind_speed, rainfall, 
                          wind_direction_degrees, dust, pressure, altitude)
            
            ssid = data.get('ssid')
            cursor.execute("UPDATE PRODUCERS SET ssid = %s WHERE chip_id = %s", (ssid, chip_id))
            get_db().commit()
            return jsonify(data), 200
        else:
            return jsonify({'error': 'Invalid or expired token', 'code': 405}), 405

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

    return app


class GunicornApp(BaseApplication):
    
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def post_fork(server, worker):
    init_db_pool()
    print(f"Worker {worker.pid} initialized with DB pool")


def main():
    workers = int(os.environ.get('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
    threads = int(os.environ.get('GUNICORN_THREADS', 2))
    
    options = {
        'bind': '0.0.0.0:5500',
        'workers': workers,
        'threads': threads,
        'worker_class': 'gthread',
        'timeout': 30,
        'keepalive': 5,
        'max_requests': 1000,
        'max_requests_jitter': 50,
        'preload_app': False,
        'post_fork': post_fork,
        'certfile': SSL_CERT,
        'keyfile': SSL_KEY,
        'accesslog': '-',
        'errorlog': '-',
        'loglevel': 'info',
        'limit_request_line': 4094,
        'limit_request_fields': 100,
    }
    
    print(f"Starting Gunicorn server with {workers} workers x {threads} threads")
    print(f"Listening on https://0.0.0.0:5500")
    
    app = create_app()
    GunicornApp(app, options).run()


if __name__ == '__main__':
    main()
