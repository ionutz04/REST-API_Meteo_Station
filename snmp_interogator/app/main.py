import subprocess
import sys
import logging
import re
import threading
import datetime
import os
import redis
osis = [
    {
        "name": "Hol",
        "osi": "iso.3.6.1.4.1.17095.5.1.0",
    },
    {
        "name": "Mansarda",
        "osi": "iso.3.6.1.4.1.17095.5.2.0"
    },
    {
        "name": "afara",
        "osi": "iso.3.6.1.4.1.17095.5.3.0"
    }
]
r = redis.Redis(
    host = os.environ.get("REDIS_HOST", "192.168.0.177"),
    port = int(os.environ.get("REDIS_PORT", "6379")),
    password = None,  # e.g. REDIS_PASSWORD=ionutqwerty
    db = 0
)
def init_buckets_with_threads():
    ts = r.ts()
    threads = []

    def create_bucket_if_missing(bucket_name):
        if not r.exists(bucket_name):
            print("Creating bucket:", bucket_name)
            ts.create(bucket_name)

    for osi_info in osis:
        bucket_name = f"sensor:snmp_temperature:{osi_info['name']}"
        thread = threading.Thread(target=create_bucket_if_missing, args=(bucket_name,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
def get_snmp_temperature(osi):
    """Get current temperature from SNMP sensor"""
    command = [
        'snmpget',
        '-v', '2c',
        '-c', 'public',
        '192.168.0.100',
        osi
    ]
    
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Extract numerical value from response
        output = result.stdout.strip()
        match = re.search(r'(?:STRING|INTEGER|GAUGE|Counter32):\s+["]?([0-9.]+)', output)
        
        if match:
            value = match.group(1)
            return float(value) if '.' in value else int(value)
        else:
            raise ValueError("No numerical value found in SNMP response")
            
    except subprocess.CalledProcessError as e:
        # _LOGGER.error(f"SNMP Error: {e.stderr.strip()}")
        print(f"SNMP Error: {e.stderr.strip()}", file=sys.stderr)
        return None
    except Exception as e:
        # _LOGGER.error(f"Temperature read error: {str(e)}")
        print(f"Temperature read error: {str(e)}", file=sys.stderr)
        return None

def main():
    threads = []
    results = {}

    def fetch_temperature(osi_info):
        temp = get_snmp_temperature(osi_info["osi"])
        results[osi_info["name"]] = temp

    for osi_info in osis:
        thread = threading.Thread(target=fetch_temperature, args=(osi_info,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for name, temp in results.items():
        if temp is not None:
            print(f"Temperature at {name}: {temp}°C")
            ts = r.ts()
            bucket_name = f"sensor:snmp_temperature:{name}"
            ts_ms = int(datetime.datetime.now().timestamp() * 1000)
            ts.add(bucket_name, ts_ms, float(temp))
        else:
            print(f"Failed to read temperature at {name}")

startTime = datetime.datetime.now()
period = 5000
init_buckets_with_threads()
if __name__ == "__main__":
    while True:
        currentTime = datetime.datetime.now()
        elapsedTime = (currentTime - startTime).total_seconds() * 1000
        if elapsedTime >= period:
            main()
            startTime = currentTime