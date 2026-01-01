from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/sensor", methods=["POST"])
def sensor():
    data = request.get_json(force=True)
    print("Received:", data)
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=5000,
            ssl_context=("jetson.crt", "jetson.key"))

