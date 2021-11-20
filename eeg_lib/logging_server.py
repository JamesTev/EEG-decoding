from flask import Flask, request, jsonify
import json

import time

app = Flask(__name__)

DEFAULT_FILENAME = "./logs/log_data.json"


def write_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def read_json(filename):
    with open(filename) as f:
        return json.load(f)


def log_data(payload, filename=None):

    filename = filename or DEFAULT_FILENAME
    session_id = payload.get("session_id", f"default_session_{int(time.time())}")
    try:
        existing_data = read_json(filename)
    except FileNotFoundError:
        existing_data = {}

    if session_id in existing_data:
        existing_data[session_id].append(payload)
        del payload["session_id"]
    else:
        existing_data[session_id] = [payload]
    write_json(filename, existing_data)
    print(f"Log file {filename} updated successfully.")


@app.route("/", methods=["POST"])
def save_data():
    data = request.get_json(force=True)
    if isinstance(data, str):
        data = json.loads(data)
    if data is not None:
        log_data(data)
        return jsonify(msg="data stored successfully"), 200
    return jsonify(msg="invalid data payload"), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
