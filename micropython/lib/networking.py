from lib.umqtt import MQTTClient
from lib.utils import load_env_vars

import ujson as json
import urandom


def rand_str(l=10):
    return "".join([chr(urandom.randint(80, 120)) for i in range(l)])


def default_sub_cb(topic, msg):
    print(topic, msg)


env_vars = load_env_vars("lib/.env")


def setup_mqtt_client(client_id=None, server=None, port=None, callback=None):

    server = server or env_vars.get("MQTT_SERVER")
    port = port or env_vars.get("MQTT_PORT")

    client_id = client_id or "eeg-esp32-" + rand_str(l=5)
    client = MQTTClient(
        client_id=client_id, server=server, port=port, keepalive=6000, ssl=False
    )

    callback = callback or default_sub_cb
    client.set_callback(callback)

    return client


def get_default_topic():
    return env_vars.get("MQTT_DEFAULT_TOPIC")


def pack_payload(raw_data, decoded_data, user_id=None, session_id=None):
    import utime as time

    payload = {
        "eeg_data": raw_data,
        "eeg_data_len": len(raw_data),
        "decoded_eeg_data": decoded_data,
        "timestamp": time.ticks_us(),
    }
    if session_id is not None:
        payload['session_id'] = session_id
        
    if user_id is not None:
        payload['user_id'] = user_id

    return json.dumps(payload)
