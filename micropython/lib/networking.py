from lib.umqtt import MQTTClient
from lib.utils import load_env_vars

import ujson as json
import urandom

def rand_str(l=10):
    return ''.join([chr(urandom.randint(80, 120)) for i in range(l)])
          
def default_sub_cb(topic, msg):
    print(topic, msg)
    
env_vars = load_env_vars("lib/.env")
    
def setup_mqtt_client(client_id=None, callback=None):
    
    server = env_vars.get('MQTT_SERVER')
    port = env_vars.get('MQTT_PORT')
    
    client_id = client_id or "james-esp32-"+rand_str(l=5)
    client = MQTTClient(client_id=client_id, server=server, port=port, keepalive=6000, ssl=False)
    
    callback = callback or default_sub_cb
    client.set_callback(callback)
    
    return client
    
def get_default_topic():
    return env_vars.get('MQTT_DEFAULT_TOPIC')

def pack_payload(raw_data, decoded_data, client_id=None):
    if client_id is None:
        client_id = "esp32_client_"+rand_str(l=5)
        
    payload = {
        "counter": 1,
        "user_id": client_id,
        "channel_code": 1,
        "eeg_data": raw_data,
        "eeg_data_len": len(raw_data),
        "decoded_eeg_data": decoded_data
    }
    return json.dumps(payload)