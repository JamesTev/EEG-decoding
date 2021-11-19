from scheduling import ScheduledFunc
import ujson as json
from utils import Enum
import config

logger_types = Enum(["SERIAL", "MQTT", "HTTP"])


class BaseLogger(ScheduledFunc):
    def __init__(self, period_sec, decoded_ref, raw_data_ref, timer_num=1):
        super().__init__(timer_num, 1 / period_sec)
        self.raw_data = raw_data_ref
        self.decoded_data = decoded_ref

    def log(self, *args):
        print(self.raw_data)

    def start(self):
        self.tim.init(freq=self.freq, callback=self.log)


class AbstractWebLogger(BaseLogger):
    def __init__(
        self,
        period_sec,
        decoded_ref,
        raw_data_ref,
        timer_num=1,
        server=None,
        send_raw=False,
    ):
        super().__init__(period_sec, decoded_ref, raw_data_ref, timer_num=timer_num)

        if period_sec < 1:
            print("Warning: async web logging at > 1Hz will be unreliable.")

        self.send_raw = send_raw  # whether or not to send full raw data
        self.server = server

    def _prepare_payload(self, payload_id=None):
        from lib.networking import pack_payload

        raw_data = self.raw_data if self.send_raw else []

        return pack_payload(raw_data, self.decoded_data, user_id=payload_id)


class MQTTLogger(AbstractWebLogger):
    def __init__(
        self,
        period_sec,
        decoded_ref,
        raw_data_ref,
        timer_num=1,
        topic=None,
        server=None,
        port=None,
        qos=1,
        send_raw=False,
    ):
        super().__init__(
            period_sec,
            decoded_ref,
            raw_data_ref,
            timer_num=timer_num,
            server=server,
            send_raw=send_raw,
        )

        from lib.networking import setup_mqtt_client, get_default_topic

        self.client = setup_mqtt_client(server=server, port=port)
        self.topic = topic or get_default_topic()
        self.qos = qos
        self.establish_connection()

    def establish_connection(self):
        msg = "ESP32 client {0} connected".format(self.client.client_id)

        self.client.connect()
        self.client.publish(
            topic=self.mqtt_topic, msg=json.dumps({"message": msg}), qos=self.qos
        )

    def _prepare_payload(self):
        return super()._prepare_payload(payload_id=self.client.client_id)

    def get_client(self):
        return self.client

    def log(self, *args):
        payload = self._prepare_payload()
        self.client.publish(topic=self.topic, msg=payload, qos=self.qos)


class HTTPLogger(AbstractWebLogger):
    def __init__(
        self,
        period_sec,
        decoded_ref,
        raw_data_ref,
        timer_num=1,
        server=None,
        send_raw=False,
    ):
        super().__init__(
            period_sec,
            decoded_ref,
            raw_data_ref,
            timer_num=timer_num,
            server=server,
            send_raw=send_raw,
        )
        self.server = server or config.HTTP_LOG_URL

    def log(self, *args):
        from lib.requests import MicroWebCli as requests

        payload = self._prepare_payload()
        requests.POSTRequest(self.server, payload)
