from websocket.ws_connection import ClientClosedError
from websocket.ws_server import WebSocketServer, WebSocketClient

class BasicClient(WebSocketClient):
    def __init__(self, conn):
        super().__init__(conn)

    def process(self):
        try:
            msg = self.connection.read()
            if not msg:
                return
            msg = msg.decode("utf-8")
            print(msg)
            self.connection.write(msg)
        except ClientClosedError:
            self.connection.close()


class BasicServer(WebSocketServer):
    def __init__(self):
        super().__init__("test.html", 2)

    def _make_client(self, conn):
        return BasicClient(conn)
    
    def broadcast(self, msg):
        for client in self._clients:
            client.connection.write(msg)
            
def test():
    server = BasicServer()
    server.start()
    try:
        while True:
            server.process_all()
    except KeyboardInterrupt:
        pass
    server.stop()