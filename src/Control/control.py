import socket
import threading
import json
import time


class ControlServer:
    _instance = None

    def __new__(cls, host='127.0.0.1', port=65432):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.host = host
            cls._instance.port = port
            cls._instance.clients = {}
            cls._instance.server_socket = None
            cls._instance.start_server()
        return cls._instance

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f"Control server started at {self.host}:{self.port}")
        threading.Thread(target=self.accept_connections, daemon=True).start()

    def accept_connections(self):
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()
            except OSError:
                break

    def handle_client(self, client_socket):
        identity = None
        try:
            # 接收客户端身份标识
            identity = client_socket.recv(1024).decode('utf-8')
            if identity in ['ui', 'llm', 'face', 'gesture', 'voice']:
                self.clients[identity] = client_socket
                print(f"Registered client: {identity}")

                # 持续接收转发消息
                while True:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    self.route_message(data, identity)
        except (ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            if identity and identity in self.clients:
                del self.clients[identity]
                print(f"Unregistered client: {identity}")
            if client_socket:
                client_socket.close()

    def route_message(self, data, sender):
        try:
            message = json.loads(data.decode('utf-8'))
            target = message['target']
            if target in self.clients:
                self.clients[target].sendall(json.dumps(message).encode('utf-8'))
                print(f"Routed message from {sender} to {target}")
            else:
                print(f"Target not found: {target}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Routing error: {str(e)}")

    def shutdown(self):
        if self.server_socket:
            self.server_socket.close()
            print("Server shutdown")
