# base_client.py (修复版本)
import socket
import threading
import json
import time


class BaseClient:
    _instances = {}

    def __new__(cls, identity, server_host='127.0.0.1', server_port=65432):
        if identity not in cls._instances:
            instance = super().__new__(cls)
            # 立即创建 socket 对象
            instance.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cls._instances[identity] = instance
        return cls._instances[identity]

    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            return

        self.identity = identity
        self.server_host = server_host
        self.server_port = server_port
        self.connected = False

        # 连接服务器
        self.connect_to_server()

        # 启动接收线程
        threading.Thread(target=self.receive_messages, daemon=True).start()

        # 标记已初始化
        self._initialized = True

    def connect_to_server(self):
        while not self.connected:
            try:
                # 确保 socket 已创建
                if not hasattr(self, 'socket') or self.socket is None:
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                self.socket.connect((self.server_host, self.server_port))
                self.socket.sendall(self.identity.encode('utf-8'))
                print(f"{self.identity.capitalize()} connected to server")
                self.connected = True
            except (ConnectionRefusedError, OSError) as e:
                print(f"{self.identity.capitalize()} connection failed: {e}, retrying in 2 seconds...")
                time.sleep(2)
            except Exception as e:
                print(f"{self.identity.capitalize()} unexpected error: {e}")
                time.sleep(2)

    def send_message(self, data, target):
        if not self.connected:
            print(f"{self.identity} not connected, cannot send message")
            return

        message = {
            'target': target,
            'sender': self.identity,
            'data': data
        }

        try:
            self.socket.sendall(json.dumps(message).encode('utf-8'))
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"{self.identity} connection error: {e}, reconnecting...")
            self.connected = False
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 重新创建socket
            self.connect_to_server()
            # 重试发送
            if self.connected:
                self.socket.sendall(json.dumps(message).encode('utf-8'))

    def receive_messages(self):
        while True:
            if not self.connected:
                time.sleep(1)
                continue

            try:
                data = self.socket.recv(4096)
                if not data:
                    print(f"{self.identity}: Connection closed by server")
                    self.connected = False
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 重新创建socket
                    self.connect_to_server()
                    continue

                self.handle_message(data)
            except ConnectionResetError:
                print(f"{self.identity}: Connection reset by server")
                self.connected = False
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 重新创建socket
                self.connect_to_server()
            except Exception as e:
                print(f"{self.identity}: Error in receiving: {e}")
                self.connected = False
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 重新创建socket
                self.connect_to_server()

    def handle_message(self, data):
        try:
            message = json.loads(data.decode('utf-8'))
            print(f"{self.identity.upper()} received: {message['data']}")
        except json.JSONDecodeError:
            print(f"{self.identity.upper()} received invalid message")
