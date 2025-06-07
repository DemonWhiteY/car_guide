#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
base_client.py

This module provides the BaseClient class, which serves as a foundation for client-server communication.
It implements a singleton pattern to ensure each client identity has only one instance.
The class handles connecting to a server, sending messages to specific targets, receiving messages from the server,
and basic message handling.

The BaseClient is designed to be extended by specific client implementations, which can override the handle_message
method to process incoming data according to their needs.

Attributes:
    BaseClient._instances (dict): Keeps track of created client instances by their identities.

Classes:
    BaseClient: A singleton-based client class for basic server communication.

Functions:
    None

Example:
    >>> client = BaseClient('client1')
    >>> client.send_message('Hello, server!', 'server')
    >>> client.receive_messages()

Note:
    Clients must connect to the server before sending messages. The client will automatically attempt to reconnect
    if the connection is lost.

See Also:
    socket: The socket library is used for network communication.
    json: The json library is used for message serialization and deserialization.
"""

import socket
import threading
import json
import time


class BaseClient:
    """
    BaseClient 是一个基础客户端类，用于与服务器建立连接并进行通信。
    它实现了单例模式，确保每个身份标识(identity)只有一个客户端实例。
    提供了连接服务器、发送消息、接收消息等功能。

    Attributes:
        _instances (dict): 存储已创建的客户端实例，键为身份标识(identity)，值为对应的客户端实例。
        identity (str): 客户端的身份标识。
        server_host (str): 服务器的主机地址。
        server_port (int): 服务器的端口号。
        connected (bool): 客户端是否已连接到服务器。
        socket (socket.socket): 用于与服务器通信的 socket 对象。

    Methods:
        __new__(cls, identity, server_host, server_port): 创建一个新的客户端实例或返回已存在的实例。
        __init__(self, identity, server_host, server_port): 初始化客户端实例。
        connect_to_server(self): 尝试连接到服务器。
        send_message(self, data, target): 向目标客户端发送消息。
        receive_messages(self): 接收来自服务器的消息。
        handle_message(self, data): 处理接收到的消息。

    Note:
        客户端需要先连接到服务器才能发送消息。
        如果连接失败，客户端会自动重试连接。
    """
    _instances = {}

    def __new__(cls, identity, server_host='127.0.0.1', server_port=65432):
        """
        创建一个新的客户端实例或返回已存在的实例。

        Args:
            identity (str): 客户端的身份标识。
            server_host (str, optional): 服务器的主机地址。 Defaults to '127.0.0.1'.
            server_port (int, optional): 服务器的端口号。 Defaults to 65432.

        Returns:
            BaseClient: 客户端实例。
        """
        if identity not in cls._instances:
            instance = super().__new__(cls)
            # 立即创建 socket 对象
            instance.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cls._instances[identity] = instance
        return cls._instances[identity]

    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        """
        初始化客户端实例。

        Args:
            identity (str): 客户端的身份标识。
            server_host (str, optional): 服务器的主机地址。 Defaults to '127.0.0.1'.
            server_port (int, optional): 服务器的端口号。 Defaults to 65432.
        """
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
        """
        尝试连接到服务器。

        如果连接失败，会自动重试连接。
        """
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
        """
        向目标客户端发送消息。

        Args:
            data (any): 要发送的消息内容。
            target (str): 目标客户端的身份标识。

        Note:
            如果客户端未连接到服务器，将无法发送消息。
        """
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
            print(f"{self.identity.capitalize()} sent message to {target}: {data}")
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"{self.identity} connection error: {e}, reconnecting...")
            self.connected = False
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connect_to_server()
            if self.connected:
                self.socket.sendall(json.dumps(message).encode('utf-8'))

    def receive_messages(self):
        """
        接收来自服务器的消息。

        该方法会持续运行，接收并处理来自服务器的消息。
        """
        while True:
            if not self.connected:
                time.sleep(1)
                continue

            try:
                data = self.socket.recv(4096)
                if not data:
                    print(f"{self.identity}: Connection closed by server")
                    self.connected = False
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.connect_to_server()
                    continue

                self.handle_message(data)
            except ConnectionResetError:
                print(f"{self.identity}: Connection reset by server")
                self.connected = False
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connect_to_server()
            except Exception as e:
                print(f"{self.identity}: Error in receiving: {e}")
                self.connected = False
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connect_to_server()

    def handle_message(self, data):
        """
        处理接收到的消息。

        Args:
            data (bytes): 接收到的消息数据。
        """
        try:
            message = json.loads(data.decode('utf-8'))
            print(f"{self.identity.upper()} received: {message['data']}")
        except json.JSONDecodeError:
            print(f"{self.identity.upper()} received invalid message")
