#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
control.py

This module provides the ControlServer class, which manages client connections and routes messages between them.
It implements a singleton pattern to ensure only one server instance is running.
The server listens for incoming client connections, accepts them, and forwards messages from one client to another.

The ControlServer is designed to handle multiple clients concurrently, using threading to manage each client connection.
It provides methods to start the server, accept connections, handle client communication, route messages, and shut down the server.

Attributes:
    ControlServer._instance (ControlServer): The singleton instance of the ControlServer class.

Classes:
    ControlServer: A singleton-based server class for managing client connections and message routing.

Functions:
    None

Example:
    >>> server = ControlServer('127.0.0.1', 65432)
    >>> server.start_server()
    >>> # The server will now accept connections and route messages between clients
    >>> # To shut down the server
    >>> server.shutdown()

Note:
    The server must be started before it can accept client connections.
    Shutting down the server will close all active client connections.

See Also:
    socket: The socket library is used for network communication.
    json: The json library is used for message serialization and deserialization.
    threading: The threading library is used for concurrent client handling.
"""

import socket
import threading
import json


class ControlServer:
    """
    ControlServer 是一个控制服务器类，用于管理客户端连接并转发消息。
    它实现了单例模式，确保只有一个服务器实例在运行。
    提供了启动服务器、接受客户端连接、处理客户端消息和关闭服务器等功能。

    Attributes:
        _instance (ControlServer): 单例实例。
        host (str): 服务器的主机地址。
        port (int): 服务器的端口号。
        clients (dict): 已连接的客户端，键为客户端身份标识(identity)，值为对应的 socket 对象。
        server_socket (socket.socket): 用于监听客户端连接的 socket 对象。

    Methods:
        __new__(cls, host, port): 创建一个新的服务器实例或返回已存在的实例。
        start_server(self): 启动服务器，开始监听客户端连接。
        accept_connections(self): 接受客户端连接请求。
        handle_client(self, client_socket): 处理客户端连接和消息接收。
        route_message(self, data, sender): 转发消息到目标客户端。
        shutdown(self): 关闭服务器。

    Note:
        服务器需要先启动才能接受客户端连接。
        关闭服务器会关闭所有已连接的客户端连接。
    """
    _instance = None

    def __new__(cls, host='127.0.0.1', port=65432):
        """
        创建一个新的服务器实例或返回已存在的实例。

        Args:
            host (str, optional): 服务器的主机地址。 Defaults to '127.0.0.1'.
            port (int, optional): 服务器的端口号。 Defaults to 65432.

        Returns:
            ControlServer: 服务器实例。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.host = host
            cls._instance.port = port
            cls._instance.clients = {}
            cls._instance.server_socket = None
            cls._instance.start_server()
        return cls._instance

    def start_server(self):
        """
        启动服务器，开始监听客户端连接。

        服务器会绑定到指定的主机和端口，并开始监听客户端连接请求。
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f"Control server started at {self.host}:{self.port}")
        threading.Thread(target=self.accept_connections, daemon=True).start()

    def accept_connections(self):
        """
        接受客户端连接请求。

        该方法会持续运行，接受客户端连接请求并为每个客户端启动一个处理线程。
        """
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()
            except OSError:
                break

    def handle_client(self, client_socket):
        """
        处理客户端连接和消息接收。

        Args:
            client_socket (socket.socket): 客户端的 socket 对象。

        该方法会接收客户端的身份标识，并注册客户端连接。
        然后持续接收客户端发送的消息并进行转发。
        """
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
        """
        转发消息到目标客户端。

        Args:
            data (bytes): 接收到的消息数据。
            sender (str): 发送消息的客户端身份标识。

        该方法解析消息内容，查找目标客户端，并将消息转发到目标客户端。
        """
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
        """
        关闭服务器。

        关闭服务器 socket 并释放资源。
        """
        if self.server_socket:
            self.server_socket.close()
            print("Server shutdown")
