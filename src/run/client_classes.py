#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
client_classes.py

This module defines specific client classes for different communication purposes in the system.
Each client class inherits from the BaseClient class and provides a specific identity for different system components.
These classes are used to establish communication between various parts of the system and the control server.

Classes:
    UICommunication: A client class for UI module communication.
    LLMCommunication: A client class for LLM module communication.
    FaceCommunication: A client class for face recognition module communication.
    GestureCommunication: A client class for gesture recognition module communication.
    VoiceCommunication: A client class for voice module communication.

Example:
    >>> ui_client = UICommunication('ui')
    >>> llm_client = LLMCommunication('llm')
    >>> face_client = FaceCommunication('face')
    >>> gesture_client = GestureCommunication('gesture')
    >>> voice_client = VoiceCommunication('voice')

Note:
    Each client class is designed to be used by a specific module in the system.
    The identity of each client is fixed and should match the expected identity in the control server.

See Also:
    BaseClient: The base class for all client classes, providing the core communication functionality.
"""

from base_client import BaseClient


class UICommunication(BaseClient):
    """
    A client class for UI module communication.

    This class provides a client instance with the identity 'ui' for communication between the UI module and the control server.

    Attributes:
        identity (str): The identity of the client, set to 'ui'.
        server_host (str): The host address of the server.
        server_port (int): The port number of the server.

    Methods:
        __init__(self, identity, server_host, server_port): Initializes the UICommunication client.

    Example:
        >>> ui_client = UICommunication('ui')
    """

    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        """
        Initializes the UICommunication client.

        Args:
            identity (str): The identity of the client, should be 'ui'.
            server_host (str, optional): The host address of the server. Defaults to '127.0.0.1'.
            server_port (int, optional): The port number of the server. Defaults to 65432.
        """
        super().__init__(identity, server_host, server_port)


class LLMCommunication(BaseClient):
    """
    A client class for LLM module communication.

    This class provides a client instance with the identity 'llm' for communication between the LLM module and the control server.

    Attributes:
        identity (str): The identity of the client, set to 'llm'.
        server_host (str): The host address of the server.
        server_port (int): The port number of the server.

    Methods:
        __init__(self, identity, server_host, server_port): Initializes the LLMCommunication client.

    Example:
        >>> llm_client = LLMCommunication('llm')
    """

    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        """
        Initializes the LLMCommunication client.

        Args:
            identity (str): The identity of the client, should be 'llm'.
            server_host (str, optional): The host address of the server. Defaults to '127.0.0.1'.
            server_port (int, optional): The port number of the server. Defaults to 65432.
        """
        super().__init__(identity, server_host, server_port)


class FaceCommunication(BaseClient):
    """
    A client class for face recognition module communication.

    This class provides a client instance with the identity 'face' for communication between the face recognition module and the control server.

    Attributes:
        identity (str): The identity of the client, set to 'face'.
        server_host (str): The host address of the server.
        server_port (int): The port number of the server.

    Methods:
        __init__(self, identity, server_host, server_port): Initializes the FaceCommunication client.

    Example:
        >>> face_client = FaceCommunication('face')
    """

    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        """
        Initializes the FaceCommunication client.

        Args:
            identity (str): The identity of the client, should be 'face'.
            server_host (str, optional): The host address of the server. Defaults to '127.0.0.1'.
            server_port (int, optional): The port number of the server. Defaults to 65432.
        """
        super().__init__(identity, server_host, server_port)


class GestureCommunication(BaseClient):
    """
    A client class for gesture recognition module communication.

    This class provides a client instance with the identity 'gesture' for communication between the gesture recognition module and the control server.

    Attributes:
        identity (str): The identity of the client, set to 'gesture'.
        server_host (str): The host address of the server.
        server_port (int): The port number of the server.

    Methods:
        __init__(self, identity, server_host, server_port): Initializes the GestureCommunication client.

    Example:
        >>> gesture_client = GestureCommunication('gesture')
    """

    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        """
        Initializes the GestureCommunication client.

        Args:
            identity (str): The identity of the client, should be 'gesture'.
            server_host (str, optional): The host address of the server. Defaults to '127.0.0.1'.
            server_port (int, optional): The port number of the server. Defaults to 65432.
        """
        super().__init__(identity, server_host, server_port)


class VoiceCommunication(BaseClient):
    """
    A client class for voice module communication.

    This class provides a client instance with the identity 'voice' for communication between the voice module and the control server.

    Attributes:
        identity (str): The identity of the client, set to 'voice'.
        server_host (str): The host address of the server.
        server_port (int): The port number of the server.

    Methods:
        __init__(self, identity, server_host, server_port): Initializes the VoiceCommunication client.

    Example:
        >>> voice_client = VoiceCommunication('voice')
    """

    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        """
        Initializes the VoiceCommunication client.

        Args:
            identity (str): The identity of the client, should be 'voice'.
            server_host (str, optional): The host address of the server. Defaults to '127.0.0.1'.
            server_port (int, optional): The port number of the server. Defaults to 65432.
        """
        super().__init__(identity, server_host, server_port)
        