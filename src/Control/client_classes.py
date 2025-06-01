from car_guide.src.Control.base_client import BaseClient


class UICommunication(BaseClient):
    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        super().__init__(identity, server_host, server_port)


class LLMCommunication(BaseClient):
    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        super().__init__(identity, server_host, server_port)


class FaceCommunication(BaseClient):
    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        super().__init__(identity, server_host, server_port)


class GestureCommunication(BaseClient):
    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        super().__init__(identity, server_host, server_port)


class VoiceCommunication(BaseClient):
    def __init__(self, identity, server_host='127.0.0.1', server_port=65432):
        super().__init__(identity, server_host, server_port)
        