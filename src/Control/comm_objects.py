# comm_objects.py
from control import ControlServer
from client_classes import UICommunication, LLMCommunication, FaceCommunication, GestureCommunication, \
    VoiceCommunication
import threading


class CommObjects:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化占位符
            cls._instance._control_server = None
            cls._instance._server_thread = None
            cls._instance._ui_comm = None
            cls._instance._llm_comm = None
            cls._instance._face_comm = None
            cls._instance._gesture_comm = None
            cls._instance._voice_comm = None
        return cls._instance

    def start_control_server(self, host='127.0.0.1', port=65432):
        """启动控制服务器"""
        if self._control_server is not None:
            print("Control server already started")
            return

        # 创建服务器实例
        self._control_server = ControlServer(host, port)

        # 在单独线程中运行服务器
        def server_run():
            try:
                while True:
                    # 服务器主循环
                    time.sleep(1)
            except KeyboardInterrupt:
                self._control_server.shutdown()

        self._server_thread = threading.Thread(target=server_run, daemon=True)
        self._server_thread.start()
        print(f"Control server started at {host}:{port}")

    def stop_control_server(self):
        """停止控制服务器"""
        if self._control_server:
            self._control_server.shutdown()
            self._control_server = None
            self._server_thread = None
            print("Control server stopped")

    @property
    def control_server(self):
        """获取控制服务器实例"""
        return self._control_server

    @property
    def ui_comm(self):
        """获取UI通信实例"""
        if self._ui_comm is None:
            print("Creating UI communication instance")
            self._ui_comm = UICommunication()
        return self._ui_comm

    @property
    def llm_comm(self):
        """获取LLM通信实例"""
        if self._llm_comm is None:
            print("Creating LLM communication instance")
            self._llm_comm = LLMCommunication()
        return self._llm_comm

    @property
    def face_comm(self):
        """获取Face通信实例"""
        if self._face_comm is None:
            print("Creating Face communication instance")
            self._face_comm = FaceCommunication()
        return self._face_comm

    @property
    def gesture_comm(self):
        """获取Gesture通信实例"""
        if self._gesture_comm is None:
            print("Creating Gesture communication instance")
            self._gesture_comm = GestureCommunication()
        return self._gesture_comm

    @property
    def voice_comm(self):
        """获取Voice通信实例"""
        if self._voice_comm is None:
            print("Creating Voice communication instance")
            self._voice_comm = VoiceCommunication()
        return self._voice_comm


# 创建全局单例对象容器
comm_objects = CommObjects()