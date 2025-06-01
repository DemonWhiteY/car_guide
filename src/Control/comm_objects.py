from car_guide.src.Control.control import ControlServer
from car_guide.src.Control.client_classes import UICommunication, LLMCommunication, FaceCommunication, GestureCommunication, \
    VoiceCommunication
import threading
import json
import sys
import dashscope
from dashscope.audio.tts import SpeechSynthesizer

dashscope.api_key = 'sk-166fb0f2501140c8ad8e2058aaae67e9'  # set API-key manually
vol_idx = 0


def l2v(text):
    global vol_idx
    ret = f"output{vol_idx}.wav"
    result = SpeechSynthesizer.call(model='sambert-zhichu-v1',
                                    text=text,
                                    sample_rate=48000,
                                    format='wav')

    if result.get_audio_data() is not None:
        with open(ret, 'wb') as f:
            f.write(result.get_audio_data())
            vol_idx +=1
        print('SUCCESS: get audio data: %dbytes in output.wav' %
              (sys.getsizeof(result.get_audio_data())))
    else:
        print('ERROR: response is %s' % (result.get_response()))

    return ret
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
        print(f"Control server started at {host}:{port}")

    def stop_control_server(self):
        """停止控制服务器"""
        if self._control_server:
            self._control_server.shutdown()
            self._control_server = None
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
            self._ui_comm = UICommunication('ui')  # 添加身份参数
        return self._ui_comm

    @property
    def llm_comm(self):
        """获取LLM通信实例"""
        if self._llm_comm is None:
            print("Creating LLM communication instance")
            self._llm_comm = LLMCommunication('llm')  # 添加身份参数
        return self._llm_comm

    @property
    def face_comm(self):
        """获取Face通信实例"""
        if self._face_comm is None:
            print("Creating Face communication instance")
            self._face_comm = FaceCommunication('face')  # 添加身份参数
        return self._face_comm

    @property
    def gesture_comm(self):
        """获取Gesture通信实例"""
        if self._gesture_comm is None:
            print("Creating Gesture communication instance")
            self._gesture_comm = GestureCommunication('gesture')  # 添加身份参数
        return self._gesture_comm

    @property
    def voice_comm(self):
        """获取Voice通信实例"""
        if self._voice_comm is None:
            print("Creating Voice communication instance")
            self._voice_comm = VoiceCommunication('voice')  # 添加身份参数
            self._voice_comm.handle_message = self.voice_handle_message
        return self._voice_comm

    def voice_handle_message(self, data):
        try:
            message = json.loads(data.decode('utf-8'))
            sender = message['sender']
            data_content = message['data']

            print(f"Voice received from {sender}: {data_content}")
            self.voice_comm.send_message(l2v(data_content), 'ui')

        except (json.JSONDecodeError, KeyError) as e:
            print(f"UI message processing error: {str(e)}")

# 创建全局单例对象容器
comm_objects = CommObjects()
