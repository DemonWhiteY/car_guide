from comm_objects import comm_objects
import time
import random
import json


class VoiceProcessor:
    def __init__(self):
        # 获取Voice通信对象
        self.voice_comm = comm_objects.voice_comm

        # 设置自定义消息处理器
        self.voice_comm.handle_message = self.custom_handle_message
        print("Voice processor initialized")

    def capture_voice(self):
        """模拟语音输入并发送到LLM"""
        phrases = [
            "What's the weather today?",
            "Set a timer for 5 minutes",
            "Play some relaxing music",
            "Tell me a joke",
            "What time is it?"
        ]
        phrase = random.choice(phrases)

        message = {
            "type": "voice_input",
            "text": phrase,
            "timestamp": time.strftime("%H:%M:%S")
        }

        self.voice_comm.send_message(message, "llm")
        print(f"Voice captured: {phrase}")

    def custom_handle_message(self, data):
        """处理来自LLM的消息"""
        try:
            message = json.loads(data.decode('utf-8'))
            sender = message['sender']
            data_content = message['data']

            if sender == "llm":
                print(f"Voice received from LLM: {data_content}")

                # 处理LLM的响应
                if isinstance(data_content, dict) and data_content.get("type") == "tts":
                    self.synthesize_speech(data_content["text"])
                else:
                    print(f"Unhandled message type from LLM: {data_content}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Voice message processing error: {str(e)}")

    def synthesize_speech(self, text):
        """模拟语音合成"""
        print(f"Synthesizing speech: '{text}'")
        # 这里可以添加实际的语音合成逻辑

    def run(self):
        """运行语音处理循环"""
        try:
            while True:
                self.capture_voice()
                time.sleep(random.uniform(3.0, 6.0))  # 随机延迟
        except KeyboardInterrupt:
            print("Voice processor stopped")


def start_voice():
    """启动语音模块"""
    print("Starting voice module...")
    voice_processor = VoiceProcessor()
    voice_processor.run()


def start():
    start_voice()
