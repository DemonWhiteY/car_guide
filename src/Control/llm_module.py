from comm_objects import comm_objects
import time
import random
import json


class LLMProcessor:
    def __init__(self):
        # 获取LLM通信对象
        self.llm_comm = comm_objects.llm_comm

        # 设置自定义消息处理器
        self.llm_comm.handle_message = self.custom_handle_message
        print("LLM processor initialized")

    def custom_handle_message(self, data):
        """处理来自Voice的消息"""
        try:
            message = json.loads(data.decode('utf-8'))
            sender = message['sender']
            data_content = message['data']

            if sender == "voice":
                print(f"LLM received from Voice: {data_content}")

                # 处理语音输入
                if isinstance(data_content, dict) and data_content.get("type") == "voice_input":
                    self.process_voice_input(data_content["text"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"LLM message processing error: {str(e)}")

    def process_voice_input(self, text):
        """处理语音输入并生成响应"""
        print(f"Processing voice input: '{text}'")

        # 生成UI响应
        ui_response = {
            "type": "llm_response",
            "text": f"Processing your request: {text}",
            "timestamp": time.strftime("%H:%M:%S")
        }
        self.llm_comm.send_message(ui_response, "ui")

        # 生成语音响应
        tts_responses = {
            "What's the weather today?": "Today's weather is sunny with a high of 25 degrees.",
            "Set a timer for 5 minutes": "Timer set for 5 minutes starting now.",
            "Play some relaxing music": "Playing relaxing music for you.",
            "Tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
            "What time is it?": f"The current time is {time.strftime('%H:%M')}"
        }

        tts_text = tts_responses.get(text, "I'm sorry, I didn't understand that request.")
        tts_message = {
            "type": "tts",
            "text": tts_text
        }
        self.llm_comm.send_message(tts_message, "voice")

        print(f"LLM responses sent to UI and Voice")

    def run(self):
        """运行LLM处理循环"""
        try:
            while True:
                # LLM模块主要处理消息驱动，不需要主动循环
                time.sleep(1)
        except KeyboardInterrupt:
            print("LLM processor stopped")


def start_llm():
    """启动LLM模块"""
    print("Starting LLM module...")
    llm_processor = LLMProcessor()
    llm_processor.run()


def start():
    start_llm()
