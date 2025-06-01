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

                self.llm_comm.send_message(data_content, "voice")
            if sender == "gesture":
                print("from gesture:", data_content, '\n')

        except (json.JSONDecodeError, KeyError) as e:
            print(f"LLM message processing error: {str(e)}")

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
