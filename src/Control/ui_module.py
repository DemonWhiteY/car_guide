from comm_objects import comm_objects
import time
import json


class UIHandler:
    def __init__(self):
        # 获取UI通信对象
        self.ui_comm = comm_objects.ui_comm

        # 设置自定义消息处理器
        self.ui_comm.handle_message = self.custom_handle_message
        print("UI handler initialized")

    def custom_handle_message(self, data):
        """处理来自Face、Gesture和LLM的消息"""
        try:
            message = json.loads(data.decode('utf-8'))
            sender = message['sender']
            data_content = message['data']

            print(f"UI received from {sender}: {data_content}")

            # 根据发送者处理消息
            if sender == "face":
                self.handle_face_data(data_content)
            elif sender == "gesture":
                self.handle_gesture_data(data_content)
            elif sender == "llm":
                self.handle_llm_data(data_content)
            else:
                print(f"Unhandled sender: {sender}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"UI message processing error: {str(e)}")

    def handle_face_data(self, data):
        """处理人脸数据"""
        print(f"Face expression detected: {data['expression']} (confidence: {data['confidence']})")
        # 这里可以添加UI更新逻辑，如显示表情图标

    def handle_gesture_data(self, data):
        """处理手势数据"""
        print(f"Gesture recognized: {data['gesture']} (confidence: {data['confidence']})")
        # 这里可以添加UI更新逻辑，如执行手势命令

    def handle_llm_data(self, data):
        """处理LLM数据"""
        if isinstance(data, dict) and data.get("type") == "llm_response":
            print(f"LLM response: {data['text']}")
            # 这里可以添加UI更新逻辑，如在聊天窗口显示消息

    def run(self):
        """运行UI处理循环"""
        try:
            while True:
                # UI模块主要处理消息驱动，不需要主动循环
                time.sleep(1)
        except KeyboardInterrupt:
            print("UI handler stopped")


def start_ui():
    """启动UI模块"""
    print("Starting UI module...")
    ui_handler = UIHandler()
    ui_handler.run()


def start():
    start_ui()
