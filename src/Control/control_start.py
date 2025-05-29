# control_start.py
from comm_objects import comm_objects
import time


def initialize_system():
    # 1. 启动控制服务器
    comm_objects.start_control_server(host='127.0.0.1', port=65432)

    # 2. 等待服务器启动
    time.sleep(1)

    # 3. 创建需要的客户端连接（可选）
    # 如果确定需要所有连接，可以显式创建：
    comm_objects.ui_comm  # 创建UI连接
    comm_objects.llm_comm  # 创建LLM连接
    comm_objects.face_comm  # 创建Face连接
    comm_objects.gesture_comm  # 创建Gesture连接
    comm_objects.voice_comm  # 创建Voice连接

    # 或者按需创建，在后续使用时自动创建
    print("System initialized")


if __name__ == "__main__":
    initialize_system()

    # 保持程序运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 停止服务器（可选）
        comm_objects.stop_control_server()
        print("System shutdown")