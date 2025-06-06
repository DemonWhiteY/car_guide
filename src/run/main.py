from comm_objects import comm_objects
import threading
import time
import importlib


def start_module(module_name):
    """启动指定模块"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, 'start'):
            module.start()
        else:
            print(f"Module {module_name} has no start function")
    except ImportError as e:
        print(f"Error starting module {module_name}: {str(e)}")


def initialize_system():
    """初始化系统"""
    print("Initializing system...")

    # 1. 启动控制服务器
    comm_objects.start_control_server(host='127.0.0.1', port=65432)

    # 2. 等待服务器启动
    time.sleep(1)

    # 3. 启动所有模块（在独立线程中）
    modules = [
        "ui_module",
        "llm_module",
    ]
    comm_objects.face_comm
    comm_objects.gesture_comm

    threads = []
    for module in modules:
        thread = threading.Thread(target=start_module, args=(module,), daemon=True)
        thread.start()
        threads.append(thread)
        print(f"Started {module}")
        time.sleep(0.5)  # 短暂延迟确保顺序启动

    print("System initialized. Press Ctrl+C to exit.")
    return threads


if __name__ == "__main__":
    threads = initialize_system()

    # 主线程保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 停止服务器
        comm_objects.stop_control_server()
        print("System shutdown")