from comm_objects import comm_objects
import time
import random


class GestureRecognizer:
    def __init__(self):
        # 获取Gesture通信对象
        self.gesture_comm = comm_objects.gesture_comm
        print("Gesture recognizer initialized")

    def recognize_gesture(self):
        """模拟手势识别并发送结果到UI"""
        gestures = ["swipe_left", "swipe_right", "zoom_in", "zoom_out", "tap"]
        gesture = random.choice(gestures)
        confidence = random.uniform(0.8, 0.99)

        message = {
            "type": "gesture_recognition",
            "gesture": gesture,
            "confidence": round(confidence, 2),
            "timestamp": time.strftime("%H:%M:%S")
        }

        self.gesture_comm.send_message(message, "ui")
        print(f"Gesture recognized: {gesture} ({confidence:.2f})")

    def run(self):
        """运行手势识别循环"""
        try:
            while True:
                self.recognize_gesture()
                time.sleep(random.uniform(1.0, 3.0))  # 随机延迟
        except KeyboardInterrupt:
            print("Gesture recognizer stopped")


def start_gesture():
    """启动手势模块"""
    print("Starting gesture module...")
    gesture_recognizer = GestureRecognizer()
    gesture_recognizer.run()


def start():
    start_gesture()
