from comm_objects import comm_objects
import time
import random


class FaceDetector:
    def __init__(self):
        # 获取Face通信对象
        self.face_comm = comm_objects.face_comm
        print("Face detector initialized")

    def detect_face(self):
        """模拟人脸检测并发送结果到UI"""
        expressions = ["happy", "neutral", "sad", "angry", "surprised"]
        expression = random.choice(expressions)
        confidence = random.uniform(0.7, 0.99)

        message = {
            "type": "face_detection",
            "expression": expression,
            "confidence": round(confidence, 2),
            "timestamp": time.strftime("%H:%M:%S")
        }

        self.face_comm.send_message(message, "ui")
        print(f"Face detected: {expression} ({confidence:.2f})")

    def run(self):
        """运行人脸检测循环"""
        try:
            while True:
                self.detect_face()
                time.sleep(random.uniform(1.0, 3.0))  # 随机延迟
        except KeyboardInterrupt:
            print("Face detector stopped")


def start_face():
    """启动人脸模块"""
    print("Starting face module...")
    face_detector = FaceDetector()
    face_detector.run()


def start():
    start_face()
