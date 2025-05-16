import cv2
import numpy as np
import os
import time
import insightface
from insightface.app import FaceAnalysis
import mediapipe as mp
from collections import deque

# 初始化 InsightFace 模型
model_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
model_app.prepare(ctx_id=0, det_size=(640, 640))  # 自动检测 + 对齐

# 定义数据库文件路径
DB_PATH = './face_db.npy'

# 人脸裁决最低置信区间，越大越严格
mediapipe_confidence=0.9
# 比较距离最小阈值，越小越严格
identity_threshold=1.0
# 记录间隔
record_break=0.7
# 裁剪人脸额外溢出
expand_ratio=0.15

# mediapipe检测模型
face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=mediapipe_confidence)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
# 缓存
EYE_CLOSED_BUFFER = deque(maxlen=30)
LOOKING_SIDE_BUFFER = deque(maxlen=30)

# 加载/保存数据库
def load_database():
    if os.path.exists(DB_PATH):
        return np.load(DB_PATH, allow_pickle=True).item()
    return {}

def save_database(db):
    np.save(DB_PATH, db)

def get_embedding(frame):
    faces = model_app.get(frame)  # 自动检测 + 5点对齐 + 质量过滤
    if faces:
        emb = faces[0].embedding  # 默认取第一个人脸
        return emb / np.linalg.norm(emb)
    else:
        return None

#提取人脸框(仅用作绘图)
def extract_face_box(image, detection):
    h, w, _ = image.shape
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # 扩大 bbox，加入额头和下巴
    expand_x = int(bw * expand_ratio)
    expand_y = int(bh * expand_ratio)

    x1 = max(0, x - expand_x)
    y1 = max(0, y - expand_y)
    x2 = min(w, x + bw + expand_x)
    y2 = min(h, y + bh + expand_y)

    face_img = image[y1:y2, x1:x2]
    return face_img, (x1, y1, x2, y2)

# 注册新的人脸
def register_face(cap, capture_count=10):
    name = input("请输入要注册的人名：")
    embeddings = []
    print("采集中，请正对摄像头...")

    while len(embeddings) < capture_count:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 通过 MediaPipe 检测人脸
        result = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.detections:
            face_img, box = extract_face_box(frame, result.detections[0])
            x1, y1, x2, y2 = box
            emb = get_embedding(frame)

            if emb is not None:
                embeddings.append(emb)
                print(f"已采集 {len(embeddings)}/{capture_count}")

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Collecting: {len(embeddings)}/{capture_count}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                time.sleep(record_break)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        else:
            print("本帧未找到人脸")
        cv2.imshow("Face", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按Esc键退出
            break

    if embeddings:
        db = load_database()
        db[name] = {
            'embeddings': embeddings,  # 保存所有原始样本
            'mean_emb': np.mean(embeddings, axis=0),  # 保留平均值
        }
        save_database(db)
    else:
        print("注册失败：未能采集到有效人脸")

# 行为判断函数
def is_eye_closed(landmarks, side='left'):
    if side == 'left':
        top, bottom = landmarks[159], landmarks[145]
    else:
        top, bottom = landmarks[386], landmarks[374]
    return abs(top.y - bottom.y) < 0.015

def is_looking_side(landmarks):
    left_eye, right_eye, nose = landmarks[33], landmarks[263], landmarks[1]
    dx = abs(left_eye.x - right_eye.x)
    dn = abs(nose.x - (left_eye.x + right_eye.x) / 2)
    return (dn / dx) > 0.08

# 状态监测
def monitor_behavior(landmarks, frame):
    left_closed = is_eye_closed(landmarks, 'left')
    right_closed = is_eye_closed(landmarks, 'right')
    EYE_CLOSED_BUFFER.append(left_closed and right_closed)

    LOOKING_SIDE_BUFFER.append(is_looking_side(landmarks))

    if sum(EYE_CLOSED_BUFFER) / len(EYE_CLOSED_BUFFER) > 0.7:
        cv2.putText(frame, "Long term closed eyes", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        print("长时间闭眼")
    if sum(LOOKING_SIDE_BUFFER) / len(LOOKING_SIDE_BUFFER) > 0.6:
        cv2.putText(frame, "Continuously looking east and west", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        print("持续东张西望")


# 实时识别人脸
def recognize_face(cap):
    db = load_database()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 通过 MediaPipe 检测人脸
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(rgb)
        mesh_result = face_mesh.process(rgb)

        if result.detections:
            face_img, box = extract_face_box(frame, result.detections[0])
            x1, y1, x2, y2 = box
            emb = get_embedding(frame)

            # 寻找对应用户
            if emb is not None:
                candidates = []
                for name, data in db.items():
                    # 策略1：与平均向量比较
                    mean_dist = np.linalg.norm(emb - data['mean_emb'])
                    
                    # 策略2：历史最小距离
                    min_dist = min([np.linalg.norm(emb - sample) for sample in data['embeddings']])
                    
                    # 综合评分（可调整权重）
                    score = 0.6*mean_dist + 0.4*min_dist
                    candidates.append((score, name))
                
                # 找出最佳匹配
                if candidates:
                    best_score, best_name = min(candidates, key=lambda x: x[0])
                    identity = best_name if best_score < identity_threshold else "未知"

                    # 在图像上显示人脸框和识别结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{identity} ({min_dist:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 没找到人脸
        else:
            print("本帧未找到人脸")

        if mesh_result.multi_face_landmarks:
            landmarks = mesh_result.multi_face_landmarks[0].landmark
            monitor_behavior(landmarks, frame)

        # 显示识别结果
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按Esc键退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    choice = input("选择操作：1. 注册新的人脸  2. 实时识别: ")
    if choice == "1":
        register_face(cap)  # 注册新的人脸
    elif choice == "2":
        recognize_face(cap)  # 实时识别人脸
    else:
        print("无效选择")