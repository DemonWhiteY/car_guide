import cv2
import mediapipe as mp
from deepface import DeepFace
import os
import numpy as np
import pickle
from collections import deque
from keras.models import load_model

# 路径常量
DATABASE_DIR = "face_database"
EMBEDDING_DB_FILE = "face_embeddings.pkl"

# 初始化facent模型权重
# facenet_model = load_model("./facenet_weights.h5")

# 初始化模型
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 缓存
EYE_CLOSED_BUFFER = deque(maxlen=30)
LOOKING_SIDE_BUFFER = deque(maxlen=30)

# 加载或初始化人脸数据库
def load_database():
    if os.path.exists(EMBEDDING_DB_FILE):
        with open(EMBEDDING_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_database(db):
    with open(EMBEDDING_DB_FILE, "wb") as f:
        pickle.dump(db, f)

# 提取人脸区域
def extract_face_box(frame, detection):
    h, w, _ = frame.shape
    bbox = detection.location_data.relative_bounding_box
    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
    w_box, h_box = int(bbox.width * w), int(bbox.height * h)
    return frame[y:y + h_box, x:x + w_box], (x, y)

# 获取人脸 embedding
def get_embedding(face_img):
    if face_img is None or face_img.size == 0:
        print("提取的人脸图像为空或无效")
        return None
    
    face_img = cv2.resize(face_img, (160, 160))
    cv2.imshow("Face2", face_img)
    try:
        # return DeepFace.represent(face_img, model_name='Facenet')[0]['embedding']
        return DeepFace.represent(face_img, model_name='Facenet', enforce_detection = False)[0]['embedding']
    except Exception as e:
        print("embedding 失败：", e)
        return None

# 人脸注册
def register_face(cap, capture_count = 20):
    name = input("请输入要注册的人名：")
    embeddings = []
    print("采集中，请正对摄像头...")

    while len(embeddings) < capture_count:
        ret, frame = cap.read()
        if not ret:
            continue
        result = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.detections:
            face_img, _ = extract_face_box(frame, result.detections[0])
            emb = get_embedding(face_img)
            if emb:
                embeddings.append(emb)
                print(f"已采集 {len(embeddings)}/{capture_count}")
                cv2.putText(frame, f"Collecting: {len(embeddings)}/{capture_count}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("本帧没有检测到人脸")
        # cv2.imshow("Face", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if embeddings:
        db = load_database()
        db[name] = np.mean(embeddings, axis=0)
        save_database(db)
        print(f"已保存人脸：{name}")
    else:
        print("注册失败：未能采集到有效人脸")

# 识别最相似人脸
def recognize_face(face_img, db, threshold=8.0):
    emb = get_embedding(face_img)
    if emb is None:
        return "识别失败"
    min_dist, name = float("inf"), "Unregistered user"
    for db_name, db_emb in db.items():
        dist = np.linalg.norm(np.array(emb) - np.array(db_emb))
        if dist < min_dist:
            min_dist, name = dist, db_name
    return name if min_dist < 8.0 else "Unregistered user"

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

# 主识别运行
def run_recognition(cap):
    db = load_database()
    print("开始识别，按ESC退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(rgb)
        mesh_result = face_mesh.process(rgb)

        if result.detections:
            face_img, (x, y) = extract_face_box(frame, result.detections[0])
            name = recognize_face(face_img, db)
            print(name)
            cv2.putText(frame, f"Hello: {name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        if mesh_result.multi_face_landmarks:
            landmarks = mesh_result.multi_face_landmarks[0].landmark
            monitor_behavior(landmarks, frame)

        cv2.imshow("识别人脸", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# 主程序
def main():
    cap = cv2.VideoCapture(0)
    print("选择模式：\n1-注册人脸\n2-识别人脸")
    mode = input("请输入模式编号：")
    # mode = '2'
    if mode == '1':
        register_face(cap)
    elif mode == '2':
        run_recognition(cap)
    else:
        print("无效选项")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
