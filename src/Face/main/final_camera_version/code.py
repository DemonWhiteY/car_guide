import cv2
import numpy as np
import os
import time
import insightface
from insightface.app import FaceAnalysis
import mediapipe as mp
from collections import deque
import json

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

# 缓存，记录行为信息
EYE_CLOSED_BUFFER = deque(maxlen=30)
LOOKING_SIDE_BUFFER = deque(maxlen=30)

NOD_BUFFER = deque(maxlen=5)  # 存储y轴运动趋势
SHAKE_BUFFER = deque(maxlen=5)  # 存储x轴运动趋势

# 缓存，记录身份信息
IDENTITY_BUFFER = deque(maxlen=10)

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
    db = load_database()

    while True:
        name = input("请输入要注册的人名：")
        if name == "None" or name == "Unknown":
            print("人名非法，请重新注册")
        elif name in db:
            choice = input("数据库中已注册同名用户，输入1确认覆盖信息:")
            if choice == "1":
                break
            else:
                print("输入不为1，请按照指示重新输入!")
        else:
            break
    
    while True:
        identity = input("请输入注册人名的身份(Driver/Passenger)：")
        if identity != "Driver" and identity != "Passenger":
            print("身份只能为Driver/Passenger，输入非法，请重新注册")
        else:
            break

    embeddings = []
    print("采集中，请正对摄像头...")

    while len(embeddings) < capture_count:
        ret, frame = cap.read()
        if not ret:
            print("摄像头采集失败")
            break
        
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
        db[name] = {
            'identity': identity, # 身份信息，乘客或者驾驶员
            'embeddings': embeddings,  # 保存所有原始样本
            'mean_emb': np.mean(embeddings, axis=0),  # 保留平均值
        }
        save_database(db)
    else:
        print("注册失败：未能采集到有效人脸")

# 获得鼻尖运动趋势, 计算运动位移
def update_head_movement_buffer(landmarks):
    # 只使用鼻尖判断
    nose = landmarks[1]

    # 只用鼻尖也可以：更敏感
    y_pos = nose.y
    x_pos = nose.x

    # 记录连续帧的变化趋势
    if not hasattr(update_head_movement_buffer, "last_y"):
        update_head_movement_buffer.last_y = y_pos
        update_head_movement_buffer.last_x = x_pos

    dy = y_pos - update_head_movement_buffer.last_y
    dx = x_pos - update_head_movement_buffer.last_x

    # 更新缓冲区
    NOD_BUFFER.append(dy)
    SHAKE_BUFFER.append(dx)

    update_head_movement_buffer.last_y = y_pos
    update_head_movement_buffer.last_x = x_pos

def is_nodding(box):
    # 点头检测：Y方向有明显的上-下-上或下-上-下的震荡（变化方向切换次数 > 1）
    trend = [1 if dy > 0 else -1 for dy in NOD_BUFFER if abs(dy) * 270 > 0.02 * abs(box[3] - box[1])]
    changes = sum(1 for i in range(1, len(trend)) if trend[i] != trend[i-1])
    return changes >= 2

def is_shaking(box):
    # 摇头检测：X方向有明显的左-右-左或右-左-右的震荡
    trend = [1 if dx > 0 else -1 for dx in SHAKE_BUFFER if abs(dx) * 270 > 0.02 * abs(box[2] - box[0])]
    changes = sum(1 for i in range(1, len(trend)) if trend[i] != trend[i-1])
    return changes >= 2

# 行为判断函数
def is_eye_closed(box, landmarks, side='left'):
    if side == 'left':
        top, bottom = landmarks[159], landmarks[145]
    else:
        top, bottom = landmarks[386], landmarks[374]
    #print(abs(top.y - bottom.y) * 270 / abs(box[3] - box[1]))
    return abs(top.y - bottom.y) * 270 < 0.007 * abs(box[3] - box[1])

def is_looking_side(box, landmarks):
    left_eye, right_eye, nose = landmarks[33], landmarks[263], landmarks[1]
    dx = abs(left_eye.x - right_eye.x)
    dn = abs(nose.x - (left_eye.x + right_eye.x) / 2)
    return (dn / dx) > 0.08

# 状态监测
def monitor_behavior(box, landmarks, frame):
    left_closed = is_eye_closed(box, landmarks, 'left')
    right_closed = is_eye_closed(box, landmarks, 'right')
    EYE_CLOSED_BUFFER.append(left_closed and right_closed)
    LOOKING_SIDE_BUFFER.append(is_looking_side(box, landmarks))

    # 计算鼻尖移动位置和趋势
    update_head_movement_buffer(landmarks)

    action_str_list = []

    if sum(EYE_CLOSED_BUFFER) / len(EYE_CLOSED_BUFFER) > 0.5:
        cv2.putText(frame, "Long term closed eyes", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # 实时更新action信息
        action_str_list += ["长时间闭眼"]
        # print("长时间闭眼")

    if sum(LOOKING_SIDE_BUFFER) / len(LOOKING_SIDE_BUFFER) > 0.5:
        cv2.putText(frame, "Continuously looking east and west", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # 实时更新action信息
        action_str_list += ["持续东张西望"]
        # print("持续东张西望")
    
    # 判断是否点头
    if is_nodding(box):
        cv2.putText(frame, "Nodding", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        action_str_list.append("点头")

    # 判断是否摇头
    if is_shaking(box):
        cv2.putText(frame, "Shaking head", (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        action_str_list.append("摇头")
    
    return "无明显行为" if action_str_list == [] else ", ".join(action_str_list)

# User: xxx  | actions:  xxxx
"""
{   
    "user_name": str
    "user_identity": str
    "actions": str
}
"""

# 实时识别人脸
def recognize_face(cap):
    db = load_database()

    # 本次识别到的用户姓名
    identity_name = "None"
    # 当前记录的用户姓名
    record_name = "None"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头采集失败")
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
                
                # 默认为"Unknown"(数据库为空，也一定是Unknown)
                identity_name = "Unknown"
                # 找出最佳匹配
                if candidates:
                    best_score, best_name = min(candidates, key=lambda x: x[0])
                    identity_name = best_name if best_score < identity_threshold else "Unknown"

            # 正常来说，找到人脸就有编码，进入这里其实是逻辑异常，这里我们认为user是"未知用户"
            else:
                min_dist = identity_threshold
                identity_name = "Unknown"
            
        # 没找到人脸
        else:
            identity_name = "None"
        
        """---------根据本次识别信息，修改记录信息--------------"""

        # 如果是注册用户到Unknown:
        if record_name != "None" and record_name != "Unknown" and identity_name == "Unknown":
            # 连续若干帧识别为Unknown才修改记录信息, 否则保持不动
            if sum(IDENTITY_BUFFER) == len(IDENTITY_BUFFER):
                record_name = "Unknown"
        else:
            record_name = identity_name

        # 在图像上显示人脸框和识别结果
        if record_name != "None":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{record_name}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 记录识别的信息是不是Unknown
        IDENTITY_BUFFER.append(identity_name == "Unknown")

        # 根据记录的用户信息给出记录的身份结果
        if record_name == 'Unknown':
            record_role = "Unregistered Role"
        elif record_name == "None":
            record_role = "None"
        else:
            record_role = db[record_name]['identity']

        # 记录action数据
        action_str = "无明显行为"
        if mesh_result.multi_face_landmarks:
            landmarks = mesh_result.multi_face_landmarks[0].landmark
            action_str = monitor_behavior(box, landmarks, frame)

        # 构造发送数据字典
        data = {
            "user_name": record_name,
            "user_identity": record_role,
            "actions": action_str
        }

        # 将字典转换为JSON字符串
        json_string = json.dumps(data, ensure_ascii=False)  # ensure_ascii=False允许非ASCII字符
        # TODO: send_message
        print(json_string)

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