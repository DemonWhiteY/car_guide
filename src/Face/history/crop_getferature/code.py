import cv2
import numpy as np
import os
import time
import insightface
from insightface.app import FaceAnalysis
import mediapipe as mp

# 初始化 InsightFace 模型
model = insightface.model_zoo.get_model('buffalo_l')
model.prepare(ctx_id=-1)

# 定义数据库文件路径
DB_PATH = './face_db.npy'

# 人脸裁决最低置信区间，越大越严格
mediapipe_confidence=0.9
# 比较距离最小阈值，越小越严格
identity_threshold=1.0
# 记录间隔
record_break=2
# 裁剪人脸额外溢出
expand_ratio=0.03

face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=mediapipe_confidence)

# 加载/保存数据库
def load_database():
    if os.path.exists(DB_PATH):
        return np.load(DB_PATH, allow_pickle=True).item()
    return {}

def save_database(db):
    np.save(DB_PATH, db)

def enhance_face(face_img):
    """图像增强预处理（适用于低质量人脸）"""
    # 1. 自适应直方图均衡化（CLAHE）
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. 轻度锐化（避免过度锐化噪声）
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    face_img = cv2.filter2D(face_img, -1, kernel)

    # 3. 降噪（保留边缘）
    face_img = cv2.bilateralFilter(face_img, d=5, sigmaColor=15, sigmaSpace=15)
    
    return face_img

# 提取人脸的嵌入向量
def get_embedding(face_img, enhance=True):
    if face_img.size == 0:
        return None

    try:
        # Step 1: 图像增强（可选）
        if enhance:
            face_img = enhance_face(face_img)

        # Step 2: 调整大小并转RGB
        face_img = cv2.resize(face_img, (112, 112))  # 标准输入尺寸
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Step 3: 直接提取特征（跳过检测）
        emb = model.get_feat(face_img)  # 关键！绕过检测步骤
        emb /= np.linalg.norm(emb)  # L2归一化
        
        return emb
    except Exception as e:
        print(f"[WARN] 特征提取失败: {str(e)}")
        return None

#提取人脸框
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
            emb = get_embedding(face_img)

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

# 实时识别人脸
def recognize_face(cap):
    db = load_database()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 通过 MediaPipe 检测人脸
        result = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.detections:
            face_img, box = extract_face_box(frame, result.detections[0])
            x1, y1, x2, y2 = box
            emb = get_embedding(face_img)
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

    choice = input("选择操作：1. 注册新的人脸  2. 实时识别 (1/2): ")
    if choice == "1":
        register_face(cap)  # 注册新的人脸
    elif choice == "2":
        recognize_face(cap)  # 实时识别人脸
    else:
        print("无效选择")

