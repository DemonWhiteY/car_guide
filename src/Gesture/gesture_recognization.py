import cv2
import mediapipe as mp
import os
import math

# 数据库：手势名 -> 指令码
DB_PATH = 'gesture_db.txt'

def load_gesture_db():
    db = {}
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            for line in f:
                gesture, code = line.strip().split(',')
                db[gesture] = code
    return db

def save_gesture_db(db):
    with open(DB_PATH, 'w') as f:
        for gesture, code in db.items():
            f.write(f"{gesture},{code}\n")

def classify_gesture(landmarks) -> str:
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]

    fingers = []
    for tip, pip in zip(tip_ids[1:], pip_ids[1:]):  # 食指到小指
        fingers.append(landmarks[tip].y < landmarks[pip].y)
    open_count = sum(fingers)

    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    thumb_up_y = landmarks[4].y < landmarks[3].y  # 指尖高于 IP
    thumb_up_x = abs(landmarks[4].x - landmarks[5].x) > 0.05  # 与掌根 X 差异较大
    thumb_open = thumb_up_y and thumb_up_x

    # 逻辑判断
    if index_up and middle_up and not ring_up and not pinky_up:
        return "yes"
    if index_up and thumb_open and not middle_up and not ring_up and not pinky_up:
        return "tick"
    if open_count == 0 and not thumb_open:
        return "fist"
    if thumb_open and open_count <= 1:
        return "thumbs_up"
    if open_count >= 4:
        return "wave"

    return "unknown"


def recognize_gesture_from_video(video_path, show=False):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(video_path)

    gesture_counter = {}
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            gesture = classify_gesture(landmarks)
            gesture_counter[gesture] = gesture_counter.get(gesture, 0) + 1

        frame_count += 1
        if show:
            cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    if not gesture_counter:
        return "unknown"

    return max((k for k in gesture_counter if k != "unknown"), key=gesture_counter.get)

def handle_command(command_str,file_path):
    db = load_gesture_db()
    parts = command_str.strip().split()

    if len(parts) < 2 or parts[0] != "change":
        return "[错误] 格式错误，应为: change code r/d/a"

    ncode = parts[1]
    action = parts[2]

    if action == 'r':
        video_path = file_path
        gesture = recognize_gesture_from_video(video_path)
        if gesture in db:
            print( "Confliction: Gesture already exists in the database.")
            return
        if gesture != "unknown":
            old_gestures = [gesture for gesture, code in db.items() if code == ncode]
            old_gesture = old_gestures[0]
            del db[old_gesture]
            db[gesture] = ncode
            save_gesture_db(db)
            print( f"[修改成功] {gesture} -> {ncode}")
        else:
           print( "[失败] 视频中未识别出有效手势")
        return

    elif action == 'a':
        video_path = file_path
        gesture = recognize_gesture_from_video(video_path)
        if gesture in db:
            print( "Confliction: Gesture already exists in the database.")
            return
        if gesture != "unknown":
            db[gesture] = ncode
            save_gesture_db(db)
            print(f"[添加成功] {gesture} -> {ncode}")
        else:
            print("[失败] 视频中未识别出有效手势")
        return
    
    elif action == 'd':
        to_delete = [g for g, c in db.items() if c == ncode]
        for g in to_delete:
            del db[g]
        save_gesture_db(db)
        print(f"[删除成功] 相关指令 {ncode} 已移除")
    else:
        print("[错误] 无效的操作类型（应为 r/d/a）")
    return


def main_recognition(video_path = "" , change = False , ins = " ",file_path = " "):
    if change:
       handle_command(ins,file_path)
    else: 
        db = load_gesture_db()
        gesture = recognize_gesture_from_video(video_path)
        print(f"[识别结果] 手势: {gesture}")

        if gesture in db:
            print(f"[输出指令] 指令码为: {db[gesture]}")
            return db[gesture]
        else:
            print("[警告] 未识别到数据库中的指令")
            return None
if __name__ == '__main__':
    main_recognition(video_path = "gestures/fist.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    main_recognition(video_path = "gestures/thumbs_up.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    main_recognition(video_path = "gestures/wave.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    main_recognition(video_path = "gestures/yes.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    main_recognition(video_path = "gestures/tick.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/1.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/2.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/3.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/4.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/4.mp4",change = True, ins = "change 005 a ",file_path = "new_gesture/tick.mp4 ")
