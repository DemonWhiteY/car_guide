import cv2
import mediapipe as mp
import os
import time

from car_guide.src.Control import comm_objects

# 数据库：手势名 -> 指令码
DB_PATH = 'gesture_db.txt'
ges = {
    "000": '{"ges": ""}',
    "001": '{"ges": "用户要暂停音乐"}',
    "002": '{"ges": "用户表示了赞同"}',
    "003": '{"ges": "用户表示了拒绝"}',
    "004": '{"ges": "用户需要打开空调"}',
    "005": '{"ges": "用户需要播放音乐"}'
}

def load_gesture_db():
    """从文本文件加载手势数据库
    
    返回:
        dict: 手势名称到指令码的映射字典
    """
    db = {}
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            for line in f:
                gesture, code = line.strip().split(',')
                db[gesture] = code
    return db

def save_gesture_db(db):
    """将手势数据库保存到文本文件
    
    参数:
        db (dict): 手势名称到指令码的映射字典
    """
    with open(DB_PATH, 'w') as f:
        for gesture, code in db.items():
            f.write(f"{gesture},{code}\n")

def classify_gesture(landmarks) -> str:
    """根据手部关键点坐标进行手势分类
    
    参数:
        landmarks (list): mediapipe手部关键点坐标列表
        
    返回:
        str: 识别出的手势名称，可能值包括：
            'yes', 'tick', 'fist', 'thumbs_up', 'wave', 'unknown'
    """
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
    """从视频中识别手势
    
    参数:
        video_path (str): 视频文件路径
        show (bool): 是否实时显示视频画面
        
    返回:
        str: 识别出的主要手势名称
    """
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
    """处理用户指令，修改手势数据库
    
    参数:
        command_str (str): 用户输入的指令字符串
        file_path (str): 手势视频文件路径
        
    返回:
        None: 结果通过print输出
    """
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

        if gesture in db:
            print(ges[db[gesture]])
            #send_command(ges[db[gesture]])
            comm_objects.gesture_comm.send_message(ges[db[gesture]], 'llm')
            return db[gesture]
        else:
            print("[警告] 未识别到数据库中的指令")
            return None


def camera_gesture_recognition():
    """通过摄像头实时识别手势并输出指令"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    db = load_gesture_db()
    
    gesture_counter = {}
    last_command_time = time.time()
    current_command = None
    pr_command = None
    command_display_duration = 3  # 显示命令的持续时间(秒)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("无法读取摄像头画面")
            break
        
        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        gesture_name = "unknown"
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            gesture_name = classify_gesture(landmarks)
            
            # 更新手势计数器
            gesture_counter[gesture_name] = gesture_counter.get(gesture_name, 0) + 1
        
        # 显示当前帧识别结果
        cv2.putText(frame, f"Current: {gesture_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 每1秒确定一次主要手势
        current_time = time.time()
        if current_time - last_command_time > 1.0 and gesture_counter:
            # 找到出现次数最多的手势
            main_gesture = max(gesture_counter, key=gesture_counter.get)
            
            # 忽略"unknown"手势
            if main_gesture != "unknown":
                if main_gesture in db:
                    current_command = db[main_gesture]
                else:
                    current_command = "未定义指令"
                
                # 重置计数器和时间
                gesture_counter = {}
                last_command_time = current_time
            else:
                current_command = "000"
                gesture_counter = {}
                last_command_time = current_time
        
        # 显示当前指令
        if current_command and (current_time - last_command_time < command_display_duration):
            cv2.putText(frame, f"Command: {current_command}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if pr_command!= current_command:
                pr_command = current_command
                print(f"[输出指令] 指令为: {ges[current_command]}")
                #send_command(ges[current_command])
                comm_objects.gesture_comm.send_message(ges[current_command], 'llm')
        
        # 显示画面
        cv2.imshow('Hand Gesture Recognition', frame)
        
        # 按ESC退出
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def gesrec(use_camera=False, video_path="", change=False, ins="", file_path=""):
    """
    手势识别主函数
    
    参数:
        use_camera (bool): 是否使用摄像头实时识别
        video_path (str): 视频文件路径(当use_camera=False时使用)
        change (bool): 是否执行数据库修改操作
        ins (str): 修改指令(当change=True时使用)
        file_path (str): 手势视频文件路径(当change=True时使用)
    """
    if use_camera:
        camera_gesture_recognition()
    else:
        if not video_path:
            print("错误: 视频模式需要指定video_path参数")
            return
            
        main_recognition(
            video_path=video_path,
            change=change,
            ins=ins,
            file_path=file_path
        )
if __name__ == '__main__':
    gesrec(False,video_path = "gestures/fist.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    gesrec(False,video_path = "gestures/thumbs_up.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    gesrec(False,video_path = "gestures/wave.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    gesrec(False,video_path = "gestures/yes.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    gesrec(False,video_path = "gestures/tick.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    gesrec(True)
    # import time
    # import cProfile
    # import pstats
    # import matplotlib.pyplot as plt

    # plt.rcParams['font.sans-serif'] = ['SimHei'] 
    # plt.rcParams['axes.unicode_minus'] = False 

    # # 测试用例列表
    # test_cases = [
    #     ("gestures/fist.mp4", "new_gesture/tick.mp4"),
    #     ("gestures/thumbs_up.mp4", "new_gesture/tick.mp4"),
    #     ("gestures/wave.mp4", "new_gesture/tick.mp4"),
    #     ("gestures/yes.mp4", "new_gesture/tick.mp4"),
    #     ("gestures/tick.mp4", "new_gesture/tick.mp4")
    # ]

    # # 性能数据存储
    # runtime_data = []
    # profiler = cProfile.Profile()

    # # 遍历所有测试用例
    # for idx, (video_path, file_path) in enumerate(test_cases):
    #     case_name = video_path.split('/')[-1].split('.')[0]  # 提取手势名称
        
    #     # 使用cProfile进行分析
    #     profiler.enable()
    #     start_time = time.time()
        
    #     # 原函数调用
    #     main_recognition(
    #         video_path=video_path,
    #         change=False,
    #         ins="change 001 r",
    #         file_path=file_path
    #     )
        
    #     # 记录性能数据
    #     elapsed = time.time() - start_time
    #     profiler.disable()
    #     runtime_data.append((case_name, elapsed))
        
    #     # 保存cProfile结果
    #     stats = pstats.Stats(profiler)
    #     stats.dump_stats(f"perf_{case_name}.prof")

    # # 生成对比图表
    # plt.figure(figsize=(10, 6))
    # names = [x[0] for x in runtime_data]
    # times = [x[1] for x in runtime_data]
    
    # bars = plt.bar(names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    # plt.title('手势识别性能对比', fontsize=14)
    # plt.xlabel('测试用例', fontsize=12)
    # plt.ylabel('运行时间 (秒)', fontsize=12)
    # plt.xticks(rotation=45)
    
    # # 添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #              f'{height:.2f}',
    #              ha='center', va='bottom')
    
    # plt.tight_layout()
    
    # # 保存并显示图表
    # plt.savefig('performance_comparison.png', dpi=300)
    # plt.show()

    # # 打印控制台报告
    # print("\n=== 性能测试报告 ===")
    # print(f"{'测试用例':<15} | {'运行时间(s)':<10}")
    # print("-" * 30)
    # for name, t in runtime_data:
    #     print(f"{name:<15} | {t:.2f}")
    # main_recognition(video_path = "gestures/fist.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/thumbs_up.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/wave.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/yes.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/tick.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/1.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/2.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/3.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    # main_recognition(video_path = "gestures/4.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
    #main_recognition(video_path = "gestures/4.mp4",change = True, ins = "change 005 a ",file_path = "new_gesture/tick.mp4 ")
