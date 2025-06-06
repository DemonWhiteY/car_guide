from comm_objects import comm_objects
import time
import random
import json
import os
from openai import OpenAI
import json
import os
from jinja2 import Template
from datetime import datetime
import threading

base_dir = os.path.dirname(os.path.abspath(__file__))
prompt1path = os.path.join(base_dir, "prompts", "prompt1.txt")
prompt2path = os.path.join(base_dir, "prompts", "prompt2.txt")

WYW_API_KEY = "sk-4cf14a298ec04194b4bcc20f1bf68501"
def get_api_key():
    if 'DASHSCOPE_API_KEY' in os.environ:
        api_key = os.environ['DASHSCOPE_API_KEY']
    else:
        api_key = WYW_API_KEY
    return api_key

class CarInfo:
    def __init__(self):
        self.time = datetime.now()

        # 硬件状态
        self.light = False                  # 灯光: True=开, False=关
        self.left_front_door = False        # 左前门: True=开, False=关
        self.right_front_door = False       # 右前门: True=开, False=关
        self.left_rear_door = False         # 左后门: True=开, False=关
        self.right_rear_door = False        # 右后门: True=开, False=关
        self.trunk = False                  # 后备箱: True=开, False=关
        self.air_conditioner_mode = "off"   # 空调模式: "cold"/"hot"/"off"
        self.temperature = 24               # 空调温度: 10~30

        # 媒体信息
        self.media_volume = 50              # 媒体音量: 0~100
        self.navigation = None              # 导航位置: string
        self.playing_song = None            # 当前播放歌曲: string
        self.is_playing = True              # True=播放, False=暂停
        self.music_control = "1"          

    def get_car_info(self):
        car_info = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "light": self.light,
            "left_front_door": self.left_front_door,
            "right_front_door": self.right_front_door,
            "left_rear_door": self.left_rear_door,
            "right_rear_door": self.right_rear_door,
            "trunk": self.trunk,
            "air_conditioner_mode(cold/hot/off)": self.air_conditioner_mode,
            "temperature(10-30)": self.temperature,
            "media_volume(0-100)": self.media_volume,
            "navigation": self.navigation,
            "playing_song": self.playing_song,
            "is_playing": self.is_playing,
            "music_control": "该字段当前值不重要，它可以被设置为next和prev两种。"
        }
        return car_info
    
    def apply_control_command(self, control_dict):
        target = control_dict.get("target", "").lower()
        value = control_dict.get("value")

        # 关键词匹配 target 到 CarInfo 内部字段
        target_mapping = {
            "light": "light",
            "left_front_door": "left_front_door",
            "right_front_door": "right_front_door",
            "left_rear_door": "left_rear_door",
            "right_rear_door": "right_rear_door",
            "trunk": "trunk",
            "air_conditioner_mode": "air_conditioner_mode",
            "temperature": "temperature",
            "media_volume": "media_volume",
            "navigation": "navigation",
            "playing_song": "playing_song",
            "is_playing": "is_playing",
        }

        attr = None
        for key, field in target_mapping.items():
            if key in target:
                attr = field
                break
        
        if attr is None or not hasattr(self, attr):
            print(f"无效或未识别的目标字段: {target}")
            return ""

        # 修改本地属性
        current_type = type(getattr(self, attr))
        try:
            if attr == "air_conditioner_mode":
                if value not in ["cold", "hot", "off"]:
                    print(f"无效空调模式取值: {value}")
                    return ""
            elif attr == "music_control":
                if isinstance(value, str):
                    value = value.lower()
                    if "next" in value or "下一" in value:
                        value = 1
                    elif "prev" in value or "上一" in value:
                        value = 0
                    else:
                        print(f"无效的music_control取值: {value}")
                        return ""
            elif current_type is bool:
                # 统一使用 0 / 1 代表 False / True
                if isinstance(value, str):
                    value = value.lower()
                    value = 1 if value in ["on", "true", "播放", "开"] else 0
                elif isinstance(value, (int, float)):
                    value = 1 if value else 0
                elif isinstance(value, bool):
                    value = 1 if value else 0
            elif current_type is int:
                value = int(value)
            elif current_type is str:
                value = str(value)
        except:
            print(f"值转换错误: {value} → {current_type}")
            return ""

        setattr(self, attr, value)
        print(f"已本地修改 {attr} 为 {value}")

        # 将 target 映射为ui模块标准名（用于发送到 UI 模块）
        ui_mapping = {
            "light": "Light",
            "left_front_door": "LF_door",
            "right_front_door": "RF_door",
            "left_rear_door": "LB_door",
            "right_rear_door": "RB_door",
            "trunk": "B_door",
            "air_conditioner_mode": "AC",
            "temperature": "Tem",
            "media_volume": "Voice",
            "navigation": "Position",
            "playing_song": "Music",
            "is_playing": "Play",
            "music_control": "Next"
        }
        ui_name = ui_mapping.get(attr, attr)
        # 构造发送给 UI 模块的消息
        ui_message = {
            "target": ui_name,
            "value": value
        }
        return ui_message

class LLMProcessor:
    def __init__(self):
        # 获取LLM通信对象
        self.llm_comm = comm_objects.llm_comm
        self.llm_comm.handle_message = self.custom_handle_message
        self.client =  OpenAI(
            api_key= get_api_key(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen-plus"
        self.retrycount = 3
        self.history = [
            {"role": "system", 
             "content": """
             你是一个俏皮可爱风格的多模态车辆智能系统，负责和驾驶员或乘客对话、帮助用户操作车内功能、提醒驾驶员或乘客等。
             每个对话周期都是由用户语音触发的，触发后我会将用户所说的内容发送给你，你需要结合用户的头部姿态和手势（如果有的话），精准理解用户需求，并作出操作或回复。
             头部姿态和手势的变化不会触发对话。"}
                """
            }
        ]

        # 人脸存储
        self.face_info = {}  
        # 当前手势
        self.gesture = None
        # 车辆信息
        self.carinfo = CarInfo()
        # 速度
        self.speed = 70
        self.speed_limit = 80
        self.fuel_quantity = 80
        self.speed_thread = threading.Thread(target=self.automatic_module, daemon=True)
        self.speed_thread.start()


    
    def automatic_module(self):
        # 初始化
        cmd = {
            "target": "汽车速度", 
            "value": self.speed
        }
        self.llm_comm.send_message(cmd, "ui")
        cmd = {
            "target": "剩余油/电", 
            "value": self.fuel_quantity
        }
        self.llm_comm.send_message(cmd, "ui")
        count = 1
        while True:
            # 更新速度
            count -= 1
            if self.speed > self.speed_limit:
                delta = random.randint(-5, 1)
            else:
                delta = random.randint(-5, 5)
            self.speed = max(0, self.speed + delta)
            cmd = {
                "target": "汽车速度", 
                "value": self.speed
            }
            self.llm_comm.send_message(cmd, "ui")
            if self.speed > self.speed_limit and count < 0:
                data = f"限速{self.speed_limit}，您已超速，当前时速{self.speed}。"
                self.llm_comm.send_message(data, "voice")
                count = 10
            
            # 更新油量
            if random.random() > 0.9:
                self.fuel_quantity = max(0, self.fuel_quantity - 1)
                cmd = {
                    "target": "剩余油/电", 
                    "value": self.fuel_quantity
                }
                self.llm_comm.send_message(cmd, "ui")

            time.sleep(1)


    def custom_handle_message(self, data):
        """
        data格式:
            message = {
                'target': target,
                'sender': self.identity,
                'data': data
            }
        """
        data = json.loads(data.decode('utf-8'))
        target = data.get('target')
        assert(target == "llm")
        sender = data.get('sender')
        # 解析内层 data，先解析 JSON 字符串为字典
        data_str = data.get('data')
        print(data_str)
        try:
            if sender == "face":
                message = json.loads(data_str)
                user_name = message.get("user_name")
                user_identity = message.get("user_identity")
                actions = message.get("actions")
                if user_name:
                    if actions == "无明显行为":
                        if user_name in self.face_info:
                            # print(f"Removing {user_name} due to '无明显行为'")
                            del self.face_info[user_name]
                    else:
                        self.face_info[user_name] = {
                            "identity": user_identity,
                            "actions": actions
                        }
                        # print(f"Updated face_info for {user_name}: {self.face_info[user_name]}")

            elif sender == "gesture":
                message = json.loads(data_str)
                ges = message.get("ges")
                if ges == "":
                    self.gesture_status = None
                else:
                    self.gesture_status = ges
            elif sender == "voice":
                threading.Thread(target=self.askllm, args=(data_str, ), daemon=True).start()
        except (json.JSONDecodeError, KeyError) as e:
            print(f"LLM message processing error: {str(e)}")
    
    def load_prompt_template(slef, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def format_prompt(self, prompt_path, car_info, user_input):
        template = self.load_prompt_template(prompt_path)
        j2_template = Template(template)
        return j2_template.render(
            car_info=json.dumps(car_info, ensure_ascii=False),
            user_input=user_input,
            gesture=self.gesture,
            pose=self.face_info
        )
        
    def extract_json_block(self, text: str) -> str:
        count = 0
        start = -1
        for i, char in enumerate(text):
            if char == '{':
                if count == 0:
                    start = i
                count += 1
            elif char == '}':
                count -= 1
                if count == 0 and start != -1:
                    return text[start:i+1]
        return ""

    def askllm(self, user_input):
        car_info = self.carinfo.get_car_info()
        prompt1_text = self.format_prompt(prompt1path, car_info, user_input)

        messages = self.history + [{"role": "user", "content": prompt1_text}]

        for i in range(self.retrycount):
            #print(messages)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            #print(response)
            reply = self.extract_json_block(response.choices[0].message.content.strip())
            try:
                data = json.loads(reply)
            except:
                if i == self.retrycount - 1:
                    data = "服务器异常，请稍后再试。"
                    self.llm_comm.send_message(data, "voice")
                    return
                else:
                    continue

            if "reply" in data and "control" not in data:
                self.history.append({"role": "user", "content": prompt1_text})
                self.history.append({"role": "assistant", "content": reply})
                self.llm_comm.send_message(data["reply"], "voice")
                return
            
            # 如果是控制指令，执行所有命令
            control_list = data["control"]
            if isinstance(control_list, dict):
                control_list = [control_list]
            flag = True
            for cmd in control_list:
                mess = self.carinfo.apply_control_command(cmd)
                if mess:
                    self.llm_comm.send_message(mess, "ui")
                else:
                    flag = False
            if flag:
                break

        # 第五步：更新后的车辆状态，构造 prompt_2
        car_info_updated = self.carinfo.get_car_info()
        prompt2_text = self.format_prompt(prompt2path, car_info_updated, user_input)
        messages += [{"role": "assistant", "content": reply}]
        messages +=[{"role": "user", "content": prompt2_text}]
        response2 = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        reply2 = self.extract_json_block(response2.choices[0].message.content.strip())
        try:
            data2 = json.loads(reply2)
            self.history.append({"role": "user", "content": prompt1_text})
            self.history.append({"role": "assistant", "content": reply})
            self.history.append({"role": "user", "content": prompt2_text})
            self.history.append({"role": "assistant", "content": reply2})
            self.llm_comm.send_message(data2["reply"], "voice")
            return
        except:
            data = "服务器异常，请稍后再试。"
            self.llm_comm.send_message(data, "voice")
            return


def start_llm():
    """启动LLM模块"""
    print("Starting LLM module...")
    LLMProcessor()
    while True:
        time.sleep(10)


def start():
    start_llm()


if __name__ == "__main__":
    start_llm()


