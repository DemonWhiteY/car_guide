from openai import OpenAI
import json
import os
from jinja2 import Template


prompt1path = "./llm/prompt1.txt"
prompt2path = "./llm/prompt2.txt"

WYW_API_KEY = "sk-4cf14a298ec04194b4bcc20f1bf68501"
def get_api_key():
    if 'DASHSCOPE_API_KEY' in os.environ:
        api_key = os.environ['DASHSCOPE_API_KEY']
    else:
        api_key = WYW_API_KEY
    return api_key

class SmartCarAssistant:
    def __init__(self, model="qwen-plus", retrycount=3):
        self.client =  OpenAI(
            api_key= get_api_key(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.history = [{"role": "system", "content": "你是一个俏皮可爱风格的车辆智能系统，负责和驾驶员或乘客对话，或帮助用户操作车内功能。"}]
        self.retrycount = retrycount

    def build_prompt(self, prompt_path, user_input, car_info):
        prompt_text = self.format_prompt(prompt_path, car_info, user_input)
        return [{"role": "user", "content": prompt_text}]
    
    def load_prompt_template(slef, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def format_prompt(slef, prompt_path, car_info, user_input):
        template = slef.load_prompt_template(prompt_path)
        j2_template = Template(template)
        return j2_template.render(car_info=json.dumps(car_info, ensure_ascii=False), user_input=user_input)
    
    def extract_json_block(self, text: str) -> str:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        else:
            return ""

    def run(self, user_input):
        car_info = ui.get_car_info()
        prompt1_text = self.format_prompt(prompt1path, car_info, user_input)
        messages = self.history + [{"role": "user", "content": prompt1_text}]
        for i in range(self.retrycount):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            reply = self.extract_json_block(response.choices[0].message.content.strip())
            try:
                data = json.loads(reply)
            except:
                return "服务器异常，请稍后再试。"
            if "reply" in data and "control" not in data:
                self.history.append({"role": "user", "content": prompt1_text})
                self.history.append({"role": "assistant", "content": reply})
                return data["reply"]
            
            # 如果是控制指令，执行所有命令
            control_list = data["control"]
            if isinstance(control_list, dict):
                control_list = [control_list]
            flag = True
            for cmd in control_list:
                flag &= ui.apply_control_command(cmd)
            if flag:
                break       

        # 第五步：更新后的车辆状态，构造 prompt_2
        car_info_updated = ui.get_car_info()
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
            return data2["reply"]
        except:
            return "服务器异常，请稍后再试"



##############################################
#########       一个临时的UI模块       #########
##############################################
from datetime import datetime
class UI:
    def __init__(self):
        self.hour = datetime.now()
        self.air_conditioner = False
        self.temperature = 24
        self.music = True
        self.volume = 5
        self.sunroof = False
        self.navigation = None
        self.seat_heat = False

    def bool_to_on_off(self, value):
        return "on" if value else "off"

    def get_car_info(self):
        car_info = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "air_conditioner(on/off)": self.bool_to_on_off(self.air_conditioner),
            "temperature(16-30)": self.temperature,
            "music(on/off)": self.bool_to_on_off(self.music),
            "volume(0-10)": self.volume,
            "sunroof(open/closed)": self.bool_to_on_off(self.sunroof),
            "navigation": self.navigation,
            "seat_heat(on/off)": self.bool_to_on_off(self.seat_heat),
        }
        return car_info

    def apply_control_command(self, control_dict):
        action = control_dict.get("action")
        target = control_dict.get("target")
        value = control_dict.get("value")

        # target 字段 → 类属性名
        valid_targets = {
            "air_conditioner": "air_conditioner",
            "temperature": "temperature",
            "music": "music",
            "volume": "volume",
            "sunroof": "sunroof",
            "navigation": "navigation",
            "seat_heat": "seat_heat"
        }
        attr = valid_targets.get(target)
        if attr is None:
            print(f"无效的目标字段: {target}")
            return False
        if action == "set":
            if isinstance(getattr(self, attr), bool):
                setattr(self, attr, value == "on")
            else:
                setattr(self, attr, value)
            return True

        elif action == "get":
            current_value = getattr(self, attr)
            if isinstance(current_value, bool):
                current_value = self.bool_to_on_off(current_value)
            return True
        else:
            print(f"未知操作类型: {action}")
            return False
        
ui = UI()