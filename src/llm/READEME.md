## settings
在llm文件夹下运行：
```bash
pip install -r requirements.txt
```

根据运行目录，修改ai.py前两行：
```python
prompt1path = "./llm/prompt1.txt"
prompt2path = "./llm/prompt2.txt"
```

## 功能

- 输入：用户说的话（文本，由语音模块传入）。
- 输出：系统回答（文本，返回给语音模块）。

- 系统还会对车辆信息进行修改：
  - 该部分后期对接UI版块，目前在代码中体现为`class UI`
  - 1、处理时，会获取车辆个部件信息(`get_car_info`，返回dict)。
  - 2、系统发送指令给UI（`apply_control_command`，返回bool）。
  - 指令格式：
    ```json
    {
        "control": {
            "action": "set" / "get"
            "target": "air_conditioner" / "temperature" / ...,
            "value": "on" / 26 / ...
        }
    }
    ```
  - 目前支持的汽车信息：时间、空调开关、空调温度、音乐开关、音量、导航目的地、天窗、座椅温度。（详见`class UI`类的成员变量）


## 使用
```python
from ai import SmartCarAssistant
# 创建一次就行
ai = SmartCarAssistant()

# ....... 语音处理过程，假设text是最后语音转文字后的文本：
text = "你好啊，给我讲个笑话吧"
# 调用方法：
respond = ai.run(text)
# respond就是回复
print(respond)
```