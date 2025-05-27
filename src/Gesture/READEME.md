## settings
在gesture文件夹下运行：
```bash
pip install -r requirements.txt
```

## 功能

- 输入：视频文件路径，mp4文件，字符串/视频文件+指令
- 输出：指令码，字符串类型，如001，另有一些错误异常输出

- 目前支持的手势：fist、thumbs_up、wave、yes、tick

- 其中有gesture_db.txt作为数据库，存储手势及对应的指令码，格式为（手势名，指令码）

- 已有录制好的测试视频于gesture文件夹


## 使用
已有完整的接口函数，main_recognition，只需进行调用即可

```python
"""主识别函数，支持两种模式：
        1. 手势识别模式（change=False）
        2. 数据库修改模式（change=True）
    
    参数:
        video_path (str): 要识别的视频文件路径
        change (bool): 是否为修改数据库模式
        ins (str): 修改指令字符串
        file_path (str): 新手势视频文件路径
        
    返回:
        str/None: 识别模式返回指令码，修改模式返回None
    """
#普通调用
main_recognition(video_path = "gestures/fist.mp4",change = False , ins = "change 001 r ",file_path = "new_gesture/tick.mp4 ")
#修改调用
main_recognition(video_path = "gestures/4.mp4",change = True, ins = "change 005 a ",file_path = "new_gesture/tick.mp4 ")
#指令格式
#change 指令码 a/r/d
#其中 a为add，即为数据库添加一个指令，须在第四个参数处提供一个此新手势对应的视频
#其中 d为delete，即为数据库删除一个指令，第四参数可忽略
#其中 r为replace，即为数据库替换一个指令，须在第四个参数处提供一个此新手势对应的视频，即指令码不修改，只修改指令码对应的手势
```

