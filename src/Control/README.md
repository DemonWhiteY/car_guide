# 通信传输模块

## 功能

- 将不同模块的消息转发到目标模块

## 采用方式

- 服务器-客户端模式

其中 `control` 模块为服务器，其余的 `UI`、`LLM`、`Face`、`Gesture` 和 `Voice` 五个模块作为客户端。

- 采用socket通信：源客户端->服务器->目的客户端

## 消息格式

```python
message = {
            'target': target,
            'sender': self.identity,
            'data': data
        }
```

其中target为目的地址，sender为源地址，取值为 `control`、 `ui`、`llm`、`face`、`gesture` 和 `voice` 。

data为传输数据。

## 代码目录

```txt
/Control/
├── comm_objects.py         # 通信对象管理
├── base_client.py          # 客户端基类
├── client_classes.py       # 各模块客户端类
├── control.py              # 控制服务器
├── main.py                 # 主程序入口
├── face_module.py          # 人脸模块（只发送）
├── gesture_module.py       # 手势模块（只发送）
├── voice_module.py         # 语音模块（发送+接收）
├── llm_module.py           # LLM模块（发送+接收）
└── ui_module.py            # UI模块（只接收）
```

- 只需要把对应模块的moodule稍加修改对应上自己的接口即可

- 注意：由于测试需要，我目前在各个module里面写了一个模拟函数到时候直接视情况替换成接口即可
