### 权重文件下载地址

**权重文件因为文件过大（约130mb）无法push入github，下面给出下载地址**

要运行模块代码，需要将下载的Buffalo_l文件夹（解压缩）移动到user/.insightface/models下。

Facent：https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
Buffalo_l：https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip


如果要运行别的历史代码，根据需要运行的代码按需下载，具体配置路径在代码文件夹下的"注.txt"说明。



### 模块使用说明

模块代码存储在face.py中。引入control模块后，要运行示例测试中的main函数，需要在上层目录运行：
```python
# 当前目录为src
python -m Face.face

# 当前目录为car_guide
python -m src.Face.face

# 当前目录为car_guide的上级目录
python -m car_guide.src.Face.face
```

对外提供face_entry接口, 直接调用即可，参数如下：
def face_entry(is_camera, video_path, is_record, name, identity):
    
- is_camera: 为True表示调用摄像头, 为False表示以文件模式运行(需要传入video_path);
- video_path: 文件模式下的摄像头调用
- is_record: 选择是识别模式还是记录模式，记录模式需要传入name(不能为None和Unknown) 和 identity(Passenger/Driver)。如果人名在数据库中存在，会直接覆盖原来的数据。
- name identity: 如上所述，在记录模式下需要传入的参数

**识别模式下，如果是摄像头会一直运行；如果选择放文件会一直循环播放，直到按下esc为止；**

**录入模式下，采样到足够多的样本后自动退出（视频文件在未采样完成前会一直重复播放）。**

**传入的文件路径为相对路径，数据库文件固定保存在face文件夹下的子级目录中。**





### 历史文件组织说明

**main中提供最终用于使用的版本代码final_video_version和final_camera_version，用户只需要用这两个版本**

- 基于history中的final_callmodel(best)修改。
- print信息采用实时发送，并把信息打包成json字符串。
- 添加更多代码健壮性。
- camera调用摄像头进行实时识别，video版本传入video文件进行解析。

---

**history中提供多个可选的版本代码，各个代码的说明如下（具体配置信息和使用说明在文件夹中的requirements.txt中有说明）**

- initial：采用mediapipe快速裁剪人脸，送入facent（跳过内置检测）直接提取特征，速度快，效果不好
- crop_getfeature：采用mediapipe快速裁剪人脸，送入buffalo_l（跳过内置检测）直接提取特征，速度快，效果有一定提升但还是不理想
- final_callmodel(best)：最后实际采用的方法，直接把摄像头图像送入buffalo_l，内置检测+内置转换提取特征，速度会降慢，效果有很大提升。

人脸分类均直接使用特征距离(欧几里得距离)的方法。