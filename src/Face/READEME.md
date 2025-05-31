### 权重文件下载地址

**权重文件因为文件过大（约130mb）无法push入github，下面给出下载地址**

根据需要运行的代码按需下载，具体配置路径在代码文件夹下的"注.txt"说明。

Facent：https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
Buffalo_l：https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip


### 文件组织说明

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