**提供三个版本代码，各个代码的说明如下（详细信息在文件夹的txt中有说明）**

- initial：采用mediapipe快速裁剪人脸，送入facent（跳过内置检测）直接提取特征，速度快，效果不好
- crop_getfeature：采用mediapipe快速裁剪人脸，送入buffalo_l（跳过内置检测）直接提取特征，速度快，效果有一定提升但还是不理想
- final_callmodel(best)：最后实际采用的方法，直接把摄像头图像送入buffalo_l，内置检测+内置转换提取特征，速度会降慢，效果有很大提升。

人脸分类均直接使用特征距离(欧几里得距离)的方法。

**权重文件因为文件过大（约130mb）无法push入github，下面给出下载地址**

Facent：https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
Buffalo_l：https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip