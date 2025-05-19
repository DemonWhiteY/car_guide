asr.py

最简单的文字生成语音接口，l2v输入文本，保存wav文件



voice.py

tts():

基本功能同上，输出用一个后缀记录，后面返回值是文件后缀，对应输出语音



get_voice()

从麦克风（设置默认频道1）输入语音

D:\anaconda3\envsltorchlpython.exe c:\users\Administrator\Desktop\软Ilvoice.py
Initializing ...
RecognitionCallback open.
Press 'Ctrl+C'to stop recording and recognition...
RecognitionCallback text:这
RecognitionCallback text:这是我
Recognitioncallback text:这是我说的第
RecognitionCallback text:
这是我说的第一句
ecognitioncallbacktext:
这是我说的第一句话。
:sentence end,request_id:b939102334ca4b7c9d3f5d406b4c4283, usage:{'duration': 3}ioncallback
text:ecodniti0n0alhack这
这是我ecoqnitioncallback text:
RecognitionCallback text:
这是我说的第
Recognitioncallback text: 这是我说的第二句话。Recognitioncallback sentence end, request_id:b939102334ca4b7c9d3f5d406b4c4283, usage:{'duration': 7}
Recognitioncallback text: 说完这
Recognitioncallback text:
说完这句话就
Recognitioncallback text:
说完这句话就可以闭
Recognitioncallback text:
说完这句话就可以闭嘴了。
Recognitioncallback sentence end, request_id:b939102334ca4b7c9d3f5d406b4c4283, usage:{'duration': 12}
Ctrl+C pressed, stop recognition ...
RecognitionCallback completed.
这是我说的第一句话。这是我说的第二句话。说完这句话就可以闭嘴了。
Recognitioncallback close.
Recognition stopped.

基本效果如上

每次说话停止会保存一句，输出一句

终止线程：signal

输入一个线程信号，就会进行线程终止，否则线程不断。

说话停止会识别。

线程应该开始就运行，控制台会持续输出识别结果（每一句话都能识别）



wav文件识别.py（大概率用不到）

wav转文字



需要配置APIKEY，环境同LLM

