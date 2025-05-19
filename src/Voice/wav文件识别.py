from http import HTTPStatus
from dashscope.audio.asr import Recognition

import os
import signal  # for keyboard events handling (press "Ctrl+C" to terminate recording)
import sys

import dashscope
import pyaudio
from dashscope.audio.asr import *

# 若没有将API Key配置到环境变量中，需将下面这行代码注释放开，并将apiKey替换为自己的API Key
# import dashscope
# dashscope.api_key = "apiKey"
recognition = Recognition(model='paraformer-realtime-8k-v2',
                          format='wav',
                          sample_rate=16000,
                          # “language_hints”只支持paraformer-realtime-v2模型
                          language_hints=['zh', 'en'],
                          callback=None)
result = recognition.call('test.wav')
if result.status_code == HTTPStatus.OK:
    print('识别结果：')
    res = result.get_sentence()
    print(res)
    texts = [item['text'] for item in res if 'text' in item]
    print(texts)
else:
    print('Error: ', result.message)

print(
    '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
    .format(
        recognition.get_last_request_id(),
        recognition.get_first_package_delay(),
        recognition.get_last_package_delay(),
    ))