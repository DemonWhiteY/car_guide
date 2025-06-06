# For prerequisites running the following sample, visit https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen


import os
import signal  # for keyboard events handling (press "Ctrl+C" to terminate recording)
import sys
from pathlib import Path
import json
#python car_guide/src/Voice/voice.py
# python -m car_guide.src.Voice.voice
# 绝对路径导入
from comm_objects import comm_objects  # 直接导入包模块[4](@ref)

from dashscope.audio.tts import SpeechSynthesizer

import dashscope
import pyaudio
from dashscope.audio.asr import *
mic = None
stream = None

vol_index = 0
# Set recording parameters
sample_rate = 16000  # sampling rate (Hz)
channels = 1  # mono channel
dtype = 'int16'  # data type
format_pcm = 'pcm'  # the format of the audio data
block_size = 3200  # number of frames per buffer
recognition = None
all_sentence = ""



def init_dashscope_api_key():
    """
        Set your DashScope API-key. More information:
        https://github.com/aliyun/alibabacloud-bailian-speech-demo/blob/master/PREREQUISITES.md
    """
    dashscope.api_key = 'sk-166fb0f2501140c8ad8e2058aaae67e9'  # set API-key manually


# Real-time speech recognition callback
class Callback(RecognitionCallback):
    def on_open(self) -> None:
        global mic
        global stream
        print('RecognitionCallback open.')
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True)

    def on_close(self) -> None:
        print(all_sentence)
        global mic
        global stream
        print('RecognitionCallback close.')
        stream.stop_stream()
        stream.close()
        mic.terminate()
        stream = None
        mic = None

    def on_complete(self) -> None:
        print('RecognitionCallback completed.')  # recognition completed

    def on_error(self, message) -> None:
        print('RecognitionCallback task_id: ', message.request_id)
        print('RecognitionCallback error: ', message.message)
        # Stop and close the audio stream if it is running
        if 'stream' in globals() and stream.active:
            stream.stop()
            stream.close()
        # Forcefully exit the program
        sys.exit(1)

    def on_event(self, result: RecognitionResult) -> None:
        global all_sentence
        sentence = result.get_sentence()
        if 'text' in sentence:
            print('RecognitionCallback text: ', sentence['text'])
            if RecognitionResult.is_sentence_end(sentence):

                all_sentence+=sentence['text']

                comm_objects.voice_comm.send_message(sentence['text'], 'llm')

                print(
                    'RecognitionCallback sentence end, request_id:%s, usage:%s'
                    % (result.get_request_id(), result.get_usage(sentence)))


def signal_handler(sig, frame):

    print('Ctrl+C pressed, stop recognition ...')
    # Stop recognition
    recognition.stop()
    print('Recognition stopped.')
    print(
        '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
        .format(
            recognition.get_last_request_id(),
            recognition.get_first_package_delay(),
            recognition.get_last_package_delay(),
        ))
    # Forcefully exit the program
    sys.exit(0)

def get_voice():
    global recognition
    recognition.start()

    signal.signal(signal.SIGINT, signal_handler)
    print("Press 'Ctrl+C' to stop recording and recognition...")

    while True:
        if stream:
            data = stream.read(3200, exception_on_overflow=False)
            recognition.send_audio_frame(data)
        else:
            break

    recognition.stop()


def tts(str):
    global recognition

    result = SpeechSynthesizer.call(model='sambert-zhichu-v1',
                                    text=str,
                                    sample_rate=48000,
                                    format='wav')
    global  vol_index
    vol_index= vol_index+1
    if result.get_audio_data() is not None:
        name = f"./voice_files/output{vol_index}.wav"
        with open(name, 'wb') as f:
            f.write(result.get_audio_data())
        print(f'SUCCESS: get audio data: %dbytes in {name}' %
              (sys.getsizeof(result.get_audio_data())))
    else:
        print('ERROR: response is %s' % (result.get_response()))

    return vol_index  # 由于保存的语音文件，我写了个下标，返回就对应语音文件后缀


def voice_process(data):
    print("custom")
    try:
        message = json.loads(data.decode('utf-8'))
        sender = message['sender']
        data_content = message['data']

        print(f"Voice received from {sender}: {data_content}")
        comm_objects.voice_comm.send_message(data_content, 'ui')

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Voice message processing error: {str(e)}")

def create_recognition():
    global recognition
    # Create the recognition callback
    callback = Callback()
    recognition = Recognition(
        model='paraformer-realtime-v2',
        format=format_pcm,
        sample_rate=sample_rate,
        semantic_punctuation_enabled=False,
        callback=callback
    )

# main function
if __name__ == '__main__':
    init_dashscope_api_key()
    print('Initializing ...')

    # Create the recognition callback
    #callback = Callback()


    create_recognition()


    get_voice()

    print(all_sentence)

