# coding=utf-8
import sys
from dashscope.audio.tts import SpeechSynthesizer
# import dashscope
# dashscope.api_key = "apiKey"
def l2v(text):
    result = SpeechSynthesizer.call(model='sambert-zhichu-v1',
                                    text=text,
                                    sample_rate=48000,
                                    format='wav')

    ret = 'output.wav'

    if result.get_audio_data() is not None:
        with open(ret, 'wb') as f:
            f.write(result.get_audio_data())
        print('SUCCESS: get audio data: %dbytes in output.wav' %
              (sys.getsizeof(result.get_audio_data())))
    else:
        print('ERROR: response is %s' % (result.get_response()))


    return ret

if __name__ == '__main__':
    l2v("我爱玩原神")