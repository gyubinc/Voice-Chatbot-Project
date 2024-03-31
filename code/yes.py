# streamlit
from TTS.utils.synthesizer import Synthesizer
import streamlit as st
import IPython
import numpy as np
from io import BytesIO
from scipy.io.wavfile import write
from chatbot_code.inference import answer
import streamlit as st


@st.cache_resource
def load_synthesizer():
    synthesizer = Synthesizer(
    "../../content/drive/My Drive/Colab Notebooks/data/glowtts-v2/glowtts-v2-June-25-2023_02+36PM-3aa165ae/checkpoint_113000.pth.tar",
    "../../content/drive/My Drive/Colab Notebooks/data/glowtts-v2/glowtts-v2-June-25-2023_02+36PM-3aa165ae/config.json",
    None,
    "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/hifigan-v2-June-26-2023_07+23AM-3aa165ae/checkpoint_465000.pth.tar",
    "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/hifigan-v2-June-26-2023_07+23AM-3aa165ae/config.json",
    None,
    None,
    False,
    )
    return synthesizer


synthesizer = load_synthesizer()
symbols = synthesizer.tts_config.characters.characters



# 여기서 answers 받기
if answers:


    st.write(answers)
    wav = synthesizer.tts(answers, None, None)
    IPython.display.display(IPython.display.Audio(wav, rate=22050))
    wav_norm = np.int16(wav/np.max(np.abs(wav)) * 32767)
    # wav_norm을 wav 바이트로 변환하고 BytesIO 객체를 생성합니다.
    virtual_file = BytesIO()
    write(virtual_file, 22050, wav_norm)

    # virtual_file을 처음부터 다시 읽습니다.
    virtual_file.seek(0)
    st.audio(virtual_file.read(), format = 'audio/wav')