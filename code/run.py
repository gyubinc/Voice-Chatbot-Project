# streamlit
from TTS.utils.synthesizer import Synthesizer
import streamlit as st
import librosa
from pydub import AudioSegment
import IPython
import numpy as np
import io
from voice_code.inference import normalize_multiline_text
from io import BytesIO
from scipy.io.wavfile import write
from chatbot_code.inference import answer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 오디오 모델 불러오기 함수
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

# 챗봇 모델 불러오기 함수
@st.cache_resource
def load_chatbot_model():
    model_path = 'chatbot_code/output/final_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

synthesizer = load_synthesizer()
symbols = synthesizer.tts_config.characters.characters
tokenizer, model = load_chatbot_model()

#페이지 구성
st.title('쿠빅 침착맨 초대석')
st.subheader("침착맨연KU소")
st.image("common_code/chim.jpg", width = 150)
st.subheader("반갑습니다 여러분의 귀염둥이 침착맨입니다.")
st.subheader("아래에 보낼 말을 입력해주세요")

#대화 실행
text = 0
with st.form(key="입력 form"):
    text = st.text_input("사용자")
    submitted = st.form_submit_button("submit")
    if submitted:
        
        with st.spinner("침착맨이 할 말을 생성중입니다."):
            answers = answer(model, tokenizer,synthesizer, text)
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
    
    

#실행
#streamlit run common_code/run.py --server.port 30007