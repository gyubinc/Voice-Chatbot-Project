# streamlit
import streamlit as st
import librosa
from pydub import AudioSegment
import numpy as np
import io

def wav_to_bytes(wav_array, sample_rate=22050):
    audio_segment = AudioSegment(
        wav_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=wav_array.dtype.itemsize, 
        channels=1
    )
    byte_io = io.BytesIO()
    audio_segment.export(byte_io, format="wav")
    byte_wav = byte_io.getvalue()
    byte_io.close()
    return byte_wav

def main(wav_path):
    st.title('치무차쿠맨 프로젝트')
    st.image("common_code/chim.jpg")
    st.subheader("반갑습니다 여러분의 귀염둥이 침착맨입니다.")
    # Use a button to trigger playing the wav file
    if st.button('Play'):
        wav_file_path = wav_path
        wav, sr = librosa.load(wav_file_path, sr=22050)
        wav_bytes = wav_to_bytes(wav)
        st.audio(wav_bytes, format='audio/wav')
        
main('자기소개.wav')