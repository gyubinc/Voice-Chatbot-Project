"""
SCE-TTS: 음성합성 추론 스크립트

이 문서는 SCE-TTS 프로젝트의 음성 합성 데모입니다.
원본 노트북: infer.ipynb

이 데모에 대한 더 자세한 정보는 아래 링크에서 확인하실 수 있습니다.  
https://sce-tts.github.io/
"""

import os
import sys
import re
from pathlib import Path
from unicodedata import normalize
import numpy as np
from scipy.io import wavfile as wavf

# 1. 구글 드라이브 마운트
# 음성합성을 위해 학습한 모델이 있는 구글 드라이브를 마운트합니다.  
# 마운트할 구글 드라이브 내에 다음 파일들이 존재하는지 꼭 확인해주세요.
# - `/Colab Notebooks/data/glowtts-v2/model_file.pth.tar`
# - `/Colab Notebooks/data/glowtts-v2/config.json`
# - `/Colab Notebooks/data/hifigan-v2/model_file.pth.tar`
# - `/Colab Notebooks/data/hifigan-v2/config.json`
# (존재하지 않는다면, glowtts-v2.zip, hifigan-v2.zip을 내려받아 준비해주세요.)
# 만약 아래에 `Enter your authorization code:`과 같은 메시지가 출력될 경우,  
# 같이 출력된 링크에 접속하여, 마운트할 구글 계정을 선택하신 후, 인증 코드를 복사하여 입력해주세요.

# 2. 필수 라이브러리 및 함수 불러오기
# 실행에 필요한 라이브러리 및 함수를 불러옵니다.
# 이 과정은 약 10분 정도 소요될 수 있습니다.

# 주의: 다음 패키지들이 설치되어 있어야 합니다.
# pip install folium==0.2.1
# pip install imgaug==0.2.5
# pip install librosa==0.8.1
# pip install pysbd
# pip install coqpit
# pip install unidecode
# pip install pypinyin

os.chdir('../../../')
current_dir = os.getcwd()
print(current_dir)

# TTS 및 g2pK 저장소 클론 및 설치
# 주의: 경로는 실제 환경에 맞게 수정이 필요합니다.
# os.chdir('content')
# os.system('git clone --depth 1 https://github.com/sce-tts/TTS.git -b sce-tts')
# os.system('git clone --depth 1 https://github.com/sce-tts/g2pK.git')
# os.chdir('TTS')
# os.system('pip install -q --no-cache-dir -e .')
# os.chdir('../g2pK')
# os.system('pip install -q --no-cache-dir "konlpy" "jamo" "nltk" "python-mecab-ko"')
# os.system('pip install -q --no-cache-dir -e .')

# g2pK 임포트
try:
    import g2pk
    g2p = g2pk.G2p()
except ImportError:
    print("g2pK가 설치되지 않았습니다. 설치 후 다시 시도해주세요.")
    g2p = None

# TTS 임포트
try:
    from TTS.utils.synthesizer import Synthesizer
except ImportError:
    print("TTS가 설치되지 않았습니다. 설치 후 다시 시도해주세요.")
    Synthesizer = None

# 텍스트 정규화 함수들
def normalize_text(text):
    """텍스트를 정규화합니다."""
    if g2p is None:
        raise ImportError("g2pK가 필요합니다.")
    
    text = text.strip()

    for c in ",;:":
        text = text.replace(c, ".")
    text = remove_duplicated_punctuations(text)

    text = jamo_text(text)

    text = g2p.idioms(text)
    text = g2pk.english.convert_eng(text, g2p.cmu)
    text = g2pk.utils.annotate(text, g2p.mecab)
    text = g2pk.numerals.convert_num(text)
    text = re.sub("/[PJEB]", "", text)

    text = alphabet_text(text)

    # remove unreadable characters
    text = normalize("NFD", text)
    text = "".join(c for c in text if c in symbols)
    text = normalize("NFC", text)

    text = text.strip()
    if len(text) == 0:
        return ""

    # only single punctuation
    if text in '.!?':
        return punctuation_text(text)

    # append punctuation if there is no punctuation at the end of the text
    if text[-1] not in '.!?':
        text += '.'

    return text


def remove_duplicated_punctuations(text):
    """중복된 문장부호를 제거합니다."""
    text = re.sub(r"[.?!]+\?", "?", text)
    text = re.sub(r"[.?!]+!", "!", text)
    text = re.sub(r"[.?!]+\.", ".", text)
    return text


def split_text(text):
    """텍스트를 문장 단위로 분리합니다."""
    text = remove_duplicated_punctuations(text)

    texts = []
    for subtext in re.findall(r'[^.!?\n]*[.!?\n]', text):
        texts.append(subtext.strip())

    return texts


def alphabet_text(text):
    """알파벳을 한글로 변환합니다."""
    text = re.sub(r"(a|A)", "에이", text)
    text = re.sub(r"(b|B)", "비", text)
    text = re.sub(r"(c|C)", "씨", text)
    text = re.sub(r"(d|D)", "디", text)
    text = re.sub(r"(e|E)", "이", text)
    text = re.sub(r"(f|F)", "에프", text)
    text = re.sub(r"(g|G)", "쥐", text)
    text = re.sub(r"(h|H)", "에이치", text)
    text = re.sub(r"(i|I)", "아이", text)
    text = re.sub(r"(j|J)", "제이", text)
    text = re.sub(r"(k|K)", "케이", text)
    text = re.sub(r"(l|L)", "엘", text)
    text = re.sub(r"(m|M)", "엠", text)
    text = re.sub(r"(n|N)", "엔", text)
    text = re.sub(r"(o|O)", "오", text)
    text = re.sub(r"(p|P)", "피", text)
    text = re.sub(r"(q|Q)", "큐", text)
    text = re.sub(r"(r|R)", "알", text)
    text = re.sub(r"(s|S)", "에스", text)
    text = re.sub(r"(t|T)", "티", text)
    text = re.sub(r"(u|U)", "유", text)
    text = re.sub(r"(v|V)", "브이", text)
    text = re.sub(r"(w|W)", "더블유", text)
    text = re.sub(r"(x|X)", "엑스", text)
    text = re.sub(r"(y|Y)", "와이", text)
    text = re.sub(r"(z|Z)", "지", text)

    return text


def punctuation_text(text):
    """문장부호를 한글로 변환합니다."""
    text = re.sub(r"!", "느낌표", text)
    text = re.sub(r"\?", "물음표", text)
    text = re.sub(r"\.", "마침표", text)

    return text


def jamo_text(text):
    """자모음을 한글로 변환합니다."""
    # 기본 자모음
    text = re.sub(r"ㄱ", "기역", text)
    text = re.sub(r"ㄴ", "니은", text)
    text = re.sub(r"ㄷ", "디귿", text)
    text = re.sub(r"ㄹ", "리을", text)
    text = re.sub(r"ㅁ", "미음", text)
    text = re.sub(r"ㅂ", "비읍", text)
    text = re.sub(r"ㅅ", "시옷", text)
    text = re.sub(r"ㅇ", "이응", text)
    text = re.sub(r"ㅈ", "지읒", text)
    text = re.sub(r"ㅊ", "치읓", text)
    text = re.sub(r"ㅋ", "키읔", text)
    text = re.sub(r"ㅌ", "티읕", text)
    text = re.sub(r"ㅍ", "피읖", text)
    text = re.sub(r"ㅎ", "히읗", text)
    text = re.sub(r"ㄲ", "쌍기역", text)
    text = re.sub(r"ㄸ", "쌍디귿", text)
    text = re.sub(r"ㅃ", "쌍비읍", text)
    text = re.sub(r"ㅆ", "쌍시옷", text)
    text = re.sub(r"ㅉ", "쌍지읒", text)
    text = re.sub(r"ㄳ", "기역시옷", text)
    text = re.sub(r"ㄵ", "니은지읒", text)
    text = re.sub(r"ㄶ", "니은히읗", text)
    text = re.sub(r"ㄺ", "리을기역", text)
    text = re.sub(r"ㄻ", "리을미음", text)
    text = re.sub(r"ㄼ", "리을비읍", text)
    text = re.sub(r"ㄽ", "리을시옷", text)
    text = re.sub(r"ㄾ", "리을티읕", text)
    text = re.sub(r"ㄿ", "리을피읍", text)
    text = re.sub(r"ㅀ", "리을히읗", text)
    text = re.sub(r"ㅄ", "비읍시옷", text)
    text = re.sub(r"ㅏ", "아", text)
    text = re.sub(r"ㅑ", "야", text)
    text = re.sub(r"ㅓ", "어", text)
    text = re.sub(r"ㅕ", "여", text)
    text = re.sub(r"ㅗ", "오", text)
    text = re.sub(r"ㅛ", "요", text)
    text = re.sub(r"ㅜ", "우", text)
    text = re.sub(r"ㅠ", "유", text)
    text = re.sub(r"ㅡ", "으", text)
    text = re.sub(r"ㅣ", "이", text)
    text = re.sub(r"ㅐ", "애", text)
    text = re.sub(r"ㅒ", "얘", text)
    text = re.sub(r"ㅔ", "에", text)
    text = re.sub(r"ㅖ", "예", text)
    text = re.sub(r"ㅘ", "와", text)
    text = re.sub(r"ㅙ", "왜", text)
    text = re.sub(r"ㅚ", "외", text)
    text = re.sub(r"ㅝ", "워", text)
    text = re.sub(r"ㅞ", "웨", text)
    text = re.sub(r"ㅟ", "위", text)
    text = re.sub(r"ㅢ", "의", text)

    return text


def normalize_multiline_text(long_text):
    """여러 줄 텍스트를 정규화합니다."""
    texts = split_text(long_text)
    normalized_texts = [normalize_text(text).strip() for text in texts]
    return [text for text in normalized_texts if len(text) > 0]


def synthesize(text, synthesizer):
    """텍스트를 음성으로 합성합니다."""
    if synthesizer is None:
        raise ImportError("Synthesizer가 초기화되지 않았습니다.")
    wavs = synthesizer.tts(text, None, None)
    return wavs


# 3. 학습한 모델 불러오기
# 학습한 Glow-TTS와 HiFi-GAN 모델을 불러옵니다.
# 만약 다른 체크포인트에서 불러오시려면 아래 코드에서 경로를 아래와 같이 적절하게 수정합니다.
# 주의: 경로는 실제 환경에 맞게 수정이 필요합니다.

synthesizer = None
symbols = None

def load_synthesizer(
    glowtts_model_path,
    glowtts_config_path,
    hifigan_model_path,
    hifigan_config_path
):
    """Synthesizer를 로드합니다."""
    global synthesizer, symbols
    
    if Synthesizer is None:
        raise ImportError("TTS가 설치되지 않았습니다.")
    
    synthesizer = Synthesizer(
        glowtts_model_path,
        glowtts_config_path,
        None,
        hifigan_model_path,
        hifigan_config_path,
        None,
        None,
        False,
    )
    symbols = synthesizer.tts_config.characters.characters
    return synthesizer


# 4. 음성 합성
# 실제 음성 합성을 수행합니다.
# `long_text`의 값을 변경하여 다른 문장의 합성도 시도해보실 수 있습니다.

def synthesize_text(text, output_path=None, sample_rate=22050):
    """텍스트를 음성으로 합성하고 저장합니다."""
    if synthesizer is None:
        raise ValueError("Synthesizer가 초기화되지 않았습니다. load_synthesizer()를 먼저 호출하세요.")
    
    normalized_texts = normalize_multiline_text(text)
    all_wavs = []
    
    for text in normalized_texts:
        wav = synthesizer.tts(text, None, None)
        all_wavs.append(wav)
    
    # 모든 음성을 연결
    combined_wav = np.concatenate(all_wavs)
    
    if output_path:
        wavf.write(output_path, sample_rate, combined_wav)
    
    return combined_wav


# 4.1 STT 실행
# 주의: 다음 패키지들이 필요합니다.
# apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg -y
# pip install PyAudio
# pip install SpeechRecognition
# pip install ffmpeg-python

def speech_to_text(audio_file_path, language='ko'):
    """음성 파일을 텍스트로 변환합니다."""
    try:
        import speech_recognition as sr
        
        r = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)
        
        try:
            text = r.recognize_google(audio, language=language)
            print(f"Google Speech Recognition thinks you said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None
    except ImportError:
        print("SpeechRecognition이 설치되지 않았습니다.")
        return None


if __name__ == "__main__":
    print("SCE-TTS 음성 합성 추론 스크립트")
    print("사용 예시:")
    print("1. load_synthesizer()로 모델을 로드하세요.")
    print("2. synthesize_text()로 텍스트를 음성으로 변환하세요.")
    
    # 예시 코드
    # texts = "반갑습니다"
    # for text in normalize_multiline_text(texts):
    #     wav = synthesizer.tts(text, None, None)
    #     # IPython.display.display(IPython.display.Audio(wav, rate=22050))

