"""
HiFi-GAN 학습 스크립트

이 스크립트는 HiFi-GAN 모델을 학습하기 위한 코드입니다.
원본 노트북: train_hifigan.ipynb
"""

import os
import sys
from pathlib import Path

# 1. 할당된 GPU 확인
# 현재 런타임에 할당된 GPU를 확인합니다.
# 만약, `GPU: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.` 라는 메시지가 출력된다면, 
# 위쪽 메뉴에서 `런타임 -> 런타임 유형 변경`을 클릭하고 하드웨어 가속기를 `GPU`로 변경하여 저장한 후 다시 실행해주세요.

GPU_NAME = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()
os.environ['GPU_NAME'] = GPU_NAME
print(f'GPU: {GPU_NAME}')

# 3. 필수 라이브러리 및 함수 불러오기
# 실행에 필요한 라이브러리 및 함수를 불러옵니다.
# 이 과정은 약 10분 정도 소요될 수 있습니다.

os.chdir('../../../')
current_dir = os.getcwd()
print(current_dir)

import sys
from pathlib import Path

# 가상 환경 생성 및 패키지 설치
# 주의: 실제 실행 환경에 맞게 수정이 필요할 수 있습니다.
# python3 -m venv tts-env
# source tts-env/bin/activate
# pip install --upgrade numpy
# pip install TTS==0.0.14.1

# 4. 학습할 데이터셋 불러오기
# 학습에 사용할 음성 데이터를 구글 드라이브에서 가져옵니다.
# 주의: 경로는 실제 환경에 맞게 수정이 필요합니다.
# os.chdir('content')
# os.system('cp "../../content/drive/My Drive/Colab Notebooks/data/filelists.zip" ./filelists.zip')
# os.system('rm -rf ./filelists')
# os.system('unzip -q filelists.zip -d ./filelists')

# 5. 사전 학습 데이터 불러오기
# 사전 학습 데이터가 구글 드라이브에 존재하지 않을 경우,  
# SleepingCE Speech Dataset의 사전 학습 데이터를 내려받습니다.
# 주의: 경로는 실제 환경에 맞게 수정이 필요합니다.
# os.chdir('content')
# os.system('mkdir -p "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2"')
# if not Path("../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/config.json").exists():
#     os.system('gdown --id 1vRxp1RH-U7gSzWgyxnKY4h_7pB3tjPmU -O hifigan-v2.zip')
#     os.system('unzip -q hifigan-v2.zip -d ./')
#     os.system('cp -R ./hifigan-v2/* "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/"')

# 통계 계산
# os.chdir('content')
# if not Path("../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/scale_stats_new.npy").exists():
#     os.system('python TTS/bin/compute_statistics.py "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/config.json" "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/scale_stats_new.npy" --data_path "../../content/TTS/filelists/wavs/"')

# 7. HiFi-GAN 학습 진행
# 실제 HiFi-GAN 학습을 진행합니다.
# 학습이 정상적으로 진행되면, 이 셀은 종료되지 않고 계속 실행되는 상태를 유지합니다.
# 학습 진행 정도를 시각적으로 확인하시려면 바로 위의 TensorBoard를 참고해주세요.
# 학습된 결과를 확인하고 싶으시다면, 가장 최근 체크포인트를 [SCE-TTS: 음성합성 데모]에서 불러와 확인할 수 있습니다.
# 처음 학습을 수행할 때에는 아래 셀을 그대로 실행하면 됩니다.
# 이전에 학습을 진행하던 모델을 이어서 학습을 진행하시려면 다음과 같이 수정한 후 실행해주세요.
# - 아래 셀에서 2 ~ 3번째 줄의 코드를 주석을 해제합니다.
# - 3번째 줄의 경로를 이어서 학습을 진행할 모델의 경로로 변경합니다.  
# (예시: `/content/drive/My Drive/Colab Notebooks/data/hifigan-v2/hifigan-v2-May-31-2021_08+26AM-d897f2e`)
# - 4번째 줄 아래의 코드를 제거합니다.

# 주의: 경로는 실제 환경에 맞게 수정이 필요합니다.
# 이어서 학습을 진행하는 경우:
# os.chdir('content')
# os.system('python TTS/bin/train_vocoder_gan.py --continue_path "/content/drive/My Drive/Colab Notebooks/data/hifigan-v2/hifigan-v2-June-26-2023_07+23AM-3aa165ae"')

# 처음부터 학습을 시작하는 경우:
# os.chdir('content')
# os.system('python TTS/bin/train_vocoder_gan.py '
#           '--config_path "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/config.json" '
#           '--coqpit.data_path "../../content/TTS/filelists/wavs" '
#           '--coqpit.audio.stats_path "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/scale_stats_new.npy" '
#           '--coqpit.output_path "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/" '
#           '--coqpit.num_loader_workers 2 '
#           '--coqpit.num_val_loader_workers 2 '
#           '--restore_path "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/model_file.pth.tar"')

if __name__ == "__main__":
    print("HiFi-GAN 학습 스크립트")
    print("실제 학습을 진행하려면 위의 주석 처리된 코드를 활성화하고 경로를 수정하세요.")

