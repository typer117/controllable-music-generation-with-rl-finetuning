# 강화학습 기반 파인튜닝을 활용한 음악 생성 모델의 제어가능성 개선

## 환경 설정 및 데이터셋

### Setup

아래 명령어를 통해 가상환경 생성 및 필수 패키지 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Preparing the Data

1. 사용 데이터셋 설치: [Lakh MIDI](https://colinraffel.com/projects/lmd/) 

```bash
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz
```
2. 압축 파일 제거: `rm lmd_full.tar.gz`


### Download Pre-Trained Models

1. 사전학습 모델 설치: [FIGARO](https://github.com/dvruette/figaro)

```bash
wget -O checkpoints.zip https://polybox.ethz.ch/index.php/s/a0HUHzKuPPefWkW/download
unzip checkpoints.zip
```
2. 압축 파일 제거: `rm checkpoints.zip`



## Training


```bash
python src/rl_train.py --checkpoint {path_to_figaro-expert.ckpt} --checkpoint_save {path_to_save_fine_tuned_model} --lmd_dir {path_to_lmd_dataset}
```

## Evaluation

```bash
python src/evaluate.py --checkpoint {path_to_figaro-expert.ckpt} --checkpoint_save {path_to_model.pt_for_evaluation} --lmd_dir {path_to_lmd_dataset}
```


## Parameters

### Training (`rl_train.py`)

    --checkpoint: 사전학습 모델 위치. 다운받은 figaro-expert.ckpt로의 경로
    --checkpoint_save: 파인튜닝 모델을 저장하고자 하는 경로
    --lmd_dir: Lakh MIDI 데이터셋 경로
    --max_bars: 생성 마디 길이
    --batch_size_ep: 병렬로 생성되는 음악 수
    --batch_size_opt: 모델 파라미터 업데이트에 사용할 batch_size
    --lr: 음악 생성 모델 파인튜닝 learning rate
    --value_lr: 상태가치함수 학습시 learning rate
    --temp: 음악 생성 모델의 temperature
    --p_maintainer: nucleus 일관성 보상함수 사용시 p값
    --consistency: 일관성 보상함수 종류, 'N'(nucleus) 또는 'L'(likelihood)
    --baseline: REINFORCE 알고리즘의 베이스라인 종류, 'M'(보상 가중평균) 또는 'V'(상태가치함수)
    --exploration: 탐험 전략 종류, 'E'(e-greedy) 또는 'T'(temperature annealing)


### Evaluation (`evaluate.py`)
    --checkpoint_save: 평가할 모델의 경로
    --lmd_dir: Lakh MIDI 데이터셋 경로
    --test_size: 테스트셋 사이즈
    --max_bars: 생성 마디 길이
    --temp: 음악 생성 모델의 temperature

본 repository는 사전학습 모델인 [FIGARO](https://github.com/dvruette/figaro)의  데이터 선처리, 모델 정의 등의 코드를 포함하고 있습니다. 