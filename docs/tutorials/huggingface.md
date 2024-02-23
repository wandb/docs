
# Hugging Face

<img src="https://i.imgur.com/vnejHGh.png" width="800"/>

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb)

[W&B](https://wandb.ai/site) 통합을 통해 [Hugging Face](https://github.com/huggingface/transformers) 모델의 성능을 빠르게 시각화하세요.

모델들 간에 하이퍼파라미터, 출력 메트릭, GPU 사용량과 같은 시스템 통계를 비교하세요.

## 🤔 왜 W&B를 사용해야 하나요?

<img src="https://wandb.me/mini-diagram" width="650"/>

- **통합 대시보드**: 모든 모델 메트릭과 예측값을 위한 중앙 저장소
- **경량화**: Hugging Face와 통합하기 위해 코드 변경 필요 없음
- **접근성**: 개인 및 학술 팀은 무료로 사용 가능
- **보안**: 모든 프로젝트는 기본적으로 비공개
- **신뢰성**: OpenAI, Toyota, Lyft 등의 머신 러닝 팀이 사용

W&B를 머신 러닝 모델을 위한 GitHub처럼 생각하세요 - 머신 러닝 실험을 개인 호스팅 대시보드에 저장합니다. 스크립트를 실행하는 위치에 상관없이 모델의 모든 버전이 저장되어 있으므로 안심하고 빠르게 실험할 수 있습니다.

W&B의 경량 통합은 모든 Python 스크립트와 작동하며, 무료 W&B 계정에 가입하기만 하면 모델을 추적하고 시각화하기 시작할 수 있습니다.

Hugging Face Transformers 저장소에서는 Trainer를 자동으로 W&B에 학습 및 평가 메트릭을 각 로깅 단계에서 로깅하도록 구성했습니다.

통합 작동 방식을 자세히 살펴보세요: [Hugging Face + W&B 리포트](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).

# 🚀 설치, 가져오기 및 로그인

이 튜토리얼에 필요한 Hugging Face 및 Weights & Biases 라이브러리, GLUE 데이터세트 및 학습 스크립트를 설치하세요.
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 자연어 모델과 데이터세트
- [Weights & Biases](https://docs.wandb.com/): 실험 추적 및 시각화
- [GLUE 데이터세트](https://gluebenchmark.com/): 언어 이해 벤치마크 데이터세트
- [GLUE 스크립트](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py): 시퀀스 분류를 위한 모델 학습 스크립트


```python
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
```


```python
# run_glue.py 스크립트는 transformers dev가 필요합니다
!pip install -q git+https://github.com/huggingface/transformers
```

## 🖊️ [무료 계정 등록 →](https://app.wandb.ai/login?signup=true)

## 🔑 API 키 입력
가입했다면 다음 셀을 실행하고 링크를 클릭하여 API 키를 받아 이 노트북을 인증하세요.


```python
import wandb
wandb.login()
```

선택적으로, W&B 로깅을 사용자 지정하기 위해 환경 변수를 설정할 수 있습니다. [문서](https://docs.wandb.com/library/integrations/huggingface)를 참조하세요.


```python
# 선택적: 그레이디언트와 파라미터 모두 로그
%env WANDB_WATCH=all
```

# 👟 모델 학습
다음으로, 다운로드한 학습 스크립트 [run_glue.py](https://huggingface.co/transformers/examples.html#glue)를 호출하고 학습이 Weights & Biases 대시보드에 자동으로 추적되는 것을 확인하세요. 이 스크립트는 Microsoft Research Paraphrase Corpus에서 BERT를 파인 튜닝합니다 — 문장 쌍과 이들이 의미론적으로 동등한지 여부를 나타내는 인간 주석이 포함되어 있습니다.


```python
%env WANDB_PROJECT=huggingface-demo
%env TASK_NAME=MRPC

!python run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --logging_steps 50
```

# 👀 대시보드에서 결과 시각화
위에 인쇄된 링크를 클릭하거나 [wandb.ai](https://app.wandb.ai)로 이동하여 결과가 실시간으로 스트리밍되는 것을 확인하세요. 모든 의존성이 로드된 후 브라우저에서 실행을 보려면 다음 출력을 찾으세요: "**wandb**: 🚀 [URL to your unique run]에서 실행 보기"

**모델 성능 시각화**
실험 수십 개를 쉽게 살펴보고, 흥미로운 발견에 초점을 맞추고, 고차원 데이터를 시각화할 수 있습니다.

![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79YL90K1jiq-3jeQK-%2Fhf%20gif%2015.gif?alt=media&token=523d73f4-3f6c-499c-b7e8-ef5be0c10c2a)

**아키텍처 비교**
다음은 [BERT 대 DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU)를 비교한 예입니다 — 자동 선 그래프 시각화를 통해 학습하는 동안 다양한 아키텍처가 평가 정확도에 어떤 영향을 미치는지 쉽게 볼 수 있습니다.
![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79Ytpj6q6Jlv9RKZGT%2Fgif%20for%20comparing%20bert.gif?alt=media&token=e3dee5de-d120-4330-b4bd-2e2ddbb8315e)

### 📈 기본적으로 핵심 정보를 쉽게 추적
Weights & Biases는 각 실험에 대해 새로운 실행을 저장합니다. 기본적으로 저장되는 정보는 다음과 같습니다:
- **하이퍼파라미터**: 모델의 설정은 Config에 저장됩니다
- **모델 메트릭**: 메트릭의 시계열 데이터는 Log에 저장됩니다
- **터미널 로그**: 명령 줄 출력이 저장되어 탭에서 사용할 수 있습니다
- **시스템 메트릭**: GPU 및 CPU 사용량, 메모리, 온도 등

## 🤓 더 알아보기!
- [문서](https://docs.wandb.com/huggingface): Weights & Biases와 Hugging Face 통합에 대한 문서
- [동영상](http://wandb.me/youtube): YouTube 채널에서 튜토리얼, 실무자 인터뷰 등
- 문의: 질문이 있으시면 contact@wandb.com으로 메시지 보내주세요