---
title: Hugging Face
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-huggingface
    parent: integration-tutorials
weight: 3
---

{{< img src="/images/tutorials/huggingface.png" alt="Hugging Face와 W&B 인테그레이션" >}}

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb" >}}
[Hugging Face](https://github.com/huggingface/transformers) 모델의 성능을 [W&B](https://wandb.ai/site)와의 자연스러운 연동으로 빠르게 시각화하세요.

모델별 하이퍼파라미터, 출력 메트릭, GPU 사용량과 같은 시스템 통계까지 한 번에 비교할 수 있습니다.

## 왜 W&B를 사용해야 하나요?
{.skipvale}

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B 사용 시 이점" >}}

- **통합 대시보드**: 모든 모델 메트릭과 예측값을 위한 중앙 저장소
- **가벼운 인테그레이션**: Hugging Face와 연동 시 코드 수정이 필요 없음
- **누구나 접근 가능**: 개인 및 학술팀은 무료
- **보안**: 모든 프로젝트는 기본적으로 비공개로 설정
- **신뢰성**: OpenAI, Toyota, Lyft 등의 기계학습 팀에서 사용 중

W&B는 기계학습 모델을 위한 GitHub 같은 존재입니다. 기계학습 experiment(실험)들을 개인 대시보드에 안전하게 저장하세요. 어떤 환경에서 스크립트를 실행하든 버전 관리까지 자동으로 처리해주니 빠르게 experiment하면서도 모든 버전이 잘 남습니다.

W&B의 가벼운 인테그레이션은 어떤 Python 스크립트와도 쉽게 연동되며, 무료 W&B 계정만 가입하면 바로 모델 추적 및 시각화를 시작할 수 있습니다.

Hugging Face Transformers repo에서는 Trainer가 로그마다 자동으로 W&B에 트레이닝 및 평가 메트릭을 기록하도록 세팅해 두었습니다.

인테그레이션의 자세한 동작 방식이 궁금하다면 이 리포트를 참고하세요: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).

## 설치, import, 그리고 로그인



Hugging Face와 W&B 라이브러리, 그리고 이 튜토리얼에 필요한 GLUE 데이터셋과 트레이닝 스크립트를 설치합니다.
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 자연어 처리 모델들과 데이터셋
- [W&B]({{< relref path="/" lang="ko" >}}): experiment 추적 및 시각화
- [GLUE 데이터셋](https://gluebenchmark.com/): 언어이해 벤치마크 데이터셋
- [GLUE 스크립트](https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py): 시퀀스 분류용 모델 트레이닝 스크립트


```notebook
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py
```


```notebook
# run_glue.py 스크립트는 transformers 개발 버전이 필요합니다
!pip install -q git+https://github.com/huggingface/transformers
```

다음 단계로 넘어가기 전에 [무료 계정에 가입](https://app.wandb.ai/login?signup=true)하세요.

## API 키 입력하기

가입을 완료했다면, 아래 셀을 실행 후 안내되는 링크에서 API 키를 받아 노트북 인증을 해주세요.


```python
import wandb
wandb.login()
```

추가적으로, 환경변수를 설정해서 W&B 로그 옵션을 커스터마이즈할 수도 있습니다. 자세한 내용은 [Hugging Face 인테그레이션 가이드]({{< relref path="/guides/integrations/huggingface/" lang="ko" >}})를 참고하세요.


```python
# 선택 사항: gradient와 parameter 모두를 기록
%env WANDB_WATCH=all
```

## 모델 트레이닝하기
이제 다운로드한 트레이닝 스크립트 [run_glue.py](https://huggingface.co/transformers/examples.html#glue)를 실행해 보세요. 트레이닝이 자동으로 W&B 대시보드에 기록됩니다. 이 스크립트는 BERT를 Microsoft Research Paraphrase Corpus에 대해 파인튜닝하며, 문장 쌍이 의미론적으로 동등한지 사람의 어노테이션이 포함되어 있습니다.


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

##  대시보드에서 결과 시각화하기
위에서 출력된 링크를 클릭하거나 [wandb.ai](https://app.wandb.ai)로 접속해 내 결과가 실시간으로 올라오는 모습을 확인하세요. 모든 의존성이 로드된 뒤, 브라우저에서 내 run을 볼 수 있는 링크가 나타납니다. 다음과 같은 출력 메시지를 확인하세요: "**wandb**: View run at [URL to your unique run]"

**모델 성능 시각화**
수십 개 experiment를 한눈에 비교하고, 흥미로운 발견한 내용을 심층 분석하며, 고차원 데이터도 쉽게 시각화할 수 있습니다.

{{< img src="/images/tutorials/huggingface-visualize.gif" alt="모델 메트릭 대시보드" >}}

**아키텍처 비교**
[BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) 예시처럼, 서로 다른 아키텍처가 트레이닝 과정에서 평가 정확도에 어떤 영향을 주는지 자동 시각화된 라인 플롯으로 바로 볼 수 있습니다.

{{< img src="/images/tutorials/huggingface-comparearchitectures.gif" alt="BERT vs DistilBERT 비교" >}}

## 기본적으로 주요 정보는 자동 기록
W&B는 experiment별로 새로운 run을 저장합니다. 기본적으로 저장되는 항목은 아래와 같습니다:
- **하이퍼파라미터**: 모델 설정이 Config에 저장됨
- **모델 메트릭**: 메트릭 시계열 데이터가 Log에 저장됨
- **터미널 로그**: 커맨드라인 출력이 별도 탭에 저장되어 언제든 확인 가능
- **시스템 메트릭**: GPU, CPU 사용률, 메모리, 온도 등 시스템 상태 정보

## 더 알아보기
- [Hugging Face 인테그레이션 가이드]({{< relref path="/guides/integrations/huggingface" lang="ko" >}})
- [YouTube 동영상 튜토리얼](http://wandb.me/youtube)
