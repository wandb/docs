---
title: Hugging Face
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

![](/images/tutorials/huggingface.png)

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb'/>

[Hugging Face](https://github.com/huggingface/transformers) 모델의 성능을 손쉽게 [W&B](https://wandb.ai/site) 인테그레이션으로 시각화하세요.

모델간 하이퍼파라미터, 출력 메트릭, 및 GPU 활용과 같은 시스템 통계를 비교하세요.

## 왜 W&B를 사용해야 할까요?

![](/images/tutorials/huggingface-why.png)

- **통합 대시보드**: 모든 모델 메트릭과 예측값을 위한 중앙 저장소
- **경량성**: Hugging Face와 인테그레이션 하기 위해 코드 변경이 필요 없음
- **접근성**: 개인 및 학술 연구 팀을 위한 무료
- **보안성**: 모든 프로젝트는 기본적으로 비공개
- **신뢰성**: OpenAI, Toyota, Lyft 등의 기계학습 팀에서 사용

W&B를 기계학습 모델을 위한 GitHub처럼 생각하세요— 기계학습 실험을 개인 호스팅 대시보드에 저장하세요. 스크립트를 어디서 실행하더라도 모든 버전의 모델들이 저장되므로 빠르게 실험할 수 있습니다.

W&B 경량 인테그레이션은 모든 Python 스크립트와 함께 작동하며, 모델을 추적하고 시각화하기 위해 여러분이 해야 할 일은 무료 W&B 계정에 가입하는 것 뿐입니다.

Hugging Face 트랜스포머 레포지토리에서, 우리는 Trainer를 설정하여 매 로그 스텝마다 W&B에 트레이닝 및 평가 메트릭을 자동으로 기록하도록 했습니다.

인테그레이션이 어떻게 작동하는지 자세히 알아보세요: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).

# 🚀 설치, 임포트, 그리고 로그인

Hugging Face와 Weights & Biases 라이브러리, 그리고 GLUE 데이터셋과 트레이닝 스크립트를 이 튜토리얼을 위해 설치하세요.
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 자연어 모델과 데이터셋
- [Weights & Biases](/): 실험 추적과 시각화
- [GLUE dataset](https://gluebenchmark.com/): 언어 이해 벤치마크 데이터셋
- [GLUE script](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py): 시퀀스 분류를 위한 모델 트레이닝 스크립트

```python
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
```

```python
# run_glue.py 스크립트는 트랜스포머 dev가 필요합니다
!pip install -q git+https://github.com/huggingface/transformers
```

## 🖊️ [무료 계정에 가입하기 →](https://app.wandb.ai/login?signup=true)

## 🔑 API 키 입력하기
가입이 완료되면, 다음 셀을 실행하고 링크를 클릭하여 API 키를 받아 이 노트북을 인증하세요.

```python
import wandb
wandb.login()
```

선택적으로, 우리는 환경 변수를 설정하여 W&B 로그를 커스터마이즈할 수 있습니다. [문서](/guides/integrations/huggingface)를 참조하세요.

```python
# 선택적: 그레이디언트와 파라미터 모두 로그
%env WANDB_WATCH=all
```

# 👟 모델 트레이닝
다음으로, 다운로드한 트레이닝 스크립트 [run_glue.py](https://huggingface.co/transformers/examples.html#glue)를 호출하고 Weights & Biases 대시보드에 트레이닝이 자동으로 추적되는 것을 보세요. 이 스크립트는 Microsoft Research Paraphrase Corpus에서 BERT를 파인튜닝 합니다— 두 문장이 의미적으로 동등한지 여부를 사람의 주석으로 판단하는 문장 쌍을 가지고 있습니다.

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
위에 인쇄된 링크를 클릭하거나, [wandb.ai](https://app.wandb.ai)로 직접 가서 실시간 스트림으로 결과를 확인하세요. 모든 종속성이 로드된 후 브라우저에서 당신의 run을 볼 수 있는 링크가 나타납니다 — 다음과 같은 출력 내용을 찾아보세요: "**wandb**: 🚀 View run at [URL to your unique run]"

**모델 성능 시각화**
수십 개의 실험을 훑어보거나 흥미로운 발견을 확대하여 시각화하기 쉽습니다. 고차원 데이터를 시각화하세요.

![](/images/tutorials/huggingface-visualize.gif)

**아키텍처 비교**
[BERT 대 DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) 비교 예제입니다 — 서로 다른 아키텍처가 트레이닝 동안 평가 정확성에 어떻게 영향을 미치는지, 자동 선형 플롯 시각화를 통해 쉽게 볼 수 있습니다.

![](/images/tutorials/huggingface-comparearchitectures.gif)

### 📈 기본적으로 핵심 정보 자동 추적
Weights & Biases는 각 실험에 대해 새로운 run을 저장합니다. 기본적으로 저장되는 정보는 다음과 같습니다:
- **하이퍼파라미터**: 모델의 설정은 Config에 저장됩니다
- **모델 메트릭**: 실시간 스트리밍 메트릭의 시계열 데이터는 Log에 저장됩니다
- **터미널 로그**: 커맨드라인 출력은 저장되고 탭에서 이용 가능합니다
- **시스템 메트릭**: GPU와 CPU 활용도, 메모리, 온도 등

## 🤓 더 많은 정보!
- [문서](/guides/integrations/huggingface): Weights & Biases와 Hugging Face 인테그레이션에 대한 문서
- [비디오](http://wandb.me/youtube): 튜토리얼, 실무자 인터뷰, 그리고 우리의 YouTube 채널에서 더 많은 것들
- 연락처: 궁금한 점이 있다면 contact@wandb.com 으로 메시지를 보내세요.