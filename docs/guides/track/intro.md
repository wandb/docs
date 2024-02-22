---
description: Track machine learning experiments with W&B.
slug: /guides/track
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 실험 추적하기

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb"/>

<head>
  <title>머신 러닝 및 딥 러닝 실험 추적.</title>
</head>

몇 줄의 코드로 머신 러닝 실험을 추적하세요. 그런 다음 [인터랙티브 대시보드](app.md)에서 결과를 검토하거나 [Public API](../../ref/python/public-api/README.md)를 사용하여 데이터를 Python으로 프로그래밍 방식으로 엑세스할 수 있습니다.

[PyTorch](../integrations/pytorch.md), [Keras](../integrations/keras.md), 또는 [Scikit](../integrations/scikit.md)과 같은 인기 있는 프레임워크를 사용하는 경우 W&B 통합을 활용하세요. 통합의 전체 목록과 코드에 W&B를 추가하는 방법에 대한 정보는 [통합 가이드](../integrations/intro.md)를 참조하세요.

![](/images/experiments/experiments_landing_page.png)

위 이미지는 여러 [실행](../runs/intro.md) 간 메트릭을 보고 비교할 수 있는 예제 대시보드를 보여줍니다.

## 작동 방식

몇 줄의 코드로 머신 러닝 실험을 추적하는 방법:
1. [W&B 실행](../runs/intro.md)을 생성합니다.
2. 학습률이나 모델 유형과 같은 하이퍼파라미터의 사전을 구성([`wandb.config`](./config.md))에 저장합니다.
3. 학습 루프에서 정확도와 손실과 같은 메트릭([`wandb.log()`](./log/intro.md))을 시간에 따라 기록합니다.
4. 실행의 출력물, 예를 들어 모델 가중치나 예측값 테이블을 저장합니다.

다음 의사 코드는 일반적인 W&B 실험 추적 워크플로를 보여줍니다:

```python showLineNumbers
# 1. W&B 실행 시작
wandb.init(entity="", project="my-project-name")

# 2. 모델 입력 및 하이퍼파라미터 저장
wandb.config.learning_rate = 0.01

# 모델과 데이터 가져오기
model, dataloader = get_model(), get_data()

# 여기에 모델 학습 코드 작성

# 3. 시간에 따라 성능을 시각화하기 위해 메트릭 기록
wandb.log({"loss": loss})

# 4. W&B에 아티팩트 기록
wandb.log_artifact(model)
```

## 시작 방법

사용 사례에 따라 W&B 실험을 시작하는 데 도움이 되는 다음 자료를 탐색하세요:

* W&B 아티팩트를 처음 사용하는 경우, [Experiments Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb)을 통해 진행하는 것이 좋습니다.
* [W&B 퀵스타트](../../quickstart.md)를 읽어 데이터세트 아티팩트를 생성, 추적 및 사용하기 위해 사용할 수 있는 W&B Python SDK 명령의 단계별 개요를 확인하세요.
* 다음을 배우기 위해 이 장을 탐색하세요:
  * 실험 생성하기
  * 실험 구성하기
  * 실험으로부터 데이터 로그하기
  * 실험 결과 보기
* [W&B API 참조 가이드](../../ref/README.md) 내의 [W&B Python 라이브러리](../../ref/python/README.md)를 탐색하세요.