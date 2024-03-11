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
  <title>기계학습 및 딥러닝 실험 추적하기.</title>
</head>

몇 줄의 코드로 기계학습 실험을 추적하세요. 그런 다음 [인터랙티브 대시보드](app.md)에서 결과를 검토하거나 [Public API](../../ref/python/public-api/README.md)를 사용하여 데이터를 Python으로 프로그래매틱하게 엑세스할 수 있습니다.

[PyTorch](../integrations/pytorch.md), [Keras](../integrations/keras.md), [Scikit](../integrations/scikit.md)과 같은 인기 프레임워크를 사용하는 경우 W&B 인테그레이션을 활용하세요. 모든 인테그레이션의 전체 목록과 코드에 W&B를 추가하는 방법에 대한 정보는 [인테그레이션 가이드](../integrations/intro.md)를 참조하세요.

![](/images/experiments/experiments_landing_page.png)

위의 이미지는 여러 [run](../runs/intro.md)에서 메트릭을 보고 비교할 수 있는 예시 대시보드를 보여줍니다.

## 작동 방식

몇 줄의 코드로 기계학습 실험을 추적하는 방법:
1. [W&B run](../runs/intro.md)을 생성합니다.
2. 학습률이나 모델 유형과 같은 하이퍼파라미터의 사전을 설정에 저장합니다([`wandb.config`](./config.md)).
3. 정확도 및 손실과 같은 훈련 루프에서 시간에 따라 메트릭을 로그합니다([`wandb.log()`](./log/intro.md)).
4. 모델 가중치 또는 예측값 테이블 같은 run의 출력물을 저장합니다.

다음 의사코드는 일반적인 W&B 실험 추적 워크플로우를 보여줍니다:

```python showLineNumbers
# 1. W&B Run 시작
wandb.init(entity="", project="my-project-name")

# 2. 모드 입력 및 하이퍼파라미터 저장
wandb.config.learning_rate = 0.01

# 모델 및 데이터 가져오기
model, dataloader = get_model(), get_data()

# 여기에 모델 트레이닝 코드가 위치합니다.

# 3. 성능을 시각화하기 위해 시간에 따라 메트릭을 로그
wandb.log({"loss": loss})

# 4. W&B에 아티팩트를 로그
wandb.log_artifact(model)
```

## 시작 방법

W&B 실험을 시작하기 위해 사용 사례에 따라 다음 자료를 탐색하세요:

* W&B 아티팩트를 처음 사용하는 경우, [Experiments Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb)을 통해 시작하는 것이 좋습니다.
* 데이터셋 아티팩트를 생성, 추적 및 사용하기 위해 사용할 수 있는 W&B Python SDK 코맨드의 단계별 개요를 알아보려면 [W&B 퀵스타트](../../quickstart.md)를 읽어보세요.
* 다음 챕터를 탐색하여 실험을 어떻게:
  * 생성하는지
  * 설정하는지
  * 실험에서 데이터를 로깅하는 방법
  * 실험 결과를 보는 방법
* [W&B API 참조 가이드](../../ref/README.md) 내의 [W&B Python 라이브러리](../../ref/python/README.md)를 탐색하세요.