---
title: Track experiments
description: W&B을 사용하여 기계학습 실험을 추적하세요.
slug: /guides/track
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb"/>

몇 줄의 코드로 기계학습 Experiments를 추적하세요. 그런 다음 [대시보드](app.md)에서 결과를 검토하거나 [Public API](../../ref/python/public-api/README.md)를 사용하여 프로그래밍 방식으로 Python에 데이터를 내보낼 수 있습니다.

[PyTorch](../integrations/pytorch.md), [Keras](../integrations/keras.md), [Scikit](../integrations/scikit.md)와 같은 인기 있는 프레임워크를 사용한다면 W&B Integrations을 활용하세요. 인테그레이션을 추가하는 방법과 전체 인테그레이션 목록은 [Integration guides](../integrations/intro.md)을 참조하세요.

![](/images/experiments/experiments_landing_page.png)

위 이미지는 여러 [runs](../runs/intro.md)에서 메트릭을 보고 비교할 수 있는 대시보드 예시를 보여줍니다.

## 작동 방식

몇 줄의 코드로 기계학습 experiment를 추적하세요:
1. [W&B run](../runs/intro.md)을 생성합니다.
2. 학습률이나 모델 유형과 같은 하이퍼파라미터를 설정에 사전 형태로 저장합니다 ([`wandb.config`](./config.md)).
3. 정확도와 손실과 같은 메트릭을 트레이닝 루프에서 시간에 따라 로그합니다 ([`wandb.log()`](./log/intro.md)).
4. 모델 가중치나 예측값 테이블과 같은 run의 출력을 저장합니다.

다음의 의사 코드는 일반적인 W&B Experiment 추적 워크플로우를 설명합니다:

```python showLineNumbers
# 1. W&B Run 시작
wandb.init(entity="", project="my-project-name")

# 2. 모드 입력과 하이퍼파라미터 저장
wandb.config.learning_rate = 0.01

# 모델 및 데이터 가져오기
model, dataloader = get_model(), get_data()

# 모델 트레이닝 코드는 여기에 작성됩니다

# 3. 메트릭을 시간에 따라 로그하여 성능 시각화
wandb.log({"loss": loss})

# 4. W&B에 아티팩트 로그
wandb.log_artifact(model)
```

## 시작 방법

유스 케이스에 따라 W&B Experiments 시작에 필요한 다음 리소스를 탐색하십시오:

* [W&B 퀵스타트](../../quickstart.md)를 읽어 W&B Python SDK 명령어를 사용하여 데이터셋 아티팩트를 생성, 추적, 활용하는 단계별 개요를 확인하세요.
* 이 챕터를 탐구하여 다음을 배우세요:
  * Experiment 생성
  * Experiments 설정
  * Experiments에서 데이터 로그
  * Experiment 결과 보기
* [W&B API Reference Guide](../../ref/README.md) 내의 [W&B Python 라이브러리](../../ref/python/README.md)를 탐색하세요.