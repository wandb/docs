---
description: Model registry to manage the model lifecycle from training to production
slug: /guides/model_registry
displayed_sidebar: default
---

# 모델 레지스트리
W&B 모델 레지스트리는 ML 실무자들이 생산을 위한 후보 모델을 공개하고 하류 팀 및 이해 관계자가 사용할 수 있도록 훈련된 모델을 보관하는 곳입니다. 이는 스테이징된/후보 모델을 보관하고 스테이징과 관련된 워크플로우를 관리하는 데 사용됩니다.

![](/images/models/model_reg_landing_page.png)

W&B 모델 레지스트리를 사용하면 다음을 할 수 있습니다:

* [각 기계학습 작업에 대해 최고의 모델 버전을 북마크하세요.](./link-model-version.md)
* [자동화](./automation.md) 하류 프로세스 및 모델 CI/CD.
* 모델 버전을 ML 라이프사이클을 통해 이동; 스테이징에서 프로덕션까지.
* 모델의 계보를 추적하고 프로덕션 모델의 변경 내역을 감사합니다.

![](/images/models/models_landing_page.png)

## 작동 방식
몇 가지 간단한 단계로 스테이징된 모델을 추적하고 관리합니다.

1. **모델 버전 로그**: 트레이닝 스크립트에서 몇 줄의 코드를 추가하여 모델 파일을 W&B에 아티팩트로 저장합니다.
2. **성능 비교**: 실시간 차트를 확인하여 모델 트레이닝 및 검증에서 메트릭과 샘플 예측값을 비교합니다. 어떤 모델 버전이 가장 잘 수행되었는지 확인합니다.
3. **레지스트리에 연결**: 프로그래밍 방식으로 Python에서 또는 W&B UI에서 대화식으로 최고의 모델 버전을 등록된 모델에 연결하여 북마크합니다.

다음 코드조각은 모델을 모델 레지스트리에 로그하고 연결하는 방법을 보여줍니다:

```python showLineNumbers
import wandb
import random

# 새로운 W&B run 시작
run = wandb.init(project="models_quickstart")

# 모델 메트릭 로깅 시뮬레이션
run.log({"acc": random.random()})

# 시뮬레이션된 모델 파일 생성
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# 모델을 모델 레지스트리에 로그하고 연결
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **모델 전환을 CI/DC 워크플로우에 연결**: 후보 모델을 워크플로우 단계를 통해 전환하고 웹훅이나 작업을 통해 [하류 작업을 자동화](./automation.md)합니다.

## 시작 방법
사용 사례에 따라 W&B 모델을 시작하는 데 다음 리소스를 탐색하세요:

* 두 부분으로 구성된 비디오 시리즈를 확인하세요:
  1. [모델 로깅 및 등록](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. [모델 사용 및 하류 프로세스 자동화](https://www.youtube.com/watch?v=8PFCrDSeHzw) 모델 레지스트리에서.
* W&B Python SDK 명령어를 사용하여 데이터셋 아티팩트를 생성, 추적 및 사용하는 단계별 개요인 [models walkthrough](./walkthrough.md)를 읽어보세요.
* 모델 레지스트리가 ML 워크플로우에 어떻게 맞는지 및 모델 관리를 위해 하나를 사용하는 것의 이점에 대한 [이 리포트](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx)를 검토하세요.
* 다음에 대해 알아보세요:
   * [보호된 모델 및 엑세스 제어](./access_controls.md).
   * [모델 레지스트리를 CI/CD 프로세스에 연결하는 방법](./automation.md).
   * 등록된 모델에 새 모델 버전이 연결될 때 [Slack 알림](./notifications.md) 설정하기.