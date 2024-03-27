---
description: Model registry to manage the model lifecycle from training to production
slug: /guides/model_registry
displayed_sidebar: default
---

# Model Registry
W&B Model Registry는 ML 실무자들이 프로덕션을 위한 후보 모델을 공개하고 하류 팀 및 이해 관계자가 사용할 수 있도록 훈련된 모델을 보관하는 곳입니다. 이 기능은 스테이징된/후보 모델을 보관하고 스테이징과 관련된 워크플로우를 관리하는 데 사용됩니다.

![](/images/models/model_reg_landing_page.png)

W&B Model Registry를 사용하면 다음과 같은 작업을 수행할 수 있습니다:

* [각 머신러닝 작업에 가장 적합한 모델 버전을 북마크에 추가](./link-model-version.md)
* 하류 프로세스 및 모델 CI/CD의 [자동화](./automation.md)
* 스테이징에서 프로덕션까지 모델 버전을 ML 라이프사이클에 따라 이동
* 모델의 계보 추적과 프로덕션 모델의 변경 내역 감사

![](/images/models/models_landing_page.png)

## 작동 방식
몇 가지 간단한 단계로 스테이징된 모델을 추적하고 관리할 수 있습니다.

1. **Log a model version**: 트레이닝 스크립트에서 몇 줄의 코드를 추가하여 모델 파일을 W&B에 아티팩트로 저장합니다.
2. **Compare performance**: 라이브 차트를 통해 모델 트레이닝 및 검증에서 얻은 메트릭과 샘플 예측값을 비교합니다. 어떤 모델 버전이 가장 우수한 성능을 보였는지 확인할 수 있습니다.
3. **Link to registry**: Python에서 프로그래밍 방식으로 또는 W&B UI에서 대화식으로 가장 우수한 모델 버전을 등록된 모델에 연결하여 북마크에 추가합니다.

다음 코드조각은 모델을 Model Registry에 로깅하고 연결하는 방법을 보여줍니다:

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

# 모델을 Model Registry에 로깅하고 연결
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **Connect model transitions to CI/DC workflows**: 후보 모델을 워크플로우 단계를 통해 전환하고 webhook이나 job을 통해 [하류 job을 자동화](./automation.md)합니다.

## 시작 방법
유스케이스에 따라 밑의 자료를 살펴보고 W&B Models를 시작하세요:

* 두 부분으로 구성된 비디오 시리즈를 확인하세요:
  1. [모델 로깅 및 등록](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. Model Registry에서의 [모델 사용 및 하류 프로세스 자동화](https://www.youtube.com/watch?v=8PFCrDSeHzw)
* [Models walkthrough](./walkthrough.md)에서 데이터셋 아티팩트 생성, 추적 및 사용하는 데 쓸 수 있는 W&B Python SDK 커맨드의 단계별 개요를 읽어보세요.
* Model Registry가 ML 워크플로우에 어떻게 적용되는지, 모델 관리에 Model Registry를 사용하면 얻을 수 있는 이점에 대해 설명하는 [이 리포트](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx)를 읽어보세요.
* 다음에 대해 알아보세요:
   * [보호된 모델 및 엑세스 제어](./access_controls.md)
   * [Model Registry를 CI/CD 프로세스에 연결하는 방법](./automation.md)
   * 등록된 모델에 새 모델 버전이 연결될 때 [Slack 알림](./notifications.md) 이 오도록 설정하는 방법