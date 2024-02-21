---
description: Model registry to manage the model lifecycle from training to production
slug: /guides/model_registry
displayed_sidebar: default
---

# 모델 레지스트리
W&B 모델 레지스트리는 ML 실무자들이 하류 팀 및 이해 관계자가 사용할 수 있도록 프로덕션 후보 모델을 게시할 수 있는 팀의 훈련된 모델을 보관하는 곳입니다. 이는 스테이징된/후보 모델을 보관하고 스테이징과 관련된 워크플로를 관리하는 데 사용됩니다.

![](/images/models/model_reg_landing_page.png)

W&B 모델 레지스트리를 사용하면 다음을 할 수 있습니다:

* [각 머신 러닝 작업에 대한 최고의 모델 버전을 북마크합니다.](./link-model-version.md)
* [자동화](./automation.md) 하류 프로세스 및 모델 CI/CD.
* 모델 버전을 ML 생명주기를 통해 이동시키기; 스테이징에서 프로덕션까지.
* 모델의 계보를 추적하고 프로덕션 모델의 변경 이력을 감사합니다.

![](/images/models/models_landing_page.png)

## 작동 방식
몇 가지 간단한 단계로 스테이징된 모델을 추적하고 관리하세요.

1. **모델 버전 로그하기**: 학습 스크립트에서 코드 몇 줄을 추가하여 모델 파일을 W&B에 아티팩트로 저장합니다.
2. **성능 비교**: 실시간 차트를 확인하여 모델 학습 및 검증에서의 메트릭과 샘플 예측값을 비교합니다. 가장 성능이 좋은 모델 버전을 식별합니다.
3. **레지스트리에 연결**: Python에서 프로그래밍 방식으로 또는 W&B UI에서 대화형으로 가장 좋은 모델 버전을 등록된 모델에 연결하여 북마크합니다.

다음 코드 조각은 모델을 모델 레지스트리에 로그하고 연결하는 방법을 보여줍니다:

```python showLineNumbers
import wandb
import random

# 새 W&B 실행 시작
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

4. **모델 전환을 CI/DC 워크플로에 연결**: 워크플로 단계를 통해 후보 모델 전환하고 웹훅이나 작업을 사용하여 [하류 작업 자동화](./automation.md).

## 시작 방법
사용 사례에 따라 W&B 모델을 시작하기 위해 다음 자료를 탐색하세요:

* 두 부분으로 구성된 비디오 시리즈를 확인하세요:
  1. [모델 로깅 및 등록](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. [모델 소비 및 하류 프로세스 자동화](https://www.youtube.com/watch?v=8PFCrDSeHzw) in the Model Registry.
* W&B Python SDK 명령을 사용하여 데이터세트 아티팩트를 생성, 추적 및 사용하는 단계별 개요를 위해 [모델 워크스루](./walkthrough.md)를 읽어보세요.
* 모델 관리에 대한 모델 레지스트리의 적합성과 이점에 대해 [이](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) 리포트를 검토하세요.
* 다음에 대해 알아보세요:
   * [보호된 모델과 엑세스 제어](./access_controls.md).
   * [모델 레지스트리를 CI/CD 프로세스에 연결하는 방법](./automation.md).
   * 등록된 모델에 새 모델 버전이 연결될 때 [Slack 알림](./notifications.md) 설정하기.