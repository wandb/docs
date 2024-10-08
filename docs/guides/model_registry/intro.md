---
title: Model registry
description: 트레이닝부터 프로덕션까지 모델 생명 주기를 관리하기 위한 모델 레지스트리
slug: /guides/model_registry
displayed_sidebar: default
---

:::안내
W&B는 2024년 이후로 W&B Model Registry에 대한 지원을 중단할 예정입니다. 사용자는 모델 아티팩트 버전을 연결하고 공유하기 위해 [W&B Registry](../registry/intro.md)를 대신 사용할 것을 권장합니다. W&B Registry는 기존 W&B Model Registry의 기능을 확장합니다. W&B Registry에 대한 자세한 내용은 [Registry docs](../registry/intro.md)를 참조하세요.

W&B는 기존 Model Registry에 연결된 모델 아티팩트를 2024년 가을이나 초 겨울에 새로운 W&B Registry로 이전할 예정입니다. 이전 프로세스에 대한 자세한 정보는 [Migrating from legacy Model Registry](../registry/model_registry_eol.md)를 참조하세요.
:::

W&B Model Registry는 팀의 트레이닝된 모델을 보관하는 곳으로, ML 실무자들이 프로덕션으로서의 후보를 게시하여 다운스트림 팀 및 이해관계자들이 사용할 수 있도록 합니다. 이는 스테이징/후보 모델을 보관하고 스테이징과 관련된 워크플로우를 관리하는 데 사용됩니다.

![](/images/models/model_reg_landing_page.png)

W&B Model Registry를 사용하면 다음을 수행할 수 있습니다:

* [각 기계학습 작업에 대해 최상의 모델 버전을 북마크합니다.](./link-model-version.md)
* 다운스트림 프로세스와 모델 CI/CD를 [자동화합니다](./model-registry-automations.md).
* 모델 버전을 ML 수명 주기 내에서 이동시킵니다; 스테이징에서 프로덕션으로.
* 모델의 계보를 추적하고 프로덕션 모델에 대한 변경 이력을 감사합니다.

![](/images/models/models_landing_page.png)

## 작동 방식
단계별 몇 가지 간단한 단계로 스테이징된 모델을 추적하고 관리합니다.

1. **모델 버전 로그하기**: 트레이닝 스크립트에서 몇 줄의 코드를 추가하여 모델 파일을 W&B에 아티팩트로 저장합니다.
2. **성능 비교**: 라이브 차트를 확인하여 모델 트레이닝과 검증에서 메트릭과 샘플 예측값을 비교합니다. 어떤 모델 버전이 가장 잘 수행했는지 식별합니다.
3. **레지스트리에 연결하기**: 등록된 모델에 가장 좋은 모델 버전을 프로그램으로 Python에서 또는 W&B UI에서 인터랙티브하게 연결하여 북마크합니다.

다음 코드조각은 Model Registry에 모델을 로그하고 연결하는 방법을 보여줍니다:

```python showLineNumbers
import wandb
import random

# 새로운 W&B run 시작
run = wandb.init(project="models_quickstart")

# 모델 메트릭 로그 시뮬레이션
run.log({"acc": random.random()})

# 시뮬레이션된 모델 파일 생성
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# 모델을 Model Registry에 로그하고 연결
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **모델 전환을 CI/DC 워크플로우와 연결하기**: 웹훅 또는 작업으로 [다운스트림 작업을 자동화](./model-registry-automations.md)하여 후보 모델을 워크플로우 단계로 전환합니다.

## 시작 방법
유스 케이스에 따라, W&B Models를 시작하는 데 필요한 다음 리소스를 탐색하세요:

* 두 파트의 비디오 시리즈를 확인하세요:
  1. [모델 로그 및 등록](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. Model Registry에서 모델 소비와 다운스트림 프로세스 자동화 [모델 소비 및 다운스트림 프로세스 자동화](https://www.youtube.com/watch?v=8PFCrDSeHzw).
* [models walkthrough](./walkthrough.md)를 읽어 W&B Python SDK 명령을 사용하여 데이터셋 아티팩트를 생성, 추적 및 사용하는 방법을 단계별로 알아보세요.
* 다음에 대해 배우세요:
   * [보호된 모델 및 엑세스 제어](./access_controls.md).
   * [CI/CD 프로세스에 Model Registry 연결하기](./model-registry-automations.md).
   * 새로운 모델 버전이 등록된 모델에 연결될 때 [Slack 알림](./notifications.md) 설정하기.
* Model Registry가 ML 워크플로우에 적합한 방법과 모델 관리에 사용하는 이점을 이해하는 [리포트](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx)를 검토하세요.
* W&B [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) 코스를 수강하여 다음을 배우세요:
  * W&B Model Registry를 사용하여 모델을 관리하고 버전화하여, 계보를 추적하고 다양한 수명 주기 단계를 통해 모델을 승격하는 방법
  * 웹훅과 작업 실행을 사용하여 모델 관리 워크플로우를 자동화하는 방법
  * 모델 평가, 모니터링 및 배포를 위한 모델 개발 수명 주기 내의 외부 기계학습 시스템 및 툴과 Model Registry가 통합되는 방법.