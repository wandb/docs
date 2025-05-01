---
title: Model registry
description: 트레이닝부터 프로덕션까지 모델 생명주기를 관리하는 모델 레지스트리
cascade:
- url: /ko/guides//core/registry/model_registry/:filename
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-_index
    parent: registry
url: /ko/guides//core/registry/model_registry
weight: 9
---

{{% alert %}}
W&B는 결국 W&B Model Registry에 대한 지원을 중단할 예정입니다. 사용자는 대신 모델 아티팩트 버전을 연결하고 공유하기 위해 [W&B Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})를 사용하는 것이 좋습니다. W&B Registry는 기존 W&B Model Registry의 기능을 확장합니다. W&B Registry에 대한 자세한 내용은 [Registry 문서]({{< relref path="/guides/core/registry/" lang="ko" >}})를 참조하세요.

W&B는 기존 Model Registry에 연결된 기존 모델 아티팩트를 가까운 시일 내에 새로운 W&B Registry로 마이그레이션할 예정입니다. 마이그레이션 프로세스에 대한 자세한 내용은 [기존 Model Registry에서 마이그레이션]({{< relref path="/guides/core/registry/model_registry_eol.md" lang="ko" >}})을 참조하세요.
{{% /alert %}}

W&B Model Registry는 팀의 트레이닝된 모델을 보관하는 곳으로, ML 전문가가 프로덕션 후보를 게시하여 다운스트림 팀과 이해 관계자가 사용할 수 있습니다. 스테이징된/후보 모델을 보관하고 스테이징과 관련된 워크플로우를 관리하는 데 사용됩니다.

{{< img src="/images/models/model_reg_landing_page.png" alt="" >}}

W&B Model Registry를 사용하면 다음을 수행할 수 있습니다.

* [각 기계 학습 작업에 대해 가장 적합한 모델 버전을 북마크합니다.]({{< relref path="./link-model-version.md" lang="ko" >}})
* 다운스트림 프로세스 및 모델 CI/CD를 [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})합니다.
* 모델 버전을 ML 라이프사이클(스테이징에서 프로덕션)을 거쳐 이동합니다.
* 모델의 계보를 추적하고 프로덕션 모델에 대한 변경 이력을 감사합니다.

{{< img src="/images/models/models_landing_page.png" alt="" >}}

## 작동 방식
몇 가지 간단한 단계를 통해 스테이징된 모델을 추적하고 관리합니다.

1. **모델 버전 로깅**: 트레이닝 스크립트에서 몇 줄의 코드를 추가하여 모델 파일을 아티팩트 로 W&B에 저장합니다.
2. **성능 비교**: 라이브 차트를 확인하여 모델 트레이닝 및 유효성 검사에서 메트릭 과 샘플 예측값을 비교합니다. 어떤 모델 버전이 가장 성능이 좋았는지 식별합니다.
3. **레지스트리에 연결**: Python에서 프로그래밍 방식으로 또는 W&B UI에서 대화식으로 등록된 모델에 연결하여 최상의 모델 버전을 북마크합니다.

다음 코드 조각은 모델을 Model Registry에 로깅하고 연결하는 방법을 보여줍니다.

```python
import wandb
import random

# Start a new W&B run
run = wandb.init(project="models_quickstart")

# Simulate logging model metrics
run.log({"acc": random.random()})

# Create a simulated model file
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# Log and link the model to the Model Registry
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **모델 전환을 CI/CD 워크플로우에 연결**: 웹훅을 사용하여 워크플로우 단계를 통해 후보 모델을 전환하고 [다운스트림 작업 자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})합니다.

## 시작 방법
유스 케이스에 따라 다음 리소스를 탐색하여 W&B Models를 시작하십시오.

* 2부작 비디오 시리즈를 확인하세요.
  1. [모델 로깅 및 등록](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. Model Registry에서 [모델 사용 및 다운스트림 프로세스 자동화](https://www.youtube.com/watch?v=8PFCrDSeHzw).
* W&B Python SDK 코맨드에 대한 단계별 개요는 [모델 둘러보기]({{< relref path="./walkthrough.md" lang="ko" >}})를 읽어보세요. 이를 통해 데이터셋 아티팩트 를 생성, 추적 및 사용할 수 있습니다.
* 다음에 대해 알아보세요.
   * [보호된 모델 및 엑세스 제어]({{< relref path="./access_controls.md" lang="ko" >}}).
   * [레지스트리를 CI/CD 프로세스에 연결하는 방법]({{< relref path="/guides/core/automations/" lang="ko" >}}).
   * 새 모델 버전이 등록된 모델에 연결되면 [Slack 알림]({{< relref path="./notifications.md" lang="ko" >}})을 설정합니다.
* Model Registry가 ML 워크플로우에 어떻게 적합하고 모델 관리에 사용하는 이점에 대한 [이] (https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) 리포트 를 검토합니다.
* W&B [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) 코스를 수강하고 다음 방법을 배우세요.
  * W&B Model Registry를 사용하여 모델을 관리 및 버전 관리하고, 계보를 추적하고, 다양한 라이프사이클 단계를 거쳐 모델을 승격합니다.
  * 웹훅을 사용하여 모델 관리 워크플로우를 자동화합니다.
  * 모델 평가, 모니터링 및 배포를 위해 Model Registry가 모델 개발 라이프사이클의 외부 ML 시스템 및 툴 과 어떻게 통합되는지 확인하세요.
