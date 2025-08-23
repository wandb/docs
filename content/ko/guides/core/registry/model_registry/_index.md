---
title: 모델 레지스트리
description: 모델 레지스트리를 통해 트레이닝부터 프로덕션까지 모델 라이프사이클을 관리하세요
cascade:
- url: guides/core/registry/model_registry/:filename
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-_index
    parent: registry
url: guides/core/registry/model_registry
weight: 9
---

{{% alert %}}
W&B는 앞으로 W&B Model Registry 지원을 종료할 예정입니다. 사용자는 모델 아티팩트 버전의 연결 및 공유를 위해 [W&B Registry]({{< relref path="/guides/core/registry/" lang="ko" >}}) 사용을 권장합니다. W&B Registry는 기존 W&B Model Registry의 기능을 확장합니다. W&B Registry에 대한 자세한 내용은 [Registry docs]({{< relref path="/guides/core/registry/" lang="ko" >}})를 참고하세요.

W&B는 곧 기존 Model Registry에 연결된 모델 아티팩트들을 새로운 W&B Registry로 이전할 예정입니다. 마이그레이션 과정에 대한 자세한 내용은 [Migrating from legacy Model Registry]({{< relref path="/guides/core/registry/model_registry_eol.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}

W&B Model Registry는 팀의 트레이닝된 모델을 보관하는 곳으로, ML 실무자들이 프로덕션 후보 모델을 게시하여 다른 팀이나 이해관계자들이 활용할 수 있도록 도와줍니다. 또한 staged/candidate 모델을 보관하고, staging과 관련된 워크플로우를 효율적으로 관리할 수 있습니다.

{{< img src="/images/models/model_reg_landing_page.png" alt="Model Registry" >}}

W&B Model Registry를 활용하면 다음과 같은 작업이 가능합니다:

* [각 기계학습 태스크별로 최고의 모델 버전을 북마크할 수 있습니다.]({{< relref path="./link-model-version.md" lang="ko" >}})
* [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})를 통해 다운스트림 프로세스와 모델 CI/CD를 연동할 수 있습니다.
* 모델 버전을 staging에서 프로덕션까지 ML 라이프사이클에 따라 관리할 수 있습니다.
* 모델의 계보를 추적하고, 프로덕션 모델에 대한 변경 이력을 감사할 수 있습니다.

{{< img src="/images/models/models_landing_page.png" alt="Models overview" >}}

## 작동 방식
아주 간단한 단계만으로 staged 모델을 추적하고 관리할 수 있습니다.

1. **모델 버전 로그**: 트레이닝 스크립트에 몇 줄의 코드만 추가하면 모델 파일을 W&B 아티팩트로 저장할 수 있습니다.
2. **성능 비교**: 라이브 차트를 보며 모델 트레이닝/검증 결과의 메트릭과 예측값을 비교할 수 있습니다. 어떤 모델 버전이 가장 성능이 좋은지도 손쉽게 확인할 수 있습니다.
3. **Registry에 연결**: 가장 뛰어난 모델 버전을 registered model에 연결하여 북마크할 수 있습니다. 이 과정은 Python 코드로도, 혹은 W&B UI에서 직접 할 수도 있습니다.

다음 코드조각은 모델을 Model Registry에 로그하고 연결하는 방법을 보여줍니다:

```python
import wandb
import random

# 새로운 W&B run 시작
run = wandb.init(project="models_quickstart")

# 모델 메트릭 기록 예시
run.log({"acc": random.random()})

# 예시 모델 파일 생성
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# 모델을 Model Registry에 로그 및 연결
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **모델 전이와 CI/CD 워크플로우 연결**: 후보 모델을 워크플로우 스테이지별로 전이시키고, [다운스트림 작업을 자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})하는 Webhook을 연동할 수 있습니다.


## 시작 방법
유스 케이스에 맞게 W&B Models를 시작하려면 다음 리소스를 참고하세요:

* 2부 영상 시리즈를 확인하세요:
  1. [모델 로깅 및 등록하기](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. Model Registry 내 [모델 활용 및 다운스트림 자동화](https://www.youtube.com/watch?v=8PFCrDSeHzw)
* [models walkthrough]({{< relref path="./walkthrough.md" lang="ko" >}})를 읽고, W&B Python SDK 명령어를 활용해 데이터셋 아티팩트를 만들고, 추적하고, 사용하는 방법을 단계별로 따라 해보세요.
* 다음 주제도 함께 알아보세요:
   * [Protected models 및 엑세스 제어]({{< relref path="./access_controls.md" lang="ko" >}})
   * [Registry를 CI/CD 프로세스에 연결하는 방법]({{< relref path="/guides/core/automations/" lang="ko" >}})
   * registered model에 새로운 모델 버전이 연결될 때 [Slack 알림 설정하기]({{< relref path="./notifications.md" lang="ko" >}})
* [What is an ML Model Registry?](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx)를 읽고, Model Registry를 ML 워크플로우에 어떻게 통합할 수 있을지 알아보세요.
* W&B [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) 코스를 수강해보세요. 이 코스에서는 다음을 배울 수 있습니다:
  * W&B Model Registry로 모델을 관리 및 버전 관리하고, 계보 추적, 라이프사이클 단계별 모델 승격을 할 수 있습니다.
  * Webhook을 활용해 모델 관리 워크플로우를 자동화할 수 있습니다.
  * Model Registry를 외부 ML 시스템 및 도구와 연동하여 모델 평가, 모니터링, 배포 등의 모델 개발 라이프사이클 전체에서 어떻게 활용하는지 확인할 수 있습니다.