---
title: Registry
slug: /guides/registry
displayed_sidebar: default
---

:::info
W&B Registry는 이제 퍼블릭 프리뷰 상태입니다. 배포 유형에 대한 활성화 방법을 알아보려면 [이 섹션](#enable-wb-registry)을 방문하세요.
:::

W&B Registry는 모델과 데이터셋의 버전 관리, 에일리어스, 계보 추적 및 거버넌스를 제공하는 큐레이션된 중앙 저장소입니다. Registry는 조직 내의 개인과 팀이 모든 모델, 데이터셋 및 기타 아티팩트의 수명 주기를 공유하고 협업하여 관리할 수 있도록 해줍니다. 프로덕션에 있는 모델의 유일한 진실의 원천으로서, Registry는 적절한 모델을 식별하여 재현, 재트레이닝, 평가 및 배포할 수 있도록 효과적인 CI/CD 파이프라인의 기초를 제공합니다.

![](/images/registry/registry_landing_page.png)

W&B Registry를 사용하여 다음을 수행할 수 있습니다:

- 각 기계학습 작업에 대한 최고의 아티팩트를 [즐겨찾기 등록](./link_version.md)하세요.
- 하류 프로세스 및 모델 CI/CD를 [자동화](../model_registry/model-registry-automations.md) 하세요.
- 프로덕션 아티팩트 변경 이력을 감사하고 [아티팩트 계보](../model_registry/model-lineage.md)를 추적하세요.
- 전체 조직 사용자를 위한 뷰어, 멤버, 관리자 엑세스를 레지스트리에 대해 [구성](./configure_registry.md)하세요.
- 에일리어스로 알려진 고유한 식별자로 중요한 아티팩트를 빠르게 찾거나 참조하세요.
- [태그](./organize-with-tags.md)를 사용하여 레지스트리에서 자산을 레이블로 지정하고 그룹화하고 발견하세요.

## 작동 방식

이 몇 가지 단계를 통해 W&B Registry에 배포할 아티팩트를 추적하고 게시하세요:

1. 아티팩트 버전 로그 기록: 트레이닝 또는 실험 스크립트에 몇 줄의 코드를 추가하여 아티팩트를 W&B run에 저장합니다.
2. 레지스트리에 연결: 가장 관련 있고 가치 있는 아티팩트 버전을 레지스트리에 연결하여 즐겨찾기 등록합니다.

다음 코드조각은 모델을 W&B Registry 내부의 모델 레지스트리에 로그 및 연결하는 방법을 보여줍니다:

```python
import wandb
import random

# 새로운 W&B run을 시작하여 실험을 추적합니다
run = wandb.init(project="registry_quickstart") 

# 모델 메트릭을 로그하도록 시뮬레이션합니다
run.log({"acc": random.random()})

# 시뮬레이션된 모델 파일을 생성합니다
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# W&B Registry 내부의 모델 레지스트리에 모델을 로그하고 연결합니다
logged_artifact = run.log_artifact(artifact_or_path="./my_model.txt", name="gemma-finetuned-3twsov9e", type="model")
run.link_artifact(artifact=logged_artifact, target_path=f"<INSERT-ORG-NAME>/wandb-registry-model/registry-quickstart-collection"),

run.finish()
```
레지스트리에 연결에 대해 더 알아보려면 [이 가이드](/guides/registry/link_version)를 방문하세요.

## W&B Registry 활성화

배포 유형에 따라 W&B Registry를 활성화하기 위해 다음 조건을 충족해야 합니다:

| 배포 유형 | 활성화 방법 |
| ----- | ----- |
| 멀티 테넌트 클라우드 | 별도의 액션이 필요 없습니다. W&B Registry는 W&B App에서 사용 가능합니다. |
| 전용 클라우드 | 계정 팀에 연락하세요. 솔루션 아키텍트(SA) 팀은 사용자의 인스턴스 운영자 콘솔 내에서 W&B Registry를 활성화합니다. 인스턴스가 서버 릴리스 버전 0.59.2 이상이어야 합니다.|
| 자체 관리 | `ENABLE_REGISTRY_UI`라는 환경 변수를 활성화하세요. 서버에서 환경 변수를 활성화하는 방법에 대해 더 알아보려면 [이 문서들](/guides/hosting/env-vars)을 방문하세요. 자체 관리 인스턴스에서는 인프라 관리자가 이 환경 변수를 활성화하고 `true`로 설정해야 합니다. 인스턴스가 서버 릴리스 버전 0.59.2 이상이어야 합니다.|

## 시작하기 위한 리소스

사용 사례에 따라 W&B Registry를 시작하기 위해 다음 리소스를 탐색하세요:

* 튜토리얼 비디오를 확인하세요:
    * [Weights & Biases의 Registry 시작하기](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) 코스를 수강하고 다음을 배워보세요:
    * W&B Registry를 사용하여 아티팩트를 관리하고, 버전 관리하고 계보를 추적하며, 모델을 다양한 수명주기 단계에 따라 승격시키는 방법.
    * 웹훅과 런치 작업을 사용하여 모델 관리 워크플로우를 자동화하는 방법.
    * 모델 평가, 모니터링 및 배포를 위한 모델 개발 수명주기 내 외부 ML 시스템 및 도구와의 Registry 통합을 확인하는 방법.

## 레거시 모델 레지스트리에서 W&B Registry로 마이그레이션

정확한 날짜는 결정되지 않았으나 레거시 모델 레지스트리가 폐기될 예정입니다. 레거시 모델 레지스트리를 폐기하기 전에 W&B는 레거시 모델 레지스트리의 내용을 W&B Registry로 마이그레이션할 것입니다.

레거시 모델 레지스트리에서 마이그레이션 프로세스에 대해 더 많은 정보를 얻으려면 [레거시 모델 레지스트리에서 마이그레이션](./model_registry_eol.md)을 참조하세요.

마이그레이션이 발생하기 전까지는 W&B는 레거시 모델 레지스트리와 새로운 Registry 모두를 지원합니다.

:::info
레거시 모델 레지스트리를 보려면 W&B App에서 모델 레지스트리로 이동하세요. 페이지 상단에 레거시 모델 레지스트리 App UI를 사용할 수 있도록 하는 배너가 표시됩니다.

![](/images/registry/nav_to_old_model_reg.gif)
:::

마이그레이션에 대한 질문이 있거나 W&B 제품 팀과 상담을 원하시면 support@wandb.com으로 연락하세요.