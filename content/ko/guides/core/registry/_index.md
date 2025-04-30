---
title: Registry
cascade:
- url: guides/core/registry/:filename
menu:
  default:
    identifier: ko-guides-core-registry-_index
    parent: core
url: guides/core/registry
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" >}}

{{% alert %}}
W&B Registry는 현재 공개 미리보기 상태입니다. 배포 유형에 맞게 활성화하는 방법은 [이]({{< relref path="./#enable-wb-registry" lang="ko" >}}) 섹션을 참조하세요.
{{% /alert %}}

W&B Registry는 조직 내에서 [artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 버전들을 체계적으로 관리할 수 있도록 구성된 중앙 저장소입니다. 조직 내에서 [권한을 가진]({{< relref path="./configure_registry.md" lang="ko" >}}) 사용자는 사용자가 속한 팀에 관계없이 모든 artifact의 라이프사이클을 [다운로드]({{< relref path="./download_use_artifact.md" lang="ko" >}}), 공유 및 공동으로 관리할 수 있습니다.

Registry를 사용하여 [artifact 버전 추적]({{< relref path="./link_version.md" lang="ko" >}}), artifact 사용 및 변경 내역 감사, artifact의 거버넌스 및 규정 준수 보장, [모델 CI/CD와 같은 다운스트림 프로세스 자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})를 할 수 있습니다.

요약하면 W&B Registry를 사용하여 다음을 수행할 수 있습니다.

- 기계 학습 작업에 적합한 artifact 버전을 조직의 다른 사용자에게 [홍보]({{< relref path="./link_version.md" lang="ko" >}})합니다.
- 특정 artifact를 찾거나 참조할 수 있도록 [태그로 artifact 구성]({{< relref path="./organize-with-tags.md" lang="ko" >}})합니다.
- [artifact의 계보]({{< relref path="/guides/core/registry/lineage.md" lang="ko" >}})를 추적하고 변경 내역을 감사합니다.
- 모델 CI/CD와 같은 다운스트림 프로세스를 [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})합니다.
- 각 registry에서 artifact에 엑세스할 수 있는 [조직 내 사용자 제한]({{< relref path="./configure_registry.md" lang="ko" >}})합니다.

{{< img src="/images/registry/registry_landing_page.png" alt="" >}}

위의 이미지는 "Model" 및 "Dataset" 코어 registry와 함께 사용자 지정 registry가 있는 Registry App을 보여줍니다.

## 기본 사항 알아보기
각 조직에는 모델 및 데이터셋 artifact를 구성하는 데 사용할 수 있는 **Models** 및 **Datasets**라는 두 개의 registry가 초기에 포함되어 있습니다. [조직의 요구 사항에 따라 다른 artifact 유형을 구성하기 위해 추가 registry를 만들 수 있습니다]({{< relref path="./registry_types.md" lang="ko" >}}).

각 [registry]({{< relref path="./configure_registry.md" lang="ko" >}})는 하나 이상의 [컬렉션]({{< relref path="./create_collection.md" lang="ko" >}})으로 구성됩니다. 각 컬렉션은 고유한 작업 또는 유스 케이스를 나타냅니다.

{{< img src="/images/registry/homepage_registry.png" >}}

artifact를 registry에 추가하려면 먼저 [특정 artifact 버전을 W&B에 기록]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ko" >}})합니다. artifact를 기록할 때마다 W&B는 해당 artifact에 버전을 자동으로 할당합니다. artifact 버전은 0부터 인덱싱되므로 첫 번째 버전은 `v0`, 두 번째 버전은 `v1`과 같습니다.

artifact를 W&B에 기록한 후에는 해당 특정 artifact 버전을 registry의 컬렉션에 연결할 수 있습니다.

{{% alert %}}
"링크"라는 용어는 W&B가 artifact를 저장하는 위치와 registry에서 artifact에 엑세스할 수 있는 위치를 연결하는 포인터를 나타냅니다. W&B는 artifact를 컬렉션에 연결할 때 artifact를 복제하지 않습니다.
{{% /alert %}}

예를 들어, 다음 코드 예제는 "my_model.txt"라는 모델 artifact를 [코어 registry]({{< relref path="./registry_types.md" lang="ko" >}})의 "first-collection"이라는 컬렉션에 기록하고 연결하는 방법을 보여줍니다.

1. W&B run을 초기화합니다.
2. artifact를 W&B에 기록합니다.
3. artifact 버전을 연결할 컬렉션 및 registry의 이름을 지정합니다.
4. artifact를 컬렉션에 연결합니다.

이 Python 코드를 스크립트에 저장하고 실행합니다. W&B Python SDK 버전 0.18.6 이상이 필요합니다.

```python title="hello_collection.py"
import wandb
import random

# track the artifact를 추적하기 위해 W&B run을 초기화합니다.
run = wandb.init(project="registry_quickstart") 

# 기록할 수 있도록 시뮬레이션된 모델 파일을 만듭니다.
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# artifact를 W&B에 기록합니다.
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # artifact 유형을 지정합니다.
)

# artifact를 게시할 컬렉션 및 registry 이름을 지정합니다.
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# artifact를 registry에 연결합니다.
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

반환된 run 오브젝트의 `link_artifact(target_path = "")` 메소드에서 지정한 컬렉션이 지정한 registry 내에 없는 경우 W&B는 자동으로 컬렉션을 만듭니다.

{{% alert %}}
터미널에 출력되는 URL은 W&B가 artifact를 저장하는 프로젝트로 연결됩니다.
{{% /alert %}}

Registry App으로 이동하여 사용자와 조직의 다른 구성원이 게시하는 artifact 버전을 봅니다. 이렇게 하려면 먼저 W&B로 이동합니다. **애플리케이션** 아래 왼쪽 사이드바에서 **Registry**를 선택합니다. "Model" registry를 선택합니다. registry 내에서 연결된 artifact 버전이 있는 "first-collection" 컬렉션을 볼 수 있습니다.

artifact 버전을 registry 내의 컬렉션에 연결하면 조직 구성원은 적절한 권한이 있는 경우 artifact 버전을 보고, 다운로드하고, 관리하고, 다운스트림 자동화를 만들 수 있습니다.

{{% alert %}}
artifact 버전이 (`run.log_artifact()`를 사용하여) 메트릭을 기록하는 경우 해당 버전의 세부 정보 페이지에서 해당 버전에 대한 메트릭을 볼 수 있으며 컬렉션 페이지에서 artifact 버전 간에 메트릭을 비교할 수 있습니다. [registry에서 연결된 artifact 보기]({{< relref path="link_version.md#view-linked-artifacts-in-a-registry" lang="ko" >}})를 참조하십시오.
{{% /alert %}}

## W&B Registry 활성화

배포 유형에 따라 다음 조건을 충족하여 W&B Registry를 활성화합니다.

| 배포 유형 | 활성화 방법 |
| ----- | ----- |
| Multi-tenant Cloud | 별도의 조치가 필요하지 않습니다. W&B Registry는 W&B App에서 사용할 수 있습니다. |
| Dedicated Cloud | 계정 팀에 문의하십시오. SA(Solutions Architect) 팀은 인스턴스의 운영자 콘솔 내에서 W&B Registry를 활성화합니다. 인스턴스가 서버 릴리스 버전 0.59.2 이상인지 확인합니다. |
| Self-Managed   | `ENABLE_REGISTRY_UI`라는 환경 변수를 활성화합니다. 서버에서 환경 변수를 활성화하는 방법에 대한 자세한 내용은 [이 문서]({{< relref path="/guides/hosting/env-vars/" lang="ko" >}})를 참조하십시오. 자체 관리형 인스턴스에서는 인프라 관리자가 이 환경 변수를 활성화하고 `true`로 설정해야 합니다. 인스턴스가 서버 릴리스 버전 0.59.2 이상인지 확인합니다. |

## 시작하기 위한 리소스

유스 케이스에 따라 다음 리소스를 탐색하여 W&B Registry를 시작하십시오.

* 튜토리얼 비디오를 확인하십시오.
    * [Weights & Biases에서 Registry 시작하기](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) 코스를 수강하고 다음 방법을 배우십시오.
    * W&B Registry를 사용하여 artifact를 관리하고 버전 관리하고, 계보를 추적하고, 다양한 라이프사이클 단계를 통해 모델을 홍보합니다.
    * 웹훅을 사용하여 모델 관리 워크플로를 자동화합니다.
    * 모델 평가, 모니터링 및 배포를 위해 registry를 외부 ML 시스템 및 툴과 통합합니다.

## 레거시 Model Registry에서 W&B Registry로 마이그레이션

레거시 Model Registry는 정확한 날짜가 아직 결정되지 않았지만 더 이상 사용되지 않을 예정입니다. 레거시 Model Registry를 더 이상 사용하지 않기 전에 W&B는 레거시 Model Registry의 내용을 W&B Registry로 마이그레이션합니다.

레거시 Model Registry에서 W&B Registry로의 마이그레이션 프로세스에 대한 자세한 내용은 [레거시 Model Registry에서 마이그레이션]({{< relref path="./model_registry_eol.md" lang="ko" >}})을 참조하십시오.

마이그레이션이 발생할 때까지 W&B는 레거시 Model Registry와 새 Registry를 모두 지원합니다.

{{% alert %}}
레거시 Model Registry를 보려면 W&B App에서 Model Registry로 이동하십시오. 페이지 상단에 배너가 나타나 레거시 Model Registry App UI를 사용할 수 있습니다.

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="" >}}
{{% /alert %}}

마이그레이션에 대한 질문이 있거나 W&B Product Team에 우려 사항에 대해 이야기하려면 support@wandb.com으로 문의하십시오.
