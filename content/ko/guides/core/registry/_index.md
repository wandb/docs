---
title: 레지스트리
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

W&B Registry는 조직 내에서 [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 버전을 관리하는 엄선된 중앙 저장소입니다. 조직 내에서 [권한을 가진 사용자]({{< relref path="./configure_registry.md" lang="ko" >}})는 [artifact를 다운로드하고 사용]({{< relref path="./download_use_artifact.md" lang="ko" >}})할 수 있으며, 팀에 상관없이 모든 artifact의 라이프사이클을 공유하고 협업하여 관리할 수 있습니다.

Registry를 통해 [artifact 버전 추적]({{< relref path="./link_version.md" lang="ko" >}}), artifact 사용 및 변경 이력 감사, artifact 거버넌스 및 컴플라이언스 준수 보장, [모델 CI/CD와 같은 다운스트림 프로세스의 자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})가 가능합니다.

요약하면, W&B Registry는 다음과 같은 용도로 사용합니다:

- 조직 내 다른 사용자들에게 기계학습 작업에 적합한 artifact 버전을 [승격]({{< relref path="./link_version.md" lang="ko" >}})합니다.
- [태그로 artifact를 정리]({{< relref path="./organize-with-tags.md" lang="ko" >}})하여 특정 artifact를 쉽게 찾고 참조할 수 있습니다.
- [artifact의 계보]({{< relref path="/guides/core/registry/lineage.md" lang="ko" >}})를 추적하고 변경 이력을 감사합니다.
- 모델 CI/CD 등 [다운스트림 프로세스를 자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})합니다.
- 각 registry에 엑세스할 수 있는 [조직 내 권한을 제한]({{< relref path="./configure_registry.md" lang="ko" >}})할 수 있습니다.



{{< img src="/images/registry/registry_landing_page.png" alt="W&B Registry" >}}

위 이미지는 Registry App에서 "Model"과 "Dataset"의 코어 registry와 커스텀 registry를 함께 보여줍니다.


## 기본 사항 알아보기
각 조직에는 처음에 모델과 데이터셋 artifact를 관리할 수 있는 **Models** 및 **Datasets**라는 두 개의 registry가 포함되어 있습니다. [조직의 필요에 따라 다른 artifact 타입을 관리할 추가 registry를 만들 수도 있습니다]({{< relref path="./registry_types.md" lang="ko" >}}).

각 [registry]({{< relref path="./configure_registry.md" lang="ko" >}})는 하나 이상의 [collection]({{< relref path="./create_collection.md" lang="ko" >}})으로 구성됩니다. 각 collection은 고유한 작업 또는 유스 케이스를 나타냅니다.

{{< img src="/images/registry/homepage_registry.png" alt="W&B Registry" >}}

artifact를 registry에 추가하려면 먼저 [특정 artifact 버전을 W&B에 로그]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ko" >}})해야 합니다. artifact를 로그할 때마다 W&B는 해당 artifact에 버전을 자동으로 할당합니다. artifact 버전은 0번 인덱스를 사용하기 때문에 첫 번째 버전은 `v0`, 두 번째 버전은 `v1` 등으로 표시됩니다.

artifact를 W&B에 로그한 후, 해당 artifact 버전을 registry의 collection에 연결할 수 있습니다.

{{% alert %}}
"link"라는 용어는 W&B가 artifact를 저장하는 위치와 registry에서 artifact에 엑세스할 수 있는 위치를 연결하는 포인터를 의미합니다. artifact를 collection에 링크해도 W&B는 artifact를 복제하지 않습니다.
{{% /alert %}}

예를 들어, 아래의 코드 예시는 "my_model.txt"라는 모델 artifact를 "first-collection"이라는 collection에 [core registry]({{< relref path="./registry_types.md" lang="ko" >}})로 로그하고 링크하는 방법을 보여줍니다:

1. W&B Run을 초기화합니다.
2. artifact를 W&B에 로그합니다.
3. artifact 버전을 링크할 collection과 registry의 이름을 지정합니다.
4. artifact를 collection에 링크합니다.

아래 Python 코드를 스크립트로 저장하고 실행하세요. W&B Python SDK 버전 0.18.6 이상이 필요합니다.

```python title="hello_collection.py"
import wandb
import random

# artifact 추적을 위해 W&B Run을 초기화합니다.
run = wandb.init(project="registry_quickstart") 

# 로그할 모델 파일을 임의로 생성합니다.
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# artifact를 W&B에 로그합니다.
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # artifact 타입 지정
)

# collection과 registry의 이름을 지정합니다.
# artifact를 게시할 대상입니다.
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# artifact를 registry에 링크합니다.
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

`link_artifact(target_path = "")` 메소드에 지정한 collection이 해당 registry에 존재하지 않을 경우, W&B는 collection을 자동으로 생성합니다.

{{% alert %}}
터미널에 출력되는 URL은 W&B가 artifact를 저장한 프로젝트로 이동합니다.
{{% /alert %}}

Registry App으로 이동하면, 조직 내에서 여러분과 동료들이 게시한 artifact 버전을 확인할 수 있습니다. 방법은 다음과 같습니다. 우선 W&B에 접속하세요. 좌측 사이드바에서 **Applications** 아래의 **Registry**를 선택합니다. "Model" registry를 선택하면 registry 내에서 "first-collection" collection과 그 안에 링크된 artifact 버전을 볼 수 있습니다.

artifact 버전을 registry 내 collection에 링크하면, 조직 구성원은 (적절한 권한이 있다면) artifact 버전 보기, 다운로드, 관리, 다운스트림 자동화 생성 등을 할 수 있습니다.

{{% alert %}}
artifact 버전이 메트릭을 로그했다면(예: `run.log_artifact()` 사용), 해당 버전의 상세 페이지에서 메트릭을 볼 수 있으며, collection 페이지에서 artifact 버전 간 메트릭을 비교할 수 있습니다. 자세한 내용은 [링크된 artifact 보기]({{< relref path="link_version.md#view-linked-artifacts-in-a-registry" lang="ko" >}})를 참고하세요.
{{% /alert %}}

## W&B Registry 활성화하기

배포 유형에 따라, W&B Registry를 활성화하려면 다음 조건을 충족해야 합니다:

| 배포 유형 | 활성화 방법 |
| ----- | ----- |
| Multi-tenant Cloud | 별도의 조치 필요 없음. W&B App에서 W&B Registry를 바로 사용할 수 있습니다. |
| Dedicated Cloud | 배포에 W&B Registry를 활성화하려면 담당 계정팀에 문의하세요. |
| Self-Managed | 환경 변수 `ENABLE_REGISTRY_UI`를 `true`로 설정하세요. [환경 변수 설정 방법]({{< relref path="/guides/hosting/env-vars.md" lang="ko" >}})을 참고하세요. Server v0.59.2 이상 필요합니다. |


## 시작을 위한 리소스

유스 케이스에 따라, W&B Registry로 시작하려면 다음 리소스를 참고하세요:

* 튜토리얼 영상 확인:
    * [W&B Registry 시작하기](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) 코스를 수강하고 아래 내용을 배울 수 있습니다:
    * W&B Registry를 사용하여 artifact 관리 및 버전 관리, 계보 추적, 모델 라이프사이클 여러 단계에서 모델 승격하는 방법
    * 웹훅을 활용한 모델 관리 워크플로우 자동화
    * 모델 평가, 모니터링, 배포를 위한 외부 ML 시스템 및 툴과 registry 연동 방법



## 기존 Model Registry에서 W&B Registry로 마이그레이션

기존 Model Registry는 아직 정확한 종료 날짜가 정해지지 않았으나, 향후 지원이 중단될 예정입니다. 지원 종료 전, W&B는 기존 Model Registry의 모든 내용을 새 W&B Registry로 마이그레이션할 예정입니다.

기존 Model Registry에서 W&B Registry로의 마이그레이션 프로세스에 대한 자세한 내용은 [기존 Model Registry 마이그레이션]({{< relref path="./model_registry_eol.md" lang="ko" >}})을 참고하세요.

마이그레이션이 완료될 때까지는 기존 Model Registry와 새 Registry를 모두 사용할 수 있습니다.

{{% alert %}}
기존 Model Registry를 보려면 W&B App에서 Model Registry로 이동하세요. 페이지 상단의 배너에서 기존 Model Registry App UI로 전환할 수 있습니다.

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="Legacy Model Registry UI" >}}
{{% /alert %}}


마이그레이션에 대해 궁금한 점이나 W&B Product Team과 직접 상담이 필요한 경우 support@wandb.com 으로 문의하세요.