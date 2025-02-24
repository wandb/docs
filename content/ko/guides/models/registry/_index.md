---
title: Registry
cascade:
- url: guides/registry/:filename
menu:
  default:
    identifier: ko-guides-models-registry-_index
    parent: w-b-models
url: guides/registry
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" >}}

{{% alert %}}
W&B Registry 는 현재 공개 미리보기 상태입니다. 배포 유형에 맞게 활성화하는 방법은 [이]({{< relref path="./#enable-wb-registry" lang="ko" >}}) 섹션을 참조하세요.
{{% /alert %}}

W&B Registry 는 조직 내 [artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 버전의 선별된 중앙 저장소입니다. 조직 내에서 [권한을 가진]({{< relref path="./configure_registry.md" lang="ko" >}}) 사용자는 사용자가 속한 팀에 관계없이 모든 artifact 의 라이프사이클을 [다운로드]({{< relref path="./download_use_artifact.md" lang="ko" >}}), 공유 및 공동으로 관리할 수 있습니다.

Registry 를 사용하여 [artifact 버전 추적]({{< relref path="./link_version.md" lang="ko" >}}), artifact 사용 및 변경 이력 감사, artifact 의 거버넌스 및 규정 준수 보장, [모델 CI/CD 와 같은 다운스트림 프로세스 자동화]({{< relref path="/guides/models/automations/" lang="ko" >}})를 할 수 있습니다.

요약하자면, W&B Registry 를 사용하여 다음을 수행합니다.

- 기계 학습 작업을 충족하는 artifact 버전을 조직의 다른 사용자에게 [홍보]({{< relref path="./link_version.md" lang="ko" >}})합니다.
- 특정 artifact 를 찾거나 참조할 수 있도록 [태그로 artifact 구성]({{< relref path="./organize-with-tags.md" lang="ko" >}})합니다.
- [artifact 의 계보]({{< relref path="/guides/models/registry/lineage.md" lang="ko" >}})를 추적하고 변경 이력을 감사합니다.
- 모델 CI/CD 와 같은 다운스트림 프로세스를 [자동화]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ko" >}})합니다.
- [조직 내에서 누가]({{< relref path="./configure_registry.md" lang="ko" >}}) 각 registry 의 artifact 에 엑세스할 수 있는지 제한합니다.

{{< img src="/images/registry/registry_landing_page.png" alt="" >}}

위의 이미지는 "Model" 및 "Dataset" 코어 registry 와 함께 사용자 정의 registry 가 있는 Registry App 을 보여줍니다.

## 기본 사항 학습
각 조직은 초기에는 **Models** 및 **Datasets** 라는 모델 및 데이터셋 artifact 를 구성하는 데 사용할 수 있는 두 개의 registry 를 포함합니다. [조직의 요구 사항에 따라 다른 artifact 유형을 구성하기 위해 추가 registry 를 만들 수 있습니다]({{< relref path="./registry_types.md" lang="ko" >}}).

각 [registry]({{< relref path="./configure_registry.md" lang="ko" >}})는 하나 이상의 [컬렉션]({{< relref path="./create_collection.md" lang="ko" >}})으로 구성됩니다. 각 컬렉션은 고유한 작업 또는 유스 케이스를 나타냅니다.

{{< img src="/images/registry/homepage_registry.png" >}}

artifact 를 registry 에 추가하려면 먼저 [특정 artifact 버전을 W&B 에 기록]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ko" >}})합니다. artifact 를 기록할 때마다 W&B 는 해당 artifact 에 버전을 자동으로 할당합니다. artifact 버전은 0 인덱싱을 사용하므로 첫 번째 버전은 `v0`, 두 번째 버전은 `v1` 등입니다.

W&B 에 artifact 를 기록한 후에는 해당 특정 artifact 버전을 registry 의 컬렉션에 연결할 수 있습니다.

{{% alert %}}
"링크"라는 용어는 W&B 가 artifact 를 저장하는 위치와 registry 에서 artifact 에 엑세스할 수 있는 위치를 연결하는 포인터를 나타냅니다. artifact 를 컬렉션에 연결할 때 W&B 는 artifact 를 복제하지 않습니다.
{{% /alert %}}

예를 들어, 다음 코드 예제는 "my_model.txt"라는 가짜 모델 artifact 를 [코어 Model registry]({{< relref path="./registry_types.md" lang="ko" >}})의 "first-collection"이라는 컬렉션에 기록하고 연결하는 방법을 보여줍니다. 더 구체적으로 말하면, 코드는 다음을 수행합니다.

1. artifact 를 추적하기 위해 W&B run 을 초기화합니다.
2. artifact 를 W&B 에 기록합니다.
3. artifact 버전을 연결할 컬렉션 및 registry 의 이름을 지정합니다.
4. artifact 를 컬렉션에 연결합니다.

다음 코드 조각을 복사하여 Python 스크립트에 붙여넣고 실행합니다. W&B Python SDK 버전 0.18.6 이상이 있는지 확인합니다.

```python title="hello_collection.py"
import wandb
import random

# Track the artifact by initializing a W&B run
run = wandb.init(project="registry_quickstart") 

# Create a simulated model file so you can log it
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# Log the artifact to W&B
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # Specifies artifact type
)

# Specify the name of the collection and registry
# you want to publish the artifact to
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# Link the artifact to the registry
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

반환된 run 오브젝트의 `link_artifact(target_path = "")` 메소드에서 지정한 컬렉션이 지정한 registry 내에 존재하지 않는 경우 W&B 는 자동으로 컬렉션을 만듭니다.

{{% alert %}}
터미널에 출력되는 URL 은 W&B 가 artifact 를 저장하는 project 로 연결됩니다.
{{% /alert %}}

Registry App 으로 이동하여 귀하와 조직의 다른 구성원이 게시하는 artifact 버전을 봅니다. 이렇게 하려면 먼저 W&B 로 이동합니다. **Applications** 아래의 왼쪽 사이드바에서 **Registry** 를 선택합니다. "Model" registry 를 선택합니다. registry 내에서 연결된 artifact 버전이 있는 "first-collection" 컬렉션이 표시됩니다.

artifact 버전을 registry 내의 컬렉션에 연결하면 조직 구성원은 적절한 권한이 있는 경우 artifact 버전을 보고, 다운로드하고, 관리하고, 다운스트림 자동화를 만들 수 있습니다.

## W&B Registry 활성화

배포 유형에 따라 다음 조건을 충족하여 W&B Registry 를 활성화합니다.

| 배포 유형 | 활성화 방법 |
| ----- | ----- |
| Multi-tenant Cloud | 작업이 필요하지 않습니다. W&B Registry 는 W&B App 에서 사용할 수 있습니다. |
| Dedicated Cloud | 계정 팀에 문의하십시오. SA (Solutions Architect) 팀은 인스턴스의 운영자 콘솔 내에서 W&B Registry 를 활성화합니다. 인스턴스가 서버 릴리스 버전 0.59.2 이상인지 확인하십시오. |
| Self-Managed | `ENABLE_REGISTRY_UI` 라는 환경 변수를 활성화합니다. 서버에서 환경 변수를 활성화하는 방법에 대한 자세한 내용은 [이 문서]({{< relref path="/guides/hosting/env-vars/" lang="ko" >}})를 참조하십시오. 자체 관리 인스턴스에서 인프라 관리자는 이 환경 변수를 활성화하고 `true` 로 설정해야 합니다. 인스턴스가 서버 릴리스 버전 0.59.2 이상인지 확인하십시오. |

## 시작하기 위한 리소스

유스 케이스에 따라 다음 리소스를 탐색하여 W&B Registry 를 시작하십시오.

* 튜토리얼 비디오를 확인하십시오.
    * [Weights & Biases 에서 Registry 시작하기](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B [모델 CI/CD](https://www.wandb.courses/courses/enterprise-model-management) 코스를 수강하고 다음 방법을 배우십시오.
    * W&B Registry 를 사용하여 artifact 를 관리하고 버전 관리하고, 계보를 추적하고, 다양한 라이프사이클 단계를 통해 모델을 승격합니다.
    * 웹훅을 사용하여 모델 관리 워크플로우를 자동화합니다.
    * 모델 평가, 모니터링 및 배포를 위해 registry 를 외부 ML 시스템 및 툴과 통합합니다.

## 레거시 Model Registry 에서 W&B Registry 로 마이그레이션

레거시 Model Registry 는 정확한 날짜가 아직 결정되지 않은 채로 더 이상 사용되지 않을 예정입니다. 레거시 Model Registry 를 더 이상 사용하지 않기 전에 W&B 는 레거시 Model Registry 의 내용을 W&B Registry 로 마이그레이션합니다.

레거시 Model Registry 에서 W&B Registry 로의 마이그레이션 프로세스에 대한 자세한 내용은 [레거시 Model Registry 에서 마이그레이션]({{< relref path="./model_registry_eol.md" lang="ko" >}})을 참조하십시오.

마이그레이션이 발생할 때까지 W&B 는 레거시 Model Registry 와 새 Registry 를 모두 지원합니다.

{{% alert %}}
레거시 Model Registry 를 보려면 W&B App 에서 Model Registry 로 이동합니다. 페이지 상단에 레거시 Model Registry App UI 를 사용할 수 있는 배너가 나타납니다.

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="" >}}
{{% /alert %}}

질문이 있거나 마이그레이션에 대한 우려 사항에 대해 W&B 제품 팀과 이야기하려면 support@wandb.com 으로 문의하십시오.
