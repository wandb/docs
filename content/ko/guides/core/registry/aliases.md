---
title: 에일리어스를 사용하여 아티팩트 버전에 참조하기
menu:
  default:
    identifier: ko-guides-core-registry-aliases
weight: 5
---

하나 이상의 에일리어스를 사용하여 특정 [artifact 버전]({{< relref path="guides/core/artifacts/create-a-new-artifact-version" lang="ko" >}})을 참조할 수 있습니다. [W&B는 동일한 이름으로 연결된 각 artifact에 대해 자동으로 에일리어스]({{< relref path="aliases#default-aliases" lang="ko" >}})를 지정합니다. 또한 [사용자 지정 에일리어스]({{< relref path="aliases#custom-aliases" lang="ko" >}})를 생성하여 특정 artifact 버전을 참조할 수도 있습니다.

에일리어스는 Registry UI에서 해당 에일리어스의 이름이 표시된 사각형으로 나타납니다. [에일리어스가 보호된 경우]({{< relref path="aliases#protected-aliases" lang="ko" >}}), 잠금 아이콘이 있는 회색 사각형으로 표시됩니다. 그렇지 않으면 주황색 사각형으로 표시됩니다. 에일리어스는 레지스트리 간에 공유되지 않습니다.

{{% alert title="에일리어스와 태그 사용 시점" %}}
에일리어스는 특정 artifact 버전을 참조하는 데 사용하세요. 컬렉션 내의 각 에일리어스는 고유합니다. 동일한 에일리어스는 한 번에 하나의 artifact 버전에만 지정될 수 있습니다.

태그는 공통 주제에 따라 artifact 버전이나 컬렉션을 조직하고 그룹화하는 데 사용하세요. 여러 artifact 버전과 컬렉션에 동일한 태그를 지정할 수 있습니다.
{{% /alert %}}

artifact 버전에 에일리어스를 추가할 때, 옵션으로 [Registry 자동화]({{< relref path="/guides/core/automations/automation-events/#registry" lang="ko" >}})를 시작하여 Slack 채널에 알림을 보내거나 웹훅을 트리거할 수 있습니다.

## 기본 에일리어스

W&B는 동일한 이름으로 연결한 각 artifact 버전에 대해 아래와 같은 에일리어스를 자동으로 지정합니다.

* 컬렉션에 가장 최근에 연결된 artifact 버전에 `latest` 에일리어스를 지정합니다.
* 고유 버전 번호를 부여합니다. W&B는 연결된 각 artifact 버전의 수(0부터 시작)를 계산하여, 해당 숫자를 고유 버전 번호로 지정합니다.

예를 들어, `zoo_model`이라는 artifact를 세 번 연결하면, W&B는 각각 `v0`, `v1`, `v2`라는 세 개의 에일리어스를 생성합니다. `v2`에는 `latest` 에일리어스도 함께 지정됩니다.

## 사용자 지정 에일리어스

고유한 유스 케이스에 따라 특정 artifact 버전에 하나 이상의 사용자 지정 에일리어스를 생성할 수 있습니다. 예를 들어:

- 모델이 어떤 데이터셋에서 훈련되었는지 식별하기 위해 `dataset_version_v0`, `dataset_version_v1`, `dataset_version_v2` 와 같은 에일리어스를 사용할 수 있습니다.
- 가장 성능이 좋은 artifact 모델 버전을 추적하기 위해 `best_model` 에일리어스를 사용할 수 있습니다.

해당 레지스트리에서 [Member 또는 Admin 레지스트리 역할]({{< relref path="guides/core/registry/configure_registry/#registry-roles" lang="ko" >}})을 가진 사용자는 연결된 artifact에서 사용자 지정 에일리어스를 추가하거나 제거할 수 있습니다. 필요한 경우, [보호된 에일리어스]({{< relref path="aliases/#protected-aliases" lang="ko" >}})를 사용하여 수정이나 삭제로부터 보호할 artifact 버전을 구분할 수 있습니다.

W&B Registry UI 또는 Python SDK를 통해 사용자 지정 에일리어스를 생성할 수 있습니다. 유스 케이스에 따라 아래 탭에서 상황에 맞는 안내를 참고하세요.

{{< tabpane text=true >}}
{{% tab header="W&B Registry" value="app" %}}

1. W&B Registry로 이동합니다.
2. 컬렉션에서 **View details** 버튼을 클릭합니다.
3. **Versions** 섹션에서 특정 artifact 버전 옆에 있는 **View** 버튼을 클릭합니다.
4. **Aliases** 필드 옆의 **+** 버튼을 클릭하여 하나 이상의 에일리어스를 추가합니다.

{{% /tab %}}

{{% tab header="Python SDK" value="python" %}}
Python SDK로 artifact 버전을 컬렉션에 연결할 때, `alias` 파라미터에 하나 이상의 에일리어스를 인수로 제공할 수 있습니다. 이때 에일리어스가 존재하지 않으면, W&B가 새로운 에일리어스([비보호 에일리어스]({{< relref path="#custom-aliases" lang="ko" >}}))를 생성합니다.

아래 코드조각은 artifact 버전을 컬렉션에 연결하고 에일리어스를 추가하는 방법을 보여줍니다. `< >` 부분은 여러분의 값으로 대체하세요.

```python
import wandb

# Run을 초기화합니다.
run = wandb.init(entity = "<team_entity>", project = "<project_name>")

# 아티팩트 오브젝트를 생성합니다.
# type 파라미터는 아티팩트 오브젝트의 타입과 컬렉션 타입을 지정합니다.
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# 파일을 아티팩트 오브젝트에 추가합니다.
# 로컬 머신에 있는 파일 경로를 지정합니다.
artifact.add_file(local_path = "<local_path_to_artifact>")

# 아티팩트를 연결할 컬렉션과 레지스트리를 지정합니다.
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# 아티팩트 버전을 컬렉션에 연결합니다.
# 이 artifact 버전에 하나 이상의 에일리어스를 추가합니다.
run.link_artifact(
    artifact = artifact, 
    target_path = target_path, 
    aliases = ["<alias_1>", "<alias_2>"]
    )
```
{{% /tab %}}
{{< /tabpane >}}

### 보호된 에일리어스
[보호된 에일리어스]({{< relref path="aliases/#protected-aliases" lang="ko" >}})는 수정하거나 삭제하지 않아야 할 artifact 버전을 구분하고 식별하기 위해 사용하세요. 예를 들어, 조직에서 기계학습 프로덕션 파이프라인에 사용 중인 artifact 버전을 표시하기 위해 `production` 보호 에일리어스를 사용할 수 있습니다.

[Registry 관리자]({{< relref path="/guides/core/registry/configure_registry/#registry-roles" lang="ko" >}})와 [서비스 계정]({{< relref path="/support/kb-articles/service_account_useful" lang="ko" >}})이 Admin 역할을 가진 경우 보호된 에일리어스를 생성하거나, artifact 버전에서 보호된 에일리어스를 추가/제거할 수 있습니다. Member와 Viewer는 보호 버전의 연결을 해제하거나 보호된 에일리어스가 포함된 컬렉션을 삭제할 수 없습니다. 자세한 내용은 [레지스트리 엑세스 구성]({{< relref path="/guides/core/registry/configure_registry.md" lang="ko" >}})을 참고하세요.

자주 사용하는 보호된 에일리어스 예시는 다음과 같습니다.

- **Production**: 이 artifact 버전은 프로덕션용으로 준비되었습니다.
- **Staging**: 이 artifact 버전은 테스트용으로 준비되었습니다.

#### 보호된 에일리어스 생성 방법

W&B Registry UI에서 보호된 에일리어스를 생성하는 단계는 다음과 같습니다.

1. Registry App으로 이동합니다.
2. 레지스트리를 선택합니다.
3. 페이지 오른쪽 상단의 톱니바퀴 아이콘을 클릭해 레지스트리 설정을 확인합니다.
4. **Protected Aliases** 섹션에서 **+** 버튼을 클릭하여 보호된 에일리어스를 추가합니다.

생성 후 각 보호된 에일리어스는 **Protected Aliases** 섹션에 잠금 아이콘이 표시된 회색 사각형으로 나타납니다.  

{{% alert %}}
일반(비보호) 에일리어스와는 달리, 보호된 에일리어스 생성은 W&B Registry UI에서만 가능합니다. (Python SDK로는 생성 불가) 이미 생성된 보호된 에일리어스를 artifact 버전에 추가할 때에는 W&B Registry UI 또는 Python SDK를 사용할 수 있습니다.
{{% /alert %}}

W&B Registry UI에서 artifact 버전에 보호된 에일리어스를 추가하는 방법은 다음과 같습니다.

1. W&B Registry로 이동합니다.
2. 컬렉션에서 **View details** 버튼을 클릭합니다.
3. **Versions** 섹션에서 특정 artifact 버전 옆의 **View** 버튼을 클릭합니다.
4. **Aliases** 필드 옆의 **+** 버튼을 클릭하여 하나 이상의 보호된 에일리어스를 추가합니다.

보호된 에일리어스가 생성된 후, 관리자는 Python SDK를 이용해 artifact 버전에 이를 추가할 수 있습니다. 실제 사용 예시는 위 [사용자 지정 에일리어스 생성하기](#custom-aliases) 섹션에서 Registry 및 Python SDK 탭을 참고하세요.

## 기존 에일리어스 찾기
[W&B Registry의 글로벌 검색 창]({{< relref path="/guides/core/registry/search_registry/#search-for-registry-items" lang="ko" >}})을 이용해 기존 에일리어스를 찾을 수 있습니다. 보호된 에일리어스를 찾으려면:

1. W&B Registry App으로 이동합니다.
2. 페이지 상단의 검색 창에 검색어를 입력하고 Enter 키를 누릅니다.

입력한 내용이 기존의 레지스트리, 컬렉션명, artifact 버전 태그, 컬렉션 태그, 또는 에일리어스와 일치하면 검색결과가 검색창 아래에 나타납니다.

## 예시

{{% alert %}}
아래 코드 예시는 [W&B Registry 튜토리얼](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb)의 연장선입니다. 아래 코드를 사용하려면 먼저 [노트북에서 설명된 방식대로 Zoo 데이터셋을 받아서 가공해야 합니다](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb#scrollTo=87fecd29-8146-41e2-86fb-0bb4e3e3350a). Zoo 데이터셋이 준비되면, artifact 버전을 만들고 여기에 사용자 지정 에일리어스를 지정할 수 있습니다.
{{% /alert %}}

아래 코드조각은 artifact 버전을 생성하고, 여기에 사용자 지정 에일리어스를 추가하는 방법을 보여줍니다. 예시는 [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/111/zoo)의 Zoo 데이터셋과 `Zoo_Classifier_Models` 레지스트리의 `Model` 컬렉션을 사용합니다.

```python
import wandb

# Run을 초기화합니다.
run = wandb.init(entity = "smle-reg-team-2", project = "zoo_experiment")

# 아티팩트 오브젝트를 생성합니다.
# type 파라미터는 아티팩트 오브젝트와 컬렉션 타입을 모두 지정합니다.
artifact = wandb.Artifact(name = "zoo_dataset", type = "dataset")

# 파일을 아티팩트 오브젝트에 추가합니다.
# 로컬 머신 경로를 지정하세요.
artifact.add_file(local_path="zoo_dataset.pt", name="zoo_dataset")
artifact.add_file(local_path="zoo_labels.pt", name="zoo_labels")

# 아티팩트를 연결할 컬렉션과 레지스트리를 지정합니다.
REGISTRY_NAME = "Model"
COLLECTION_NAME = "Zoo_Classifier_Models"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# 아티팩트 버전을 컬렉션에 연결합니다.
# 이 artifact 버전에 하나 이상의 에일리어스를 추가합니다.
run.link_artifact(
    artifact = artifact,
    target_path = target_path,
    aliases = ["production-us", "production-eu"]
    )
```

1. 먼저 artifact 오브젝트(`wandb.Artifact()`)를 생성합니다.
2. 다음으로, 두 개의 데이터셋 PyTorch tensor를 `wandb.Artifact.add_file()`로 artifact 오브젝트에 추가합니다.
3. 마지막으로, `link_artifact()`로 artifact 버전을 `Zoo_Classifier_Models` 레지스트리의 `Model` 컬렉션에 연결하고, 두 개의 사용자 지정 에일리어스(`production-us`, `production-eu`)를 `aliases` 파라미터에 인수로 전달하여 추가합니다.