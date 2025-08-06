---
title: 컬렉션 생성
menu:
  default:
    identifier: ko-guides-core-registry-create_collection
    parent: registry
weight: 4
---

*컬렉션*은 레지스트리 내에서 서로 연결된 아티팩트 버전들의 집합입니다. 각 컬렉션은 서로 다른 작업(Task)이나 유스 케이스를 나타냅니다.

예를 들어, 코어 Dataset 레지스트리 내에 여러 개의 컬렉션이 있을 수 있습니다. 각 컬렉션은 MNIST, CIFAR-10, ImageNet 등 서로 다른 데이터셋을 포함합니다.

또 다른 예시로, "chatbot"이라는 레지스트리가 있다면, 모델 아티팩트 컬렉션, 데이터셋 아티팩트 컬렉션, 파인튜닝된 모델 아티팩트 컬렉션 등 여러 컬렉션을 가질 수 있습니다.

레지스트리와 각 컬렉션을 어떻게 구성할지는 전적으로 여러분에게 달려 있습니다.

{{% alert %}}
W&B Model Registry에 익숙하다면 registered models에 대해 알고 계실 수 있습니다. Model Registry에서의 registered models는 이제 W&B Registry에서 컬렉션(collections)이라고 불립니다.
{{% /alert %}}

## Collection types

각 컬렉션은 한 가지, 오직 *하나의* 아티팩트 타입만을 허용합니다. 지정한 타입은 여러분과 소속 조직의 다른 구성원이 해당 컬렉션에 연결할 수 있는 아티팩트를 제한합니다.

{{% alert %}}
아티팩트 타입은 프로그래밍 언어, 예를 들어 Python의 데이터 타입과 비슷하다고 생각할 수 있습니다. 이 비유에서 컬렉션은 문자열, 정수, 또는 실수를 저장할 수 있지만, 여러 데이터 타입을 섞어서 저장할 수는 없습니다.
{{% /alert %}}

예를 들어, "dataset" 아티팩트 타입을 허용하는 컬렉션을 만든다고 가정해 봅시다. 이 경우 해당 컬렉션에는 "dataset" 타입의 향후 아티팩트 버전들만 연결할 수 있습니다. 마찬가지로, "model" 타입만 허용하는 컬렉션에는 "model" 타입의 아티팩트만 연결할 수 있습니다.

{{% alert %}}
아티팩트 오브젝트를 생성할 때 아티팩트의 타입을 지정합니다. `wandb.Artifact()`의 `type` 필드를 참고하세요:

```python
import wandb

# run 초기화
run = wandb.init(
  entity = "<team_entity>",
  project = "<project>"
  )

# 아티팩트 오브젝트 생성
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
{{% /alert %}}
 

컬렉션을 생성할 때, 미리 정의된 아티팩트 타입 목록 중에서 선택할 수 있습니다. 사용할 수 있는 아티팩트 타입은 컬렉션이 속한 레지스트리에 따라 다릅니다.

아티팩트를 컬렉션에 연결하거나 새로운 컬렉션을 만들기 전에, [해당 컬렉션이 허용하는 아티팩트 타입을 확인하세요]({{< relref path="#check-the-types-of-artifact-that-a-collection-accepts" lang="ko" >}}).

### 컬렉션이 허용하는 아티팩트 타입 확인하기

특정 컬렉션에 연결하기 전에, 그 컬렉션이 허용하는 아티팩트 타입을 미리 확인하세요. 이 정보는 W&B Python SDK를 사용해 프로그래밍적으로 확인하거나 W&B App에서 직접 인터랙티브하게 알 수 있습니다.

{{% alert %}}
지원하지 않는 아티팩트 타입을 컬렉션에 연결하려고 하면 오류 메시지가 나타납니다.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
홈페이지의 레지스트리 카드나, 각 레지스트리 설정 페이지에서 허용되는 아티팩트 타입을 확인할 수 있습니다.

두 방법 모두 먼저 W&B Registry App으로 이동하세요.

Registry App의 메인 페이지에서, 등록된 각 레지스트리 카드까지 스크롤을 내리면 해당 레지스트리 카드에 회색의 가로 타원 형태로 허용되는 아티팩트 타입들이 나열되어 있습니다.

{{< img src="/images/registry/artifact_types_model_card.png" alt="Artifact types selection" >}}

위 이미지는 Registry App의 홈 화면에 여러 레지스트리 카드가 있는 상황을 보여줍니다. **Model** 레지스트리 카드에서 **model** 및 **model-new** 두 가지 아티팩트 타입이 명시되어 있음을 볼 수 있습니다.

레지스트리의 설정 페이지에서 아티팩트 타입을 보려면:

1. 설정을 확인하려는 레지스트리 카드를 클릭합니다.
2. 오른쪽 상단의 톱니바퀴 아이콘을 클릭합니다.
3. **Accepted artifact types** 항목까지 스크롤합니다.  
  {{% /tab %}}
  {{% tab header="Python SDK (Beta)" %}}
W&B Python SDK를 이용하여 프로그래밍적으로 레지스트리가 허용하는 아티팩트 타입을 조회할 수 있습니다:

```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

{{% alert %}}
아래 코드조각에서는 run을 초기화하지 않습니다. W&B API를 단순히 쿼리할 뿐이고, 실험이나 아티팩트 등을 추적하는 것이 아니므로 run을 만들 필요가 없습니다.
{{% /alert %}}  
  {{% /tab %}}
{{< /tabpane >}}

어떤 컬렉션이 어떤 아티팩트 타입을 허용하는지 파악했다면, [컬렉션을 생성]({{< relref path="#create-a-collection" lang="ko" >}})할 수 있습니다.


## 컬렉션 생성하기

인터랙티브하게 또는 프로그래밍적으로 레지스트리 내에서 컬렉션을 만들 수 있습니다. 컬렉션을 만든 이후에는 그 컬렉션이 허용하는 아티팩트 타입을 변경할 수 없습니다.

### 프로그래밍적으로 컬렉션 생성하기

`wandb.init.link_artifact()` 메소드를 사용해 아티팩트를 컬렉션에 연결합니다. `target_path` 필드에 컬렉션과 레지스트리를 아래와 같은 경로 형태로 명시하세요:

```python
f"wandb-registry-{registry_name}/{collection_name}"
```

여기서 `registry_name`은 레지스트리의 이름, `collection_name`은 컬렉션의 이름입니다. 반드시 레지스트리 이름 앞에 `wandb-registry-` 접두사를 붙여야 합니다.

{{% alert %}}
존재하지 않는 컬렉션에 아티팩트를 연결하려고 하면, W&B에서 자동으로 컬렉션을 생성합니다. 이미 존재하는 컬렉션을 지정할 경우, 해당 컬렉션에 아티팩트가 연결됩니다.
{{% /alert %}}

아래 코드조각은 프로그래밍적으로 컬렉션을 생성하는 전체 과정을 보여줍니다. `<>`로 둘러싸인 값은 본인의 값으로 변경해야 합니다:

```python
import wandb

# run 초기화
run = wandb.init(entity = "<team_entity>", project = "<project>")

# 아티팩트 오브젝트 생성
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
  )

registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# 아티팩트를 컬렉션에 연결
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```

### 인터랙티브하게 컬렉션 생성하기

아래 단계에 따라 W&B Registry App UI에서 레지스트리 내에 컬렉션을 생성할 수 있습니다:

1. W&B App UI에서 **Registry** App으로 이동합니다.
2. 레지스트리를 선택합니다.
3. 오른쪽 상단의 **Create collection** 버튼을 클릭합니다.
4. **Name** 필드에 컬렉션의 이름을 입력합니다. 
5. **Type** 드롭다운에서 타입을 선택합니다. 또는, 레지스트리가 커스텀 아티팩트 타입을 허용한다면 이 컬렉션이 허용할 아티팩트 타입을 하나 이상 직접 입력할 수 있습니다.
6. 원할 경우 **Description** 필드에 컬렉션에 대한 설명을 추가합니다.
7. 원할 경우 **Tags** 필드에 하나 이상의 태그를 입력합니다.
8. **Link version**을 클릭합니다.
9. **Project** 드롭다운에서 아티팩트가 저장된 프로젝트를 선택합니다.
10. **Artifact** 컬렉션 드롭다운에서 아티팩트를 선택합니다.
11. **Version** 드롭다운에서 컬렉션에 연결할 아티팩트 버전을 선택합니다.
12. **Create collection** 버튼을 클릭합니다.

{{< img src="/images/registry/create_collection.gif" alt="Create a new collection" >}}