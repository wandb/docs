---
title: Collection 생성
menu:
  default:
    identifier: ko-guides-core-registry-create_collection
    parent: registry
weight: 4
---

*collection*은 registry 내에서 서로 연결된 artifact 버전들의 집합입니다. 각 collection은 서로 다른 작업(Task)이나 유스 케이스를 나타냅니다.

예를 들어, 코어 Dataset registry 내에 여러 개의 collection이 있을 수 있습니다. 각 collection은 MNIST, CIFAR-10, ImageNet 등 서로 다른 데이터셋을 포함합니다.

또 다른 예시로, "chatbot"이라는 registry가 있다면, 모델 artifact collection, 데이터셋 artifact collection, 파인튜닝된 모델 artifact collection 등 여러 collection을 가질 수 있습니다.

registry와 각 collection을 어떻게 구성할지는 전적으로 여러분에게 달려 있습니다.

{{% alert %}}
W&B Model Registry에 익숙하다면 registered models에 대해 알고 계실 수 있습니다. Model Registry에서의 registered models는 이제 W&B Registry에서 collection(collections)이라고 불립니다.
{{% /alert %}}

## Collection types

각 collection은 한 가지, 오직 *하나의* artifact 타입만을 허용합니다. 지정한 타입은 여러분과 소속 조직의 다른 구성원이 해당 collection에 연결할 수 있는 artifact를 제한합니다.

{{% alert %}}
artifact 타입은 프로그래밍 언어, 예를 들어 Python의 데이터 타입과 비슷하다고 생각할 수 있습니다. 이 비유에서 collection은 문자열, 정수, 또는 실수를 저장할 수 있지만, 여러 데이터 타입을 섞어서 저장할 수는 없습니다.
{{% /alert %}}

예를 들어, "dataset" artifact 타입을 허용하는 collection을 만든다고 가정해 봅시다. 이 경우 해당 collection에는 "dataset" 타입의 향후 artifact 버전들만 연결할 수 있습니다. 마찬가지로, "model" 타입만 허용하는 collection에는 "model" 타입의 artifact만 연결할 수 있습니다.

{{% alert %}}
artifact 오브젝트를 생성할 때 artifact의 타입을 지정합니다. `wandb.Artifact()`의 `type` 필드를 참고하세요:

```python
import wandb

# run 초기화
run = wandb.init(
  entity = "<team_entity>",
  project = "<project>"
  )

# artifact 오브젝트 생성
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
{{% /alert %}}
 

collection을 생성할 때, 미리 정의된 artifact 타입 목록 중에서 선택할 수 있습니다. 사용할 수 있는 artifact 타입은 collection이 속한 registry에 따라 다릅니다.

artifact를 collection에 연결하거나 새로운 collection을 만들기 전에, [해당 collection이 허용하는 artifact 타입을 확인하세요]({{< relref path="#check-the-types-of-artifact-that-a-collection-accepts" lang="ko" >}}).

### collection이 허용하는 artifact 타입 확인하기

특정 collection에 연결하기 전에, 그 collection이 허용하는 artifact 타입을 미리 확인하세요. 이 정보는 W&B Python SDK를 사용해 프로그래밍적으로 확인하거나 W&B App에서 직접 인터랙티브하게 알 수 있습니다.

{{% alert %}}
지원하지 않는 artifact 타입을 collection에 연결하려고 하면 오류 메시지가 나타납니다.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
홈페이지의 registry 카드나, 각 registry 설정 페이지에서 허용되는 artifact 타입을 확인할 수 있습니다.

두 방법 모두 먼저 W&B Registry App으로 이동하세요.

Registry App의 메인 페이지에서, 등록된 각 registry 카드까지 스크롤을 내리면 해당 registry 카드에 회색의 가로 타원 형태로 허용되는 artifact 타입들이 나열되어 있습니다.

{{< img src="/images/registry/artifact_types_model_card.png" alt="Artifact types selection" >}}

위 이미지는 Registry App의 홈 화면에 여러 registry 카드가 있는 상황을 보여줍니다. **Model** registry 카드에서 **model** 및 **model-new** 두 가지 artifact 타입이 명시되어 있음을 볼 수 있습니다.

registry의 설정 페이지에서 artifact 타입을 보려면:

1. 설정을 확인하려는 registry 카드를 클릭합니다.
2. 오른쪽 상단의 톱니바퀴 아이콘을 클릭합니다.
3. **Accepted artifact types** 항목까지 스크롤합니다.  
  {{% /tab %}}
  {{% tab header="Python SDK (Beta)" %}}
W&B Python SDK를 이용하여 프로그래밍적으로 registry가 허용하는 artifact 타입을 조회할 수 있습니다:

```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

{{% alert %}}
아래 코드조각에서는 run을 초기화하지 않습니다. W&B API를 단순히 쿼리할 뿐이고, 실험이나 artifact 등을 추적하는 것이 아니므로 run을 만들 필요가 없습니다.
{{% /alert %}}  
  {{% /tab %}}
{{< /tabpane >}}

어떤 collection이 어떤 artifact 타입을 허용하는지 파악했다면, [collection을 생성]({{< relref path="#create-a-collection" lang="ko" >}})할 수 있습니다.


## collection 생성하기

인터랙티브하게 또는 프로그래밍적으로 registry 내에서 collection을 만들 수 있습니다. collection을 만든 이후에는 그 collection이 허용하는 artifact 타입을 변경할 수 없습니다.

### 프로그래밍적으로 collection 생성하기

`wandb.init.link_artifact()` 메소드를 사용해 artifact를 collection에 연결합니다. `target_path` 필드에 collection과 registry를 아래와 같은 경로 형태로 명시하세요:

```python
f"wandb-registry-{registry_name}/{collection_name}"
```

여기서 `registry_name`은 registry의 이름, `collection_name`은 collection의 이름입니다. 반드시 registry 이름 앞에 `wandb-registry-` 접두사를 붙여야 합니다.

{{% alert %}}
존재하지 않는 collection에 artifact를 연결하려고 하면, W&B에서 자동으로 collection을 생성합니다. 이미 존재하는 collection을 지정할 경우, 해당 collection에 artifact가 연결됩니다.
{{% /alert %}}

아래 코드조각은 프로그래밍적으로 collection을 생성하는 전체 과정을 보여줍니다. `<>`로 둘러싸인 값은 본인의 값으로 변경해야 합니다:

```python
import wandb

# run 초기화
run = wandb.init(entity = "<team_entity>", project = "<project>")

# artifact 오브젝트 생성
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
  )

registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# artifact를 collection에 연결
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```

### 인터랙티브하게 collection 생성하기

아래 단계에 따라 W&B Registry App UI에서 registry 내에 collection을 생성할 수 있습니다:

1. W&B App UI에서 **Registry** App으로 이동합니다.
2. registry를 선택합니다.
3. 오른쪽 상단의 **Create collection** 버튼을 클릭합니다.
4. **Name** 필드에 collection의 이름을 입력합니다. 
5. **Type** 드롭다운에서 타입을 선택합니다. 또는, registry가 커스텀 artifact 타입을 허용한다면 이 collection이 허용할 artifact 타입을 하나 이상 직접 입력할 수 있습니다.
6. 원할 경우 **Description** 필드에 collection에 대한 설명을 추가합니다.
7. 원할 경우 **Tags** 필드에 하나 이상의 tag를 입력합니다.
8. **Link version**을 클릭합니다.
9. **Project** 드롭다운에서 artifact가 저장된 프로젝트를 선택합니다.
10. **Artifact** collection 드롭다운에서 artifact를 선택합니다.
11. **Version** 드롭다운에서 collection에 연결할 artifact 버전을 선택합니다.
12. **Create collection** 버튼을 클릭합니다.

{{< img src="/images/registry/create_collection.gif" alt="Create a new collection" >}}