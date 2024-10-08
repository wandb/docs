---
title: Create a collection
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

컬렉션을 레지스트리 내에 만들어 아티팩트를 정리하세요. *컬렉션*은 레지스트리에서 연결된 아티팩트 버전들의 집합입니다. 각 컬렉션은 개별적인 작업이나 유스 케이스를 나타내며, 해당 작업과 관련된 아티팩트 버전들의 정리된 선택을 위한 컨테이너 역할을 합니다.

:::tip
W&B Model Registry에 익숙하다면 "registered models"에 대해 알고 있을 것입니다. W&B Registry에서는 registered models가 "collections"로 이름이 변경됩니다. [모델 레지스트리에서 registered model을 생성하는 방법](../model_registry/create-registered-model.md)은 W&B Registry에서 컬렉션을 생성하는 것과 거의 동일합니다. 주요 차이점은 컬렉션은 registered models처럼 엔티티에 속하지 않는다는 점입니다.
:::

## Collection 종류

컬렉션을 생성할 때, 그 컬렉션에 연결할 수 있는 아티팩트의 종류를 선택해야 합니다. 각 컬렉션은 한 가지 유형의 아티팩트를 수용하며, 하나의 유형만 수용할 수 있습니다. 컬렉션이 가질 수 있는 아티팩트 유형은 해당 레지스트리에 대해 정의된 허용된 유형들에 의해 결정됩니다. 레지스트리 설정에서 허용되는 유형을 구성할 수 있습니다.

컬렉션에 연결 가능한 아티팩트의 유형을 제한하는 것은 아티팩트 유형이 혼합되지 않도록 하기 위함입니다. 예를 들어, 모델 아티팩트가 데이터셋 레지스트리에 연결되지 않도록 하는 것입니다.

:::tip
아티팩트를 생성할 때 그 아티팩트의 유형을 지정합니다. `wandb.Artifact()`의 `type` 필드를 주의하세요:

```python
import wandb

# Run을 초기화합니다.
run = wandb.init(entity="<team_entity>", project="<project>")

# 아티팩트 오브젝트를 생성합니다.
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
:::

예를 들어, "dataset" 아티팩트 유형을 수용하는 컬렉션을 생성한다고 가정해 봅시다. 이는 "dataset" 유형의 아티팩트만 이 컬렉션에 연결할 수 있음을 의미합니다.

### 컬렉션이 수용하는 아티팩트 유형 확인하기

컬렉션에 연결하기 전에, 그 컬렉션이 수용하는 아티팩트 유형을 확인하세요:

<Tabs
  defaultValue="ui"
  values={[
    {label: 'W&B App', value: 'ui'},
    {label: 'Python SDK (Beta)', value: 'programmatically'},
  ]}>
  <TabItem value="ui">

홈페이지의 레지스트리 카드나 레지스트리의 설정 페이지에서 컬렉션이 수용하는 아티팩트 유형을 확인하세요. 두 가지 방법 모두 먼저 당신의 W&B Registry App의 https://wandb.ai/registry/ 에 방문해야 합니다.

Registry App의 홈페이지에서, 관심 있는 레지스트리 카드까지 스크롤하여 수용된 아티팩트 유형을 볼 수 있습니다. 레지스트리 카드 내의 회색 가로형 타원은 그 레지스트리가 수용하는 아티팩트 유형을 나열합니다.

예를 들어, 다음 이미지는 Registry App의 홈페이지에 있는 여러 레지스트리 카드를 보여줍니다. **Model** 레지스트리 카드 내에서 두 가지 아티팩트 유형: **model** 및 **model-new** 를 볼 수 있습니다.

![](/images/registry/artifact_types_model_card.png)

레지스트리의 설정 페이지 내에서 수용된 아티팩트 유형을 보려면:

1. 설정을 보고 싶은 레지스트리 카드를 클릭하세요.
2. 오른쪽 상단의 기어 아이콘을 클릭하세요.
3. **Accepted artifact types** 필드로 스크롤하세요.

  </TabItem>
  <TabItem value="programmatically">

W&B Python SDK를 사용하여 레지스트리가 수용하는 아티팩트 유형을 구현적으로 확인하세요:

```python
import wandb

registry_name = "<registryName>"
org_entity = "<org_entity>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}", entity=org_entity).artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

  </TabItem>
</Tabs>

## 컬렉션을 프로그램적으로 생성하기
W&B Python SDK를 사용하여 프로그램적으로 컬렉션을 생성하세요. 컬렉션이 존재하지 않는 경우, 아티팩트를 컬렉션에 연결하려고 하면 W&B가 지정된 이름으로 자동으로 컬렉션을 생성합니다. 대상 경로는 조직의 엔티티, 접두사 "wandb-registry-", 레지스트리의 이름, 그리고 컬렉션의 이름으로 구성됩니다:

```python
f"{org_entity}/wandb-registry-{registry_name}/{collection_name}"
```

다음 코드 조각은 프로그램적으로 컬렉션을 생성하는 방법을 보여줍니다. `<>`로 둘러싸인 다른 값들을 자신의 것으로 대체하세요:

```python
import wandb

# Run을 초기화합니다.
run = wandb.init(entity="<team_entity>", project="<project>")

# 아티팩트 오브젝트를 생성합니다.
artifact = wandb.Artifact(name="<artifact_name>", type="<artifact_type>")

org_entity = "<organization_entity>"
registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"{org_entity}/wandb-registry-{registry_name}/{collection_name}"

# 아티팩트를 컬렉션에 연결합니다.
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```

## 컬렉션을 인터랙티브하게 생성하기

W&B Registry App UI를 사용하여 레지스트리 내에서 컬렉션을 생성하는 방법은 다음과 같습니다:

1. W&B App UI의 **Registry** App으로 이동합니다.
2. 레지스트리를 선택합니다.
3. 오른쪽 상단 모서리의 **Create collection** 버튼을 클릭하세요.
4. **Name** 필드에 컬렉션의 이름을 입력하세요.
5. **Type** 드롭다운에서 유형을 선택하세요. 또는, 레지스트리가 사용자 정의 아티팩트 유형을 허용하는 경우, 이 컬렉션이 수용할 하나 이상의 아티팩트 유형을 제공하세요.
:::info
아티팩트 유형은 설정에서 추가 및 저장된 후에는 레지스트리에서 제거할 수 없습니다.
:::
5. 선택적으로 **Description** 필드에 컬렉션에 대한 설명을 입력하세요.
6. 선택적으로 **Tags** 필드에 하나 이상의 태그를 추가하세요.
7. **Link version**을 클릭하세요.
8. **Project** 드롭다운에서 아티팩트가 저장된 프로젝트를 선택하세요.
9. **Artifact** 컬렉션 드롭다운에서 아티팩트를 선택하세요.
10. **Version** 드롭다운에서 컬렉션에 연결할 아티팩트 버전을 선택하세요.
11. **Create collection** 버튼을 클릭하세요.

![](/images/registry/create_collection.gif)