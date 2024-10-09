---
title: Link an artifact version to a registry
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

아티팩트 버전을 프로그래밍 방식 또는 대화형으로 레지스트리에 연결하세요.

W&B는 레지스트리가 허용하는 아티팩트 유형을 확인할 것을 권장합니다. 각 레지스트리는 해당 레지스트리에 연결할 수 있는 아티팩트의 유형을 제어합니다.

:::info
"유형"이라는 용어는 아티팩트 오브젝트 유형을 나타냅니다. 아티팩트 오브젝트를 생성할 때 ([`wandb.Artifact`](../../ref/python/artifact.md)) 또는 아티팩트를 로그할 때 ([`wandb.run.log_artifact`](../../ref/python/run.md#log_artifact)), `type` 파라미터에 대한 유형을 지정합니다.
:::

예를 들어, 기본적으로 모델 레지스트리는 "model" 유형을 가진 아티팩트 오브젝트만 허용합니다. 데이터셋 아티팩트 유형 오브젝트를 모델 레지스트리에 연결하려고 하면 W&B는 이를 허용하지 않습니다.

:::info
아티팩트를 레지스트리에 연결할 때, 이는 해당 아티팩트를 해당 레지스트리에 "게시"합니다. 해당 레지스트리에 엑세스 권한이 있는 모든 사용자는 아티팩트를 컬렉션에 연결할 때 연결된 아티팩트 버전에 엑세스할 수 있습니다.

다시 말해, 아티팩트를 레지스트리 컬렉션에 연결하면 해당 아티팩트 버전이 개인 프로젝트 수준 범위에서 공유 조직 수준 범위로 전환됩니다.
:::

## 아티팩트 버전 연결 방법

유스 케이스에 따라 아래 탭에 설명된 지침을 따라 아티팩트 버전을 연결하세요.

<Tabs
  defaultValue="python_sdk"
  values={[
    {label: 'Python SDK', value: 'python_sdk'},
    {label: 'Registry App', value: 'registry_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>

  <TabItem value="python_sdk">

`link_artifact`를 사용하여 기존 레지스트리 내의 컬렉션에 아티팩트 버전을 프로그래밍 방식으로 연결하세요. 컬렉션이 속한 레지스트리가 이미 존재하는지 확인하세요.

`target_path` 파라미터를 사용하여 아티팩트 버전을 연결할 컬렉션과 레지스트리를 지정합니다. 대상 경로는 다음으로 구성됩니다:

```text
{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

기존 레지스트리 내의 컬렉션에 아티팩트 버전을 연결하려면 아래 코드조각을 복사하여 붙여넣으세요. `<>`로 묶인 값을 여러분의 값으로 교체하세요:

```python
import wandb

TEAM_ENTITY_NAME = "<team_entity_name>"
ORG_ENTITY_NAME = "<org_entity_name>"

REGISTRY_NAME = "<registry_name>"  
COLLECTION_NAME = "<collection_name>"

run = wandb.init(
        entity=TEAM_ENTITY_NAME, project="<project_name>")

artifact = wandb.Artifact(name="<artifact_name>", type="<collection_type>")
artifact.add_file(local_path="<local_path_to_artifact>")

target_path=f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
run.link_artifact(artifact = artifact, target_path = target_path)
```

**Models** 레지스트리 또는 **Dataset** 레지스트리에 아티팩트 버전을 연결하려면, 아티팩트 유형을 각각 `"model"` 또는 `"dataset"`으로 설정하세요.

예를 들어, 다음 코드조각은 모델 레지스트리 내의 "Example ML Task"라는 컬렉션에 "my_model.txt"라는 모델 아티팩트를 로그하는 것을 시뮬레이트합니다:

```python
import wandb

TEAM_ENTITY_NAME = "<team_entity_name>"
ORG_ENTITY_NAME = "<org_entity_name>"

REGISTRY_NAME = "model" 
COLLECTION_NAME = "Example ML Task"

COLLECTION_TYPE = "model"


with wandb.init(entity=TEAM_ENTITY_NAME, project="link_quickstart") as run:
  with open("my_model.txt", "w") as f:
    f.write("simulated model file")

  logged_artifact = run.log_artifact("./my_model.txt", "artifact-name", type=COLLECTION_TYPE)
  run.link_artifact(
    artifact=logged_artifact,
    target_path=f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
  )
```

  </TabItem>
  <TabItem value="registry_ui">

1. Registry App으로 이동하세요.
![](/images/registry/navigate_to_registry_app.png)
2. 아티팩트 버전을 연결하고자 하는 컬렉션의 이름 옆에 마우스를 올리세요.
3. **View details** 옆에 있는 세 개의 수평 점이 있는 아이콘을 선택하세요.
4. 드롭다운에서 **Link new version**을 선택하세요.
5. 나타나는 사이드바에서 **Team** 드롭다운에서 팀 이름을 선택하세요.
6. **Project** 드롭다운에서 아티팩트를 포함하는 프로젝트 이름을 선택하세요.
7. **Artifact** 드롭다운에서 아티팩트 이름을 선택하세요.
8. **Version** 드롭다운에서 컬렉션에 연결하려는 아티팩트 버전을 선택하세요.

  </TabItem>
  <TabItem value="artifacts_ui">

1. 프로젝트의 아티팩트 브라우저로 이동하세요: `https://wandb.ai/<entity>/<project>/artifacts`
2. 왼쪽 사이드바에서 Artifacts 아이콘을 선택하세요.
3. 레지스트리에 연결하고자 하는 아티팩트 버전을 클릭하세요.
4. **Version overview** 섹션 내에서 **Link to registry** 버튼을 클릭하세요.
5. 화면 오른쪽에 나타나는 모달에서 **Select a register model** 메뉴 드롭다운에서 아티팩트를 선택하세요.
6. **Next step**을 클릭하세요.
7. (선택 사항) **Aliases** 드롭다운에서 에일리어스를 선택하세요.
8. **Link to registry**를 클릭하세요.

  </TabItem>
</Tabs>

## 문제 해결

아티팩트를 연결할 수 없을 경우, 확인해야 할 일반적인 항목은 다음과 같습니다.

### 개인 계정에서 아티팩트 로그

개인 엔티티로 W&B에 로그된 아티팩트는 레지스트리에 연결할 수 없습니다. 조직 내 팀 엔티티를 사용하여 아티팩트를 로그했는지 확인하세요. 조직의 팀 내에 로그된 아티팩트만 조직의 레지스트리에 연결할 수 있습니다.

:::tip
아티팩트를 레지스트리에 연결하려면 팀 엔티티로 아티팩트를 로그했는지 확인하세요.
:::

#### 팀 엔티티 찾기

W&B는 팀의 이름을 팀의 엔티티로 사용합니다. 예를 들어 팀이 "team-awesome"이라면, 팀 엔티티는 `team-awesome`입니다.

팀 이름을 확인하려면:

1. 팀의 W&B 프로필 페이지로 이동하세요.
2. 사이트의 URL을 복사하세요. 이는 `https://wandb.ai/<team>` 형식을 가지고 있으며, `<team>`은 팀의 이름이자 팀의 엔티티입니다.

#### 팀 엔티티에서 로그
1. [`wandb.init()`](/ref/python/init)으로 run을 초기화할 때 엔티티로 팀을 지정하세요. run을 초기화할 때 `entity`를 명시하지 않으면, 기본 엔티티가 사용되며, 이는 팀 엔티티가 아닐 수도 있습니다.
  ```python
  import wandb

  run = wandb.init(
    entity='<team_entity_name>', 
    project='<project_name>'
    )
  ```
2. run.log_artifact로 또는 Artifact 오브젝트를 생성한 후 파일을 추가하여 run에 아티팩트를 로그합니다:

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<collection_type>")
    run.log_artifact(artifact)
    ```
    아티팩트를 로그하는 방법에 대한 자세한 정보는 [Construct artifacts](../artifacts/construct-an-artifact.md)를 참조하세요.
3. 아티팩트가 개인 엔티티에 로그된 경우, 이를 조직 내 엔티티로 다시 로그해야 합니다.

### 팀 이름과 충돌이 있는 조직 이름

W&B는 기존 엔티티와의 이름 충돌을 피하기 위해 조직 이름에 고유한 해시를 추가합니다. 이름과 고유 해시의 조합은 조직 식별자, 즉 `ORG_ENTITY_NAME`으로 알려져 있습니다.

예를 들어, 조직 이름이 "reviewco"이고 팀 이름도 "reviewco"인 경우, W&B는 `ORG_ENTITY_NAME`이 `reviewco_XYZ123456`로 명명되도록 이름에 해시를 추가합니다.

:::tip
Python SDK로 레지스트리에 연결할 때, 항상 `target_path`에 `ORG_ENTITY_NAME` 형식을 사용하세요.
:::

예를 들어, 타겟 경로는 `reviewco_XYZ123456/wandb-registry-model/my-collection`과 같은 형태일 수 있습니다.

### W&B 앱 UI에서 레지스트리 경로 확인

UI에서 레지스트리 경로를 확인하는 두 가지 방법이 있습니다: 빈 컬렉션을 생성하고 컬렉션 세부 정보를 보거나 컬렉션의 홈 페이지에서 자동 생성된 코드를 복사하여 붙여넣습니다.

#### 자동 생성된 코드 복사 및 붙여넣기

1. Registry 앱으로 이동하세요: https://wandb.ai/registry/.
2. 아티팩트를 연결할 레지스트리를 클릭하세요.
3. 페이지 상단에 자동 생성된 코드 블록을 볼 수 있습니다.
4. 이를 코드에 복사하여 붙여넣고, 경로의 마지막 부분을 컬렉션 이름으로 교체하세요.

![](/images/registry/get_autogenerated_code.gif)

#### 빈 컬렉션 생성

1. Registry 앱으로 이동하세요: https://wandb.ai/registry/.
2. 아티팩트를 연결할 레지스트리를 클릭하세요.
3. 빈 컬렉션을 클릭하세요. 빈 컬렉션이 존재하지 않으면 새 컬렉션을 생성하세요.
4. 나타나는 코드조각에서 `run.link_artifact()` 내의 `target_path` 필드를 식별하세요.
5. (선택 사항) 컬렉션을 삭제하세요.

![](/images/registry/check_empty_collection.gif)

예를 들어, 위에서 설명한 단계를 완료한 후, `target_path` 파라미터가 포함된 코드 블록을 찾을 수 있습니다:

```python
target_path = 
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

이를 구성 요소로 나누면, 아티팩트를 프로그래밍 방식으로 연결할 경로를 생성하는 데 필요한 것을 확인할 수 있습니다:

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

:::note
임시 컬렉션에서 컬렉션 이름을 실제 연결하려는 컬렉션의 이름으로 대체해야 합니다.
:::