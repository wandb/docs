---
title: 아티팩트 버전을 레지스트리에 연결하기
menu:
  default:
    identifier: ko-guides-core-registry-link_version
    parent: registry
weight: 5
---

아티팩트 버전을 컬렉션에 연결(link)하여 조직 내 다른 멤버들이 사용할 수 있도록 하세요.

아티팩트를 레지스트리에 연결하면, 해당 아티팩트가 레지스트리에 "발행"됩니다. 이 레지스트리에 엑세스 권한이 있는 모든 사용자는 컬렉션에 연결된 아티팩트 버전에 엑세스할 수 있습니다.

즉, 아티팩트를 레지스트리 컬렉션에 연결하면, 해당 아티팩트 버전은 개인적인 프로젝트 레벨 범위에서 조직 전체에서 공유되는 범위로 이동합니다.

{{% alert %}}
"타입(type)"이라는 용어는 아티팩트 오브젝트의 타입을 의미합니다. 아티팩트 오브젝트([`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}}))를 생성하거나, 아티팩트를 로그([`wandb.init.log_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ko" >}}))할 때, `type` 파라미터로 타입을 지정합니다.
{{% /alert %}}

## 아티팩트를 컬렉션에 연결하기

아티팩트 버전을 컬렉션에 인터랙티브하게 혹은 프로그래밍적으로 연결할 수 있습니다.

{{% alert %}}
아티팩트를 레지스트리에 연결하기 전에, 해당 컬렉션에서 허용하는 아티팩트 타입을 꼭 확인하세요. 컬렉션 타입에 대한 자세한 내용은 [컬렉션 생성]({{< relref path="./create_collection.md" lang="ko" >}}) 내 "컬렉션 타입"을 참고하세요.
{{% /alert %}}

유스 케이스에 따라 아래 탭의 안내를 따라 아티팩트 버전을 연결하세요.

{{% alert %}}
아티팩트 버전이 메트릭을 로그하는 경우(예: `run.log_artifact()` 사용 시), 해당 버전의 상세 페이지에서 메트릭을 확인할 수 있고, 아티팩트 페이지에서 여러 버전의 메트릭을 비교할 수 있습니다. 자세한 내용은 [레지스트리에서 연결된 아티팩트 보기]({{< relref path="#view-linked-artifacts-in-a-registry" lang="ko" >}})를 참고하세요.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
{{% alert %}}
[버전 연결 데모 영상 보기](https://www.youtube.com/watch?v=2i_n1ExgO0A) (8분)
{{% /alert %}}

프로그래밍적으로는 [`wandb.init.Run.link_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#link_artifact" lang="ko" >}})를 사용해 아티팩트 버전을 컬렉션에 연결할 수 있습니다.

{{% alert %}}
아티팩트를 컬렉션에 연결하기 전에, 해당 컬렉션이 속한 레지스트리가 이미 존재하는지 확인하세요. 레지스트리 존재 여부는 W&B App UI에서 Registry App으로 접속하여 레지스트리 이름을 검색하세요.
{{% /alert %}}

`target_path` 파라미터를 사용해 연결할 컬렉션과 레지스트리를 지정할 수 있습니다. target_path는 "wandb-registry" 접두사에 레지스트리 이름, 컬렉션 이름을 슬래시(`/`)로 구분하여 붙입니다:

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

아래 코드조각을 복사해 기존 레지스트리 내 컬렉션에 아티팩트 버전을 연결하세요. `<>`로 표시된 값은 본인 정보로 바꾸세요.

```python
import wandb

# run 초기화
run = wandb.init(
  entity = "<team_entity>",
  project = "<project_name>"
)

# 아티팩트 오브젝트 생성
# type 파라미터는 아티팩트 오브젝트와 컬렉션 타입을 지정합니다.
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# 파일을 아티팩트 오브젝트에 추가
# 파일의 로컬 경로를 지정
artifact.add_file(local_path = "<local_path_to_artifact>")

# 연결할 컬렉션과 레지스트리 지정
REGISTRY_NAME = "<registry_name>"  
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# 아티팩트를 컬렉션에 연결
run.link_artifact(artifact = artifact, target_path = target_path)
```
{{% alert %}}
Model registry 또는 Dataset registry에 아티팩트 버전을 연결하려면, 아티팩트 타입을 각각 `"model"` 또는 `"dataset"`으로 설정하세요.
{{% /alert %}}

  {{% /tab %}}
  {{% tab header="Registry App" %}}
1. Registry App으로 이동하세요.
    {{< img src="/images/registry/navigate_to_registry_app.png" alt="Registry App navigation" >}}
2. 연결할 아티팩트 버전의 컬렉션 이름 옆에 마우스를 올려놓으세요.
3. **View details** 오른쪽의 세 점 메뉴(수평 점 3개)를 클릭하세요.
4. 드롭다운에서 **Link new version**을 선택하세요.
5. 사이드바에서 **Team** 드롭다운에서 팀 이름을 선택하세요.
5. **Project** 드롭다운에서는 아티팩트가 속한 프로젝트를 선택하세요.
6. **Artifact** 드롭다운에서 아티팩트 이름을 선택하세요.
7. **Version** 드롭다운에서 컬렉션에 연결할 아티팩트 버전을 선택하세요.

  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App에서 프로젝트의 artifact browser로 이동하세요: `https://wandb.ai/<entity>/<project>/artifacts`
2. 왼쪽 사이드바에서 Artifacts 아이콘을 클릭하세요.
3. 레지스트리에 연결할 아티팩트 버전을 클릭하세요.
4. **Version overview** 섹션에서 **Link to registry** 버튼을 클릭하세요.
5. 화면 오른쪽의 모달에서 **Select a register model** 메뉴에서 아티팩트를 선택하세요.
6. **Next step**을 클릭하세요.
7. (선택) **Aliases** 드롭다운에서 에일리어스를 선택할 수 있습니다.
8. **Link to registry**를 클릭하세요.

  {{% /tab %}}
{{< /tabpane >}}

레지스트리 App에서 연결된 아티팩트의 메타데이터, 버전 데이터, 사용 정보, 계보 등 다양한 정보를 확인할 수 있습니다.

## 레지스트리에서 연결된 아티팩트 보기

레지스트리 App에서 연결된 아티팩트의 메타데이터, 계보, 사용 정보 등 다양한 정보를 확인하세요.

1. Registry App에 접속하세요.
2. 아티팩트를 연결한 레지스트리 이름을 선택하세요.
3. 컬렉션 이름을 선택하세요.
4. 컬렉션의 아티팩트들이 메트릭을 로그했다면, **Show metrics**를 클릭해 버전별 메트릭을 비교할 수 있습니다.
4. 아티팩트 버전 목록에서 엑세스할 버전을 선택하세요. 버전 번호는 처음 연결 시점부터 `v0`같이 순차적으로 할당됩니다.
5. 아티팩트 버전의 상세정보를 볼 때는 버전을 클릭하세요. 이 페이지 상단 탭에서 해당 버전의 메타데이터(로그된 메트릭 포함), 계보, 사용 정보를 확인할 수 있습니다.

**Version** 탭에서 **Full Name** 필드를 확인하세요. 연결된 아티팩트의 전체 이름은 레지스트리, 컬렉션 이름, 아티팩트 버전의 에일리어스 혹은 인덱스로 이루어집니다.

```text title="Full name of a linked artifact"
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INTEGER}
```

연결된 아티팩트의 전체 이름은 프로그래밍적으로 해당 버전에 엑세스할 때 필요합니다.

## 문제 해결

아티팩트 연결이 제대로 되지 않을 때 확인해야 할 일반적인 체크리스트입니다.

### 개인 계정에서 아티팩트 로그 시

개인 Entity로 W&B에 로그된 아티팩트는 레지스트리에 연결할 수 없습니다. 반드시 조직 내 팀 Entity로 아티팩트 로그를 해야 하며, 조직의 팀 내에서 로그된 아티팩트만 해당 조직의 레지스트리에 연결할 수 있습니다.

{{% alert title="" %}}
아티팩트를 레지스트리에 연결하려면, 반드시 팀 Entity로 아티팩트를 로그해야 합니다.
{{% /alert %}}

#### 팀 엔터티 확인하기

W&B에서는 팀의 이름이 곧 팀 엔터티(Entity)로 사용됩니다. 예를 들어, 팀 이름이 **team-awesome**이라면, 팀 엔터티는 `team-awesome`입니다.

팀 이름을 확인하려면:

1. 팀의 W&B 프로필 페이지로 이동하세요.
2. 사이트 URL을 복사하세요. `https://wandb.ai/<team>` 형태입니다. 여기서 `<team>`이 곧 팀 이름이자 팀 엔터티입니다.

#### 팀 엔터티로 로그하기

1. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})로 run을 초기화할 때, entity 파라미터에 팀을 지정하세요. entity를 지정하지 않으면, 기본 Entity(개인 entity)가 사용될 수 있습니다.

  ```python 
  import wandb   

  run = wandb.init(
    entity='<team_entity>', 
    project='<project_name>'
    )
  ```

2. run에 아티팩트를 로그하려면 run.log_artifact를 사용하거나, Artifact 오브젝트를 생성해 파일을 추가하세요.

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    ```
    아티팩트 로그에 대해서는 [아티팩트 생성하기]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ko" >}})를 참고하세요.
3. 개인 Entity로 아티팩트가 로그된 경우, 조직 내 Entity로 다시 한번 로그해야 합니다.

### W&B App UI에서 레지스트리 경로 확인

레지스트리의 경로를 UI에서 확인하는 방법은 두 가지입니다: 빈 컬렉션을 만들고 상세정보에서 확인하거나, 컬렉션 홈에서 자동 생성된 코드를 복사하는 것입니다.

#### 자동 생성 코드 복사하기

1. https://wandb.ai/registry/에서 Registry app으로 이동하세요.
2. 아티팩트를 연결할 레지스트리를 클릭하세요.
3. 페이지 상단에 자동 생성된 코드 블록이 표시됩니다.
4. 이 코드를 복사해 코드에 붙여넣고, 경로의 마지막 부분을 원하는 컬렉션 이름으로 교체하세요.

{{< img src="/images/registry/get_autogenerated_code.gif" alt="Auto-generated code snippet" >}}

#### 빈 컬렉션 생성하기

1. https://wandb.ai/registry/에서 Registry app으로 이동하세요.
2. 아티팩트를 연결할 레지스트리를 클릭하세요.
4. 빈 컬렉션을 클릭하세요. (빈 컬렉션이 없다면 새로 만드세요.)
5. 나타나는 코드조각에서 `.link_artifact()`의 `target_path` 필드를 찾으세요.
6. (선택) 컬렉션을 삭제하세요.

{{< img src="/images/registry/check_empty_collection.gif" alt="Create an empty collection" >}}

예를 들어, 위 과정을 거치면 아래와 같이 `target_path` 파라미터가 있는 코드 블록을 확인할 수 있습니다.

```python
target_path = 
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

각 구성 요소를 보면, 프로그래밍적으로 아티팩트 연결 경로를 작성할 때 필요한 값을 알 수 있습니다.

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

{{% alert %}}
임시로 만든 컬렉션의 이름을, 실제로 연결하고자 하는 컬렉션 이름으로 꼭 바꿔서 사용하세요.
{{% /alert %}}