---
title: Link an artifact version to a registry
menu:
  default:
    identifier: ko-guides-core-registry-link_version
    parent: registry
weight: 5
---

Artifact 버전들을 컬렉션에 연결하여 조직의 다른 구성원들이 사용할 수 있도록 합니다.

Artifact를 레지스트리에 연결하면 해당 Artifact가 레지스트리에 "게시"됩니다. 해당 레지스트리에 대한 엑세스 권한이 있는 모든 사용자는 컬렉션에서 연결된 Artifact 버전에 엑세스할 수 있습니다.

즉, Artifact를 레지스트리 컬렉션에 연결하면 해당 Artifact 버전이 개인 프로젝트 수준 범위에서 공유 조직 수준 범위로 이동합니다.

{{% alert %}}
"유형"이라는 용어는 Artifact 오브젝트의 유형을 나타냅니다. Artifact 오브젝트([`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ko" >}}))를 생성하거나 Artifact([`wandb.init.log_artifact`]({{< relref path="/ref/python/run.md#log_artifact" lang="ko" >}}))를 기록할 때 `type` 파라미터에 대한 유형을 지정합니다.
{{% /alert %}}

## Artifact를 컬렉션에 연결

Artifact 버전을 대화식으로 또는 프로그래밍 방식으로 컬렉션에 연결합니다.

{{% alert %}}
Artifact를 레지스트리에 연결하기 전에 해당 컬렉션에서 허용하는 Artifact 유형을 확인하십시오. 컬렉션 유형에 대한 자세한 내용은 [컬렉션 생성]({{< relref path="./create_collection.md" lang="ko" >}}) 내의 "컬렉션 유형"을 참조하십시오.
{{% /alert %}}

유스 케이스에 따라 아래 탭에 설명된 지침을 따르십시오.

{{% alert %}}
Artifact 버전이 메트릭을 기록하는 경우(`run.log_artifact()` 사용) 해당 버전의 세부 정보 페이지에서 해당 버전에 대한 메트릭을 보고 Artifact 페이지에서 Artifact 버전 간의 메트릭을 비교할 수 있습니다. [레지스트리에서 연결된 Artifact 보기]({{< relref path="#view-linked-artifacts-in-a-registry" lang="ko" >}})를 참조하십시오.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
{{% alert %}}
[버전 연결을 시연하는 비디오](https://www.youtube.com/watch?v=2i_n1ExgO0A) (8분)를 시청하십시오.
{{% /alert %}}

[`wandb.init.Run.link_artifact()`]({{< relref path="/ref/python/run.md#link_artifact" lang="ko" >}})를 사용하여 Artifact 버전을 프로그래밍 방식으로 컬렉션에 연결합니다.

{{% alert %}}
Artifact를 컬렉션에 연결하기 전에 컬렉션이 속한 레지스트리가 이미 존재하는지 확인하십시오. 레지스트리가 존재하는지 확인하려면 W&B App UI에서 레지스트리 앱으로 이동하여 레지스트리 이름을 검색하십시오.
{{% /alert %}}

`target_path` 파라미터를 사용하여 Artifact 버전을 연결할 컬렉션 및 레지스트리를 지정합니다. 대상 경로는 "wandb-registry" 접두사, 레지스트리 이름 및 컬렉션 이름으로 구성되며 슬래시(/)로 구분됩니다.

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

아래 코드 조각을 복사하여 붙여넣어 기존 레지스트리 내의 컬렉션에 Artifact 버전을 연결합니다. 꺾쇠 괄호로 묶인 값을 자신의 값으로 바꿉니다.

```python
import wandb

# run 초기화
run = wandb.init(
  entity = "<team_entity>",
  project = "<project_name>"
)

# Artifact 오브젝트 생성
# type 파라미터는 Artifact 오브젝트의 유형과 
# 컬렉션 유형을 모두 지정합니다.
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# Artifact 오브젝트에 파일 추가
# 로컬 머신에서 파일의 경로를 지정합니다.
artifact.add_file(local_path = "<local_path_to_artifact>")

# Artifact를 연결할 컬렉션 및 레지스트리 지정
REGISTRY_NAME = "<registry_name>"  
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# Artifact를 컬렉션에 연결
run.link_artifact(artifact = artifact, target_path = target_path)
```
{{% alert %}}
Artifact 버전을 Model registry 또는 Dataset registry에 연결하려면 Artifact 유형을 각각 `"model"` 또는 `"dataset"`으로 설정하십시오.
{{% /alert %}}

  {{% /tab %}}
  {{% tab header="Registry App" %}}
1. 레지스트리 앱으로 이동합니다.
    {{< img src="/images/registry/navigate_to_registry_app.png" alt="" >}}
2. Artifact 버전을 연결할 컬렉션 이름 옆으로 마우스를 가져갑니다.
3. **세부 정보 보기** 옆에 있는 미트볼 메뉴 아이콘(가로 점 3개)을 선택합니다.
4. 드롭다운에서 **새 버전 연결**을 선택합니다.
5. 나타나는 사이드바에서 **팀** 드롭다운에서 팀 이름을 선택합니다.
6. **프로젝트** 드롭다운에서 Artifact가 포함된 프로젝트 이름을 선택합니다.
7. **Artifact** 드롭다운에서 Artifact 이름을 선택합니다.
8. **버전** 드롭다운에서 컬렉션에 연결할 Artifact 버전을 선택합니다.


  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App의 프로젝트 Artifact 브라우저(`https://wandb.ai/<entity>/<project>/artifacts`)로 이동합니다.
2. 왼쪽 사이드바에서 Artifacts 아이콘을 선택합니다.
3. 레지스트리에 연결할 Artifact 버전을 클릭합니다.
4. **버전 개요** 섹션 내에서 **레지스트리에 연결** 버튼을 클릭합니다.
5. 화면 오른쪽에 나타나는 모달에서 **레지스터 모델 선택** 메뉴 드롭다운에서 Artifact를 선택합니다.
6. **다음 단계**를 클릭합니다.
7. (선택 사항) **에일리어스** 드롭다운에서 에일리어스를 선택합니다.
8. **레지스트리에 연결**을 클릭합니다.




  
  {{% /tab %}}
{{< /tabpane >}}

레지스트리 앱에서 연결된 Artifact의 메타데이터, 버전 데이터, 사용량, 계보 정보 등을 봅니다.

## 레지스트리에서 연결된 Artifact 보기

레지스트리 앱에서 메타데이터, 계보 및 사용량 정보와 같은 연결된 Artifact에 대한 정보를 봅니다.

1. 레지스트리 앱으로 이동합니다.
2. Artifact를 연결한 레지스트리 이름을 선택합니다.
3. 컬렉션 이름을 선택합니다.
4. 컬렉션의 Artifact가 메트릭을 기록하는 경우 **메트릭 표시**를 클릭하여 버전 간의 메트릭을 비교합니다.
5. Artifact 버전 목록에서 엑세스할 버전을 선택합니다. 버전 번호는 `v0`부터 시작하여 각 연결된 Artifact 버전에 점진적으로 할당됩니다.
6. Artifact 버전에 대한 세부 정보를 보려면 해당 버전을 클릭합니다. 이 페이지의 탭에서 해당 버전의 메타데이터(기록된 메트릭 포함), 계보 및 사용량 정보를 볼 수 있습니다.

**버전** 탭 내에서 **전체 이름** 필드를 기록해 두십시오. 연결된 Artifact의 전체 이름은 레지스트리, 컬렉션 이름 및 Artifact 버전의 에일리어스 또는 인덱스로 구성됩니다.

```text title="연결된 Artifact의 전체 이름"
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INTEGER}
```

Artifact 버전을 프로그래밍 방식으로 엑세스하려면 연결된 Artifact의 전체 이름이 필요합니다.

## 문제 해결

Artifact를 연결할 수 없는 경우 몇 가지 일반적인 사항을 다시 확인하십시오.

### 개인 계정에서 Artifact 기록

개인 엔터티로 W&B에 기록된 Artifact는 레지스트리에 연결할 수 없습니다. 조직 내에서 팀 엔터티를 사용하여 Artifact를 기록해야 합니다. 조직의 팀 내에서 기록된 Artifact만 조직의 레지스트리에 연결할 수 있습니다.

{{% alert title="" %}}
Artifact를 레지스트리에 연결하려면 팀 엔터티로 Artifact를 기록해야 합니다.
{{% /alert %}}

#### 팀 엔터티 찾기

W&B는 팀 이름을 팀의 엔터티로 사용합니다. 예를 들어 팀 이름이 **team-awesome**인 경우 팀 엔터티는 `team-awesome`입니다.

다음을 통해 팀 이름을 확인할 수 있습니다.

1. 팀의 W&B 프로필 페이지로 이동합니다.
2. 사이트 URL을 복사합니다. URL은 `https://wandb.ai/<team>` 형식입니다. 여기서 `<team>`은 팀 이름과 팀 엔터티입니다.

#### 팀 엔터티에서 기록

1. [`wandb.init()`]({{< relref path="/ref/python/init" lang="ko" >}})로 run을 초기화할 때 팀을 엔터티로 지정합니다. run을 초기화할 때 `entity`를 지정하지 않으면 run은 팀 엔터티일 수도 있고 아닐 수도 있는 기본 엔터티를 사용합니다.
  ```python
  import wandb

  run = wandb.init(
    entity='<team_entity>',
    project='<project_name>'
    )
  ```
2. run.log_artifact를 사용하거나 Artifact 오브젝트를 생성한 다음 파일을 추가하여 Artifact 오브젝트에 Artifact를 기록합니다.

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    ```
    Artifact를 기록하는 방법에 대한 자세한 내용은 [Artifact 생성]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ko" >}})을 참조하십시오.
3. Artifact가 개인 엔터티에 기록된 경우 조직 내의 엔터티에 다시 기록해야 합니다.

### W&B App UI에서 레지스트리 경로 확인

UI에서 레지스트리 경로를 확인하는 방법에는 빈 컬렉션을 만들고 컬렉션 세부 정보를 보거나 컬렉션 홈페이지에서 자동 생성된 코드를 복사하여 붙여넣는 두 가지 방법이 있습니다.

#### 자동 생성된 코드 복사 및 붙여넣기

1. 레지스트리 앱(https://wandb.ai/registry/)으로 이동합니다.
2. Artifact를 연결할 레지스트리를 클릭합니다.
3. 페이지 상단에 자동 생성된 코드 블록이 표시됩니다.
4. 이 코드를 복사하여 코드에 붙여넣고 경로의 마지막 부분을 컬렉션 이름으로 바꿔야 합니다.

{{< img src="/images/registry/get_autogenerated_code.gif" alt="" >}}

#### 빈 컬렉션 만들기

1. 레지스트리 앱(https://wandb.ai/registry/)으로 이동합니다.
2. Artifact를 연결할 레지스트리를 클릭합니다.
3. 빈 컬렉션을 클릭합니다. 빈 컬렉션이 없으면 새 컬렉션을 만듭니다.
4. 나타나는 코드 조각 내에서 `.link_artifact()` 내의 `target_path` 필드를 식별합니다.
5. (선택 사항) 컬렉션을 삭제합니다.

{{< img src="/images/registry/check_empty_collection.gif" alt="" >}}

예를 들어 설명된 단계를 완료한 후 `target_path` 파라미터가 있는 코드 블록을 찾습니다.

```python
target_path = 
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

이를 구성 요소로 나누면 Artifact를 프로그래밍 방식으로 연결하는 데 사용할 경로를 만드는 데 필요한 것을 알 수 있습니다.

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

{{% alert %}}
임시 컬렉션의 컬렉션 이름을 Artifact를 연결할 컬렉션 이름으로 바꿔야 합니다.
{{% /alert %}}
```