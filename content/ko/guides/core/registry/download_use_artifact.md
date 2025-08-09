---
title: 레지스트리에서 아티팩트 다운로드하기
menu:
  default:
    identifier: ko-guides-core-registry-download_use_artifact
    parent: registry
weight: 6
---

W&B Python SDK를 사용하여 Registry에 연결된 artifact를 다운로드할 수 있습니다. artifact를 다운로드하고 사용하려면 registry 이름, 컬렉션 이름, 그리고 다운로드하려는 artifact 버전의 에일리어스(alias) 또는 인덱스(index)를 알고 있어야 합니다.

artifact의 속성을 알게 되면 [연결된 artifact의 경로를 구성]({{< relref path="#construct-path-to-linked-artifact" lang="ko" >}})하여 artifact를 다운로드할 수 있습니다. 또는, W&B App UI에서 미리 생성된 [코드조각을 복사하여 붙여넣는 방법]({{< relref path="#copy-and-paste-pre-generated-code-snippet" lang="ko" >}})으로 Registry에 연결된 artifact를 다운로드할 수도 있습니다.

## 연결된 artifact 경로 구성하기

Registry에 연결된 artifact를 다운로드하려면 해당 artifact의 경로를 알아야 합니다. 이 경로는 registry 이름, 컬렉션 이름, 그리고 다운로드하려는 artifact 버전의 에일리어스 또는 인덱스로 구성되어 있습니다.

registry, 컬렉션, 그리고 artifact 버전의 에일리어스 또는 인덱스를 알게 되었으면, 다음과 같은 문자열 템플릿을 사용하여 연결된 artifact의 경로를 구성할 수 있습니다:

```python
# 버전 인덱스를 지정한 artifact 이름
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# 에일리어스를 지정한 artifact 이름
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

중괄호( `{}` ) 안의 값들은 엑세스하려는 registry, 컬렉션, 그리고 artifact 버전의 에일리어스 또는 인덱스로 바꿔서 사용하세요.

{{% alert %}}
artifact 버전을 core Model registry 또는 core Dataset registry에 연결하려면 각각 `model` 또는 `dataset`을 지정하세요.
{{% /alert %}}

연결된 artifact의 경로를 알게 되면 `wandb.init.use_artifact` 메소드를 사용하여 artifact를 엑세스하고 그 내용을 다운로드할 수 있습니다. 아래 코드 예시에서는 W&B Registry에 연결된 artifact를 사용하고 다운로드하는 방법을 보여줍니다. `<>` 안에 있는 값은 직접 입력해야 합니다:

```python
import wandb

REGISTRY = '<registry_name>'
COLLECTION = '<collection_name>'
ALIAS = '<artifact_alias>'

run = wandb.init(
   entity = '<team_name>',
   project = '<project_name>'
   )  

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
# artifact_name = '<artifact_name>' # Registry App에 표시되는 전체 이름을 복사해서 붙여넣을 수도 있습니다.
fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
download_path = fetched_artifact.download()  
```

`.use_artifact()` 메소드는 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 생성하고, 다운로드한 artifact를 해당 run의 입력 값(input)으로 표시합니다.
artifact를 run의 입력으로 표시하면 W&B가 해당 artifact의 계보를 추적할 수 있습니다.

run을 생성하지 않고 artifact를 엑세스하려면, `wandb.Api()` 오브젝트를 사용할 수 있습니다:

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

api = wandb.Api()
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

<details>
<summary>예시: W&B Registry에 연결된 artifact 사용 및 다운로드</summary>

다음 코드 예시는 사용자가 **Fine-tuned Models** registry에 있는 `phi3-finetuned` 컬렉션에 연결된 artifact를 다운로드하는 방법을 보여줍니다. 이 example에서는 artifact 버전의 에일리어스가 `production`으로 설정되어 있습니다.

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# 지정한 팀과 프로젝트 내에서 run을 초기화
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# artifact를 엑세스하고, 계보 추적을 위해 run의 입력으로 표시
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# artifact 다운로드. 다운로드된 내용의 경로를 반환합니다.
downloaded_path = fetched_artifact.download()  
```
</details>

파라미터 및 반환 타입 등 자세한 내용은 API Reference의 [`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ko" >}}) 와 [`Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ko" >}}) 부분을 참고하세요.

{{% alert title="여러 조직에 속한 개인 엔티티 사용자 안내" %}} 
여러 조직에 속한 개인 엔티티를 가진 사용자는 registry에 연결된 artifact를 엑세스할 때 조직의 이름을 지정하거나, 팀 엔티티를 사용해야 합니다.

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# API를 인스턴스화할 때 팀 엔티티를 사용해야 합니다.
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# 경로에 조직 표시 이름 또는 엔티티 사용
api = wandb.Api()
artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

여기서 `ORG_NAME`은 조직의 표시 이름입니다. Multi-tenant SaaS 사용자는 조직 설정 페이지 `https://wandb.ai/account-settings/` (조직의 settings)에서 조직 이름을 확인할 수 있습니다. 전용 클라우드 및 셀프 관리(자체 호스팅) 사용자의 경우, 조직의 표시 이름을 확인하려면 계정 관리자에게 문의하세요.
{{% /alert %}}

## 미리 생성된 코드조각 복사 및 붙여넣기

W&B는 Registry에 연결된 artifact를 다운로드할 수 있도록, 복사해서 Python 스크립트, 노트북, 터미널에 붙여넣을 수 있는 코드조각을 자동으로 생성합니다.

1. Registry App으로 이동합니다.
2. artifact가 들어 있는 registry의 이름을 선택합니다.
3. 컬렉션 이름을 선택합니다.
4. artifact 버전 목록 중 엑세스할 버전을 선택합니다.
5. **Usage** 탭을 선택합니다.
6. **Usage API** 섹션에 표시되는 코드 조각을 복사합니다.
7. 복사한 코드 조각을 Python 스크립트, 노트북, 터미널에 붙여넣어 사용합니다.

{{< img src="/images/registry/find_usage_in_registry_ui.gif" >}}