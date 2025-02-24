---
title: Download an artifact from a registry
menu:
  default:
    identifier: ko-guides-models-registry-download_use_artifact
    parent: registry
weight: 6
---

W&B Python SDK를 사용하여 레지스트리에 연결된 artifact를 다운로드합니다. artifact를 다운로드하고 사용하려면 레지스트리 이름, 컬렉션 이름, 다운로드할 artifact 버전의 에일리어스 또는 인덱스를 알아야 합니다.

artifact의 속성을 알게 되면 [연결된 artifact의 경로를 구성]({{< relref path="#construct-path-to-linked-artifact" lang="ko" >}})하고 artifact를 다운로드할 수 있습니다. 또는 W&B App UI에서 [미리 생성된 코드 조각을 복사하여 붙여넣어]({{< relref path="#copy-and-paste-pre-generated-code-snippet" lang="ko" >}}) 레지스트리에 연결된 artifact를 다운로드할 수 있습니다.

## 연결된 artifact의 경로 구성

레지스트리에 연결된 artifact를 다운로드하려면 해당 연결된 artifact의 경로를 알아야 합니다. 경로는 레지스트리 이름, 컬렉션 이름, 액세스하려는 artifact 버전의 에일리어스 또는 인덱스로 구성됩니다.

레지스트리, 컬렉션, artifact 버전의 에일리어스 또는 인덱스가 있으면 다음 문자열 템플릿을 사용하여 연결된 artifact의 경로를 구성할 수 있습니다.

```python
# Artifact name with version index specified
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# Artifact name with alias specified
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

중괄호 `{}` 안의 값을 레지스트리 이름, 컬렉션 이름, 액세스하려는 artifact 버전의 에일리어스 또는 인덱스로 바꿉니다.

{{% alert %}}
artifact 버전을 코어 Model registry 또는 코어 Dataset registry에 연결하려면 각각 `model` 또는 `dataset`을 지정하십시오.
{{% /alert %}}

`wandb.init.use_artifact` 메소드를 사용하여 연결된 artifact의 경로가 있으면 artifact에 액세스하고 해당 콘텐츠를 다운로드합니다. 다음 코드 조각은 W&B Registry에 연결된 artifact를 사용하고 다운로드하는 방법을 보여줍니다. `<>` 안의 값을 자신의 값으로 바꾸십시오.

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
# artifact_name = '<artifact_name>' # Copy and paste Full name specified on the Registry App
fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
download_path = fetched_artifact.download()
```

`.use_artifact()` 메소드는 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 생성하고 다운로드하는 artifact를 해당 run의 입력으로 표시합니다. artifact를 run의 입력으로 표시하면 W&B가 해당 artifact의 계보를 추적할 수 있습니다.

run을 생성하지 않으려면 `wandb.Api()` 오브젝트를 사용하여 artifact에 액세스할 수 있습니다.

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
<summary>예시: W&B Registry에 연결된 artifact를 사용하고 다운로드합니다.</summary>

다음 코드 예제는 사용자가 **Fine-tuned Models** 레지스트리의 `phi3-finetuned`라는 컬렉션에 연결된 artifact를 다운로드하는 방법을 보여줍니다. artifact 버전의 에일리어스는 `production`으로 설정됩니다.

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# Initialize a run inside the specified team and project
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# Access an artifact and mark it as input to your run for lineage tracking
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# Download artifact. Returns path to downloaded contents
downloaded_path = fetched_artifact.download()
```
</details>

가능한 파라미터 및 반환 유형에 대한 자세한 내용은 API Reference 가이드에서 [`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ko" >}}) 및 [`Artifact.download()`]({{< relref path="/ref/python/artifact#download" lang="ko" >}})를 참조하십시오.

{{% alert title="여러 조직에 속한 개인 엔터티가 있는 사용자" %}}
여러 조직에 속한 개인 엔터티가 있는 사용자는 레지스트리에 연결된 artifact에 액세스할 때 조직 이름을 지정하거나 팀 엔터티를 사용해야 합니다.

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# Ensure you are using your team entity to instantiate the API
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# Use org display name or org entity in the path
api = wandb.Api()
artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

여기서 `ORG_NAME`은 조직의 표시 이름입니다. Multi-tenant SaaS 사용자는 `https://wandb.ai/account-settings/`의 조직 설정 페이지에서 조직 이름을 찾을 수 있습니다. Dedicated Cloud 및 Self-Managed 사용자는 계정 관리자에게 문의하여 조직의 표시 이름을 확인하십시오.
{{% /alert %}}

## 미리 생성된 코드 조각 복사 및 붙여넣기

W&B는 레지스트리에 연결된 artifact를 다운로드하기 위해 Python 스크립트, 노트북 또는 터미널에 복사하여 붙여넣을 수 있는 코드 조각을 생성합니다.

1. Registry App으로 이동합니다.
2. artifact가 포함된 레지스트리 이름을 선택합니다.
3. 컬렉션 이름을 선택합니다.
4. artifact 버전 목록에서 액세스하려는 버전을 선택합니다.
5. **Usage** 탭을 선택합니다.
6. **Usage API** 섹션에 표시된 코드 조각을 복사합니다.
7. 코드 조각을 Python 스크립트, 노트북 또는 터미널에 붙여넣습니다.

{{< img src="/images/registry/find_usage_in_registry_ui.gif" >}}
