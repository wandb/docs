---
title: Download an artifact from a registry
menu:
  default:
    identifier: ko-guides-core-registry-download_use_artifact
    parent: registry
weight: 6
---

W&B Python SDK를 사용하여 레지스트리에 연결된 아티팩트를 다운로드합니다. 아티팩트를 다운로드하여 사용하려면 레지스트리 이름, 컬렉션 이름, 다운로드할 아티팩트 버전의 에일리어스 또는 인덱스를 알아야 합니다.

아티팩트의 속성을 알면 [연결된 아티팩트의 경로를 구성]({{< relref path="#construct-path-to-linked-artifact" lang="ko" >}})하고 아티팩트를 다운로드할 수 있습니다. 또는 W&B App UI에서 [미리 생성된 코드 조각을 복사하여 붙여넣어]({{< relref path="#copy-and-paste-pre-generated-code-snippet" lang="ko" >}}) 레지스트리에 연결된 아티팩트를 다운로드할 수 있습니다.

## 연결된 아티팩트의 경로 구성

레지스트리에 연결된 아티팩트를 다운로드하려면 해당 연결된 아티팩트의 경로를 알아야 합니다. 경로는 레지스트리 이름, 컬렉션 이름, 엑세스하려는 아티팩트 버전의 에일리어스 또는 인덱스로 구성됩니다.

레지스트리, 컬렉션, 아티팩트 버전의 에일리어스 또는 인덱스가 있으면 다음 문자열 템플릿을 사용하여 연결된 아티팩트의 경로를 구성할 수 있습니다.

```python
# 버전 인덱스가 지정된 아티팩트 이름
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# 에일리어스가 지정된 아티팩트 이름
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

중괄호 `{}` 안의 값을 엑세스하려는 레지스트리 이름, 컬렉션 이름, 아티팩트 버전의 에일리어스 또는 인덱스로 바꿉니다.

{{% alert %}}
아티팩트 버전을 핵심 Model registry 또는 핵심 Dataset registry에 연결하려면 `model` 또는 `dataset`을 지정하십시오.
{{% /alert %}}

연결된 아티팩트의 경로가 있으면 `wandb.init.use_artifact` 메소드를 사용하여 아티팩트에 엑세스하고 해당 콘텐츠를 다운로드합니다. 다음 코드 조각은 W&B Registry에 연결된 아티팩트를 사용하고 다운로드하는 방법을 보여줍니다. `<>` 안의 값을 자신의 값으로 바꾸십시오.

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
# artifact_name = '<artifact_name>' # Registry App에 지정된 전체 이름을 복사하여 붙여넣습니다.
fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
download_path = fetched_artifact.download()  
```

`.use_artifact()` 메소드는 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 생성하고 다운로드하는 아티팩트를 해당 run의 입력으로 표시합니다.
아티팩트를 run의 입력으로 표시하면 W&B가 해당 아티팩트의 계보를 추적할 수 있습니다.

run을 생성하지 않으려면 `wandb.Api()` 오브젝트를 사용하여 아티팩트에 엑세스할 수 있습니다.

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
<summary>예시: W&B Registry에 연결된 아티팩트 사용 및 다운로드</summary>

다음 코드 예제는 사용자가 **Fine-tuned Models** 레지스트리의 `phi3-finetuned`라는 컬렉션에 연결된 아티팩트를 다운로드하는 방법을 보여줍니다. 아티팩트 버전의 에일리어스는 `production`으로 설정됩니다.

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# 지정된 팀 및 프로젝트 내에서 run 초기화
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# 아티팩트에 엑세스하고 계보 추적을 위해 run에 대한 입력으로 표시
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# 아티팩트 다운로드. 다운로드한 콘텐츠의 경로를 반환합니다.
downloaded_path = fetched_artifact.download()
```
</details>

가능한 파라미터 및 반환 유형에 대한 자세한 내용은 API Reference 가이드의 [`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ko" >}}) 및 [`Artifact.download()`]({{< relref path="/ref/python/artifact#download" lang="ko" >}})를 참조하십시오.

{{% alert title="여러 조직에 속한 개인 엔터티를 가진 Users" %}}
여러 조직에 속한 개인 엔터티를 가진 Users는 레지스트리에 연결된 아티팩트에 엑세스할 때 조직 이름을 지정하거나 팀 엔터티를 사용해야 합니다.

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# 팀 엔터티를 사용하여 API를 인스턴스화해야 합니다.
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# 경로에 조직 표시 이름 또는 조직 엔터티 사용
api = wandb.Api()
artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

여기서 `ORG_NAME`은 조직의 표시 이름입니다. 멀티 테넌트 SaaS Users는 `https://wandb.ai/account-settings/`의 조직 설정 페이지에서 조직 이름을 찾을 수 있습니다. Dedicated Cloud 및 Self-Managed Users는 계정 관리자에게 문의하여 조직의 표시 이름을 확인하십시오.
{{% /alert %}}

## 미리 생성된 코드 조각 복사 및 붙여넣기

W&B는 Python 스크립트, 노트북 또는 터미널에 복사하여 붙여넣어 레지스트리에 연결된 아티팩트를 다운로드할 수 있는 코드 조각을 생성합니다.

1. Registry App으로 이동합니다.
2. 아티팩트가 포함된 레지스트리 이름을 선택합니다.
3. 컬렉션 이름을 선택합니다.
4. 아티팩트 버전 목록에서 엑세스하려는 버전을 선택합니다.
5. **Usage** 탭을 선택합니다.
6. **Usage API** 섹션에 표시된 코드 조각을 복사합니다.
7. 코드 조각을 Python 스크립트, 노트북 또는 터미널에 붙여넣습니다.

{{< img src="/images/registry/find_usage_in_registry_ui.gif" >}}
