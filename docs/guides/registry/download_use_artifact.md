---
title: Download and use an artifact from a registry
displayed_sidebar: default
---

W&B Python SDK를 사용하여 W&B Registry에 연결된 아티팩트를 사용하고 다운로드하세요.

:::note
특정 아티팩트에 대한 경로 정보가 미리 채워진 사용법 코드조각을 찾으려면, [Copy and paste the usage path from the Registry UI](#copy-and-paste-the-usage-path-from-the-registry-ui) 섹션을 참조하세요.
:::

`<>` 안의 값을 여러분의 값으로 교체하세요:

```python
import wandb

ORG_ENTITY_NAME = '<org-entity-name>'
REGISTRY_NAME = '<registry-name>'
COLLECTION_NAME = '<collection-name>'
ALIAS = '<artifact-alias>'
INDEX = '<artifact-index>'

run = wandb.init()  # 추가적으로 entity와 project 인수를 사용하여 run이 생성될 위치를 지정할 수 있습니다

registered_artifact_name = f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
registered_artifact = run.use_artifact(artifact_or_name=name)  # 이 아티팩트를 run의 입력으로 표시합니다
artifact_dir = registered_artifact.download()  
```

다음에 나열된 형식을 사용하여 아티팩트 버전을 참조하세요:

```python
# 버전 인덱스가 지정된 아티팩트 이름
f"{ORG_ENTITY}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INDEX}"

# 에일리어스가 지정된 아티팩트 이름
f"{ORG_ENTITY}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
```
여기서:
* `latest` - 가장 최근에 연결된 버전을 지정하기 위해 `latest` 에일리어스를 사용하세요.
* `v#` - 컬렉션에서 특정 버전을 가져오기 위해 `v0`, `v1`, `v2` 등을 사용하세요.
* `alias` - 아티팩트 버전에 부착된 사용자 정의 에일리어스를 지정하세요.

가능한 파라미터와 반환 타입에 대한 자세한 정보는 API Reference 가이드의 [`use_artifact`](../../ref/python/run.md#use_artifact) 및 [`download`](/ref/python/artifact#download)를 참조하세요.

<details>
<summary>예제: W&B Registry에 연결된 아티팩트를 사용하고 다운로드</summary>

예를 들어, 다음 코드조각에서 사용자는 `use_artifact` API를 호출했습니다. 그들은 가져오고자 하는 모델 아티팩트의 이름을 지정하고 버전/에일리어스를 제공했습니다. 그런 다음, API에서 반환된 경로를 `downloaded_path` 변수에 저장했습니다.

```python
import wandb
TEAM_NAME = "product-team-applications"
PROJECT_NAME = "user-stories"

ORG_ENTITY_NAME = "wandb"
REGISTRY_NAME = "Fine-tuned Models"
COLLECTION_NAME = "phi3-finetuned"
ALIAS = 'production'

# 지정된 팀과 프로젝트 내부에서 run을 초기화
run = wandb.init(entity=TEAM_NAME, propject=PROJECT_NAME)

registered_artifact_name = f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"

# 아티팩트에 엑세스하고, 계보 추적을 위해 run의 입력으로 표시
registered_artifact = run.use_artifact(artifact_or_name=name)  # 
# 아티팩트를 다운로드합니다. 다운로드된 내용의 경로를 반환합니다
downloaded_path = registered_artifact.download()  
```
</details>

## Registry UI에서 사용 경로 복사 및 붙여넣기

Registry UI의 Usage 탭에서 특정 아티팩트 버전을 사용하고 다운로드할 수 있는 정확한 코드조각을 찾아 경로를 직접 구성하지 않아도 됩니다. 필요한 필드는 보고 있는 아티팩트 버전의 세부 정보에 따라 채워집니다:

1. **W&B Registry로 이동:**
   - 사이드바의 "Registry" 탭으로 가서 레지스트리 목록에 엑세스합니다.

2. **원하는 레지스트리를 선택:**
   - 레지스트리 목록에서 사용하려는 아티팩트를 포함한 레지스트리를 클릭하세요.

3. **아티팩트 컬렉션 찾기:**
   - 레지스트리 세부 페이지에서 원하는 아티팩트를 포함한 컬렉션을 찾습니다. 컬렉션 이름을 클릭하여 버전을 확인합니다.

4. **Usage 탭 엑세스:**
   - 필요한 아티팩트의 버전을 클릭하세요. 이는 아티팩트 버전 세부 페이지를 엽니다.
   - 아티팩트 버전 세부 페이지에서 "Usage" 탭으로 전환하세요.

5. **코드조각 복사:**
   - "Usage" 탭에서 아티팩트를 사용하고 다운로드할 수 있는 코드조각을 볼 수 있습니다. 이 조각들은 정확한 경로로 미리 채워져 있습니다.
   - 자신의 유스 케이스에 맞는 코드를 복사하세요. 코드조각은 다음과 같습니다:

   ```python
   import wandb

   run = wandb.init()

   artifact = run.use_artifact('registries-bug-bash/wandb-registry-model/registry-quickstart-collection:v3', type='model')
   artifact_dir = artifact.download()
   ```
   ![](/images/registry/find_usage_in_registry_ui.gif)