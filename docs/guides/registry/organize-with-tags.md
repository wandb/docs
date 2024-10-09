---
title: Organize versions with tags
description: 컬렉션 내의 컬렉션 또는 아티팩트 버전을 조직화하려면 태그를 사용하세요. Python SDK 또는 W&B App UI를 사용하여 태그를 추가, 제거, 편집할 수 있습니다.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

컬렉션이나 아티팩트 버전을 레지스트리 내에서 구성하기 위해 태그를 생성하고 추가하세요. W&B App UI 또는 W&B Python SDK를 사용하여 컬렉션이나 아티팩트 버전에 태그를 추가, 수정, 보기 또는 제거할 수 있습니다.

:::tip 태그 사용 시점 vs 에일리어스 사용 시점
특정 아티팩트 버전을 고유하게 참조해야 할 때는 에일리어스를 사용하세요. 예를 들어, 'production'이나 'latest'와 같은 에일리어스를 사용하면 `artifact_name:alias`가 항상 단일, 특정 버전을 가리키도록 할 수 있습니다.

보다 유연하게 그룹화하거나 검색할 필요가 있을 때는 태그를 사용하세요. 동일한 라벨을 여러 버전 또는 컬렉션이 공유할 수 있고, 특정 식별자에 하나의 버전만 연관되어야 한다는 보장이 필요 없는 경우에 태그가 이상적입니다.
:::

## 컬렉션에 태그 추가하기

W&B App UI 또는 Python SDK를 사용하여 컬렉션에 태그를 추가하세요:

<Tabs
  defaultValue="app_ui"
  values={[
    {label: 'W&B App UI', value: 'app_ui'},
    {label: 'Python SDK', value: 'python'},
  ]}>
  <TabItem value="app_ui">

W&B App UI를 사용하여 컬렉션에 태그를 추가하세요:

1. https://wandb.ai/registry 에 있는 W&B Registry로 이동
2. 레지스트리 카드 클릭
3. 컬렉션 이름 옆에 **View details** 클릭
4. 컬렉션 카드에서 **Tags** 필드 옆의 플러스 아이콘(**+**) 클릭 후 태그 이름 입력
5. 키보드의 **Enter** 키 누르기

![](/images/registry/add_tag_collection.gif)

  </TabItem>
  <TabItem value="python">

```python
import wandb

COLLECTION_TYPE = "<collection_type>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"

full_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

collection = wandb.Api().artifact_collection(
    type_name = COLLECTION_TYPE, 
    name = full_name
    )

collection.tags = ["your-tag"]
collection.save()
```

  </TabItem>
</Tabs>

## 컬렉션에 속한 태그 업데이트하기

프로그래밍적으로 태그를 재할당하거나 `tags` 속성을 변형하여 업데이트하세요. W&B는 `tags` 속성을 즉시 변형하는 대신 재할당하는 것이 좋으며, 이는 좋은 Python 관례이기도 합니다.

예를 들어, 아래 코드조각은 리스트를 재할당으로 업데이트하는 일반적인 방법을 보여줍니다. 간결성을 위해 [컬렉션에 태그 추가 섹션](#add-a-tag-to-a-collection)의 코드 예제를 계속 사용합니다:

```python
collection.tags = [*collection.tags, "new-tag", "other-tag"]
collection.tags = collection.tags + ["new-tag", "other-tag"]

collection.tags = set(collection.tags) - set(tags_to_delete)
collection.tags = []  # 모든 태그 삭제
```

다음 코드조각은 아티팩트 버전에 속한 태그를 즉시 변형하여 업데이트하는 방법을 보여줍니다:

```python
collection.tags += ["new-tag", "other-tag"]
collection.tags.append("new-tag")

collection.tags.extend(["new-tag", "other-tag"])
collection.tags[:] = ["new-tag", "other-tag"]
collection.tags.remove("existing-tag")
collection.tags.pop()
collection.tags.clear()
```

## 컬렉션에 속한 태그 보기

W&B App UI를 사용하여 컬렉션에 추가된 태그를 보세요:

1. https://wandb.ai/registry 에 있는 W&B Registry로 이동
2. 레지스트리 카드 클릭
3. 컬렉션 이름 옆에 **View details** 클릭

컬렉션에 하나 이상의 태그가 있으면 컬렉션 카드 내의 **Tags** 필드 옆에서 해당 태그를 볼 수 있습니다.

![](/images/registry/tag_collection_selected.png)

컬렉션에 추가된 태그는 해당 컬렉션 이름 옆에도 표시됩니다.

예를 들어, 아래 이미지에서는 "zoo-dataset-tensors" 컬렉션에 "tag1"이라는 태그가 추가되었습니다.

![](/images/registry/tag_collection.png)

## 컬렉션에서 태그 제거하기

W&B App UI를 사용하여 컬렉션에서 태그를 제거하세요:

1. https://wandb.ai/registry 에 있는 W&B Registry로 이동
2. 레지스트리 카드 클릭
3. 컬렉션 이름 옆에 **View details** 클릭
4. 컬렉션 카드 내에서 제거할 태그 이름 위에 마우스 오버
5. 취소 버튼(**X** 아이콘) 클릭

## 아티팩트 버전에 태그 추가하기

컬렉션과 연결된 아티팩트 버전에 W&B App UI 또는 Python SDK를 사용하여 태그를 추가하세요.

<Tabs
  defaultValue="app_ui"
  values={[
    {label: 'W&B App UI', value: 'app_ui'},
    {label: 'Python SDK', value: 'python'},
  ]}>
  <TabItem value="app_ui">

1. https://wandb.ai/registry 에 있는 W&B Registry로 이동
2. 레지스트리 카드 클릭
3. 태그를 추가할 컬렉션 이름 옆에 **View details** 클릭
4. **Versions**로 스크롤
5. 아티팩트 버전 옆에 **View** 클릭
6. **Version** 탭 내부의 **Tags** 필드 옆의 플러스 아이콘(**+**) 클릭 후 태그 이름 입력
7. 키보드의 **Enter** 키 누르기

![](/images/registry/add_tag_linked_artifact_version.gif)


  </TabItem>  
  <TabItem value="python">

추가하거나 업데이트하고자 하는 아티팩트 버전을 가져오세요. 아티팩트 버전을 보유하면 아티팩트 오브젝트의 `tag` 속성에 엑세스하여 해당 아티팩트에 태그를 추가하거나 수정할 수 있습니다. 리스트로 한 개 이상의 태그를 아티팩트의 `tag` 속성에 전달하세요.

다른 아티팩트처럼, W&B에서 run을 생성하지 않고 아티팩트를 가져오거나, run을 생성하고 그 run 내에서 아티팩트를 가져올 수 있습니다. 어느 경우든지, W&B 서버에 아티팩트를 업데이트하려면 아티팩트 오브젝트의 `save` 메소드를 호출하세요.

유스 케이스에 따라, 아래 코드 셀 중 하나를 복사하여 아티팩트 버전의 태그를 추가 또는 수정하세요. `<>` 안의 값을 귀하의 값으로 교체하세요.

다음 코드조각은 새 run을 생성하지 않고 아티팩트를 가져와서 태그를 추가하는 방법을 보여줍니다:
```python title="Add a tag to an artifact version without creating a new run"
import wandb

ARTIFACT_TYPE = "<TYPE>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = wandb.Api().artifact(name = artifact_name, type = ARTIFACT_TYPE)
artifact.tags = ["tag2"] # Provide one or more tags in a list
artifact.save()
```

다음 코드조각은 새 run을 생성하여 아티팩트를 가져와 태그를 추가하는 방법을 보여줍니다:
```python title="Add a tag to an artifact version during a run"
import wandb

ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

run = wandb.init(entity = "<entity>", project="<project>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
artifact.tags = ["tag2"] # Provide one or more tags in a list
artifact.save()
```

  </TabItem>
</Tabs>

## 아티팩트 버전에 속한 태그 업데이트하기

프로그래밍적으로 태그를 재할당하거나 `tags` 속성을 변형하여 업데이트하세요. W&B는 이가 좋은 Python 관례이기도 하며, `tags` 속성을 즉시 변형하는 대신 재할당하는 것을 추천합니다.

예를 들어, 다음 코드조각은 리스트를 재할당으로 업데이트하는 일반적인 방법을 보여줍니다. 간결성을 위해 [아티팩트 버전에 태그 추가 섹션](#add-a-tag-to-an-artifact-version)의 코드 예제를 계속 사용합니다: 

```python
artifact.tags = [*artifact.tags, "new-tag", "other-tag"]
artifact.tags = artifact.tags + ["new-tag", "other-tag"]

artifact.tags = set(artifact.tags) - set(tags_to_delete)
artifact.tags = []  # 모든 태그 삭제
```

다음 코드조각은 아티팩트 버전에 속한 태그를 즉시 변형하여 업데이트하는 방법을 보여줍니다:

```python
artifact.tags += ["new-tag", "other-tag"]
artifact.tags.append("new-tag")

artifact.tags.extend(["new-tag", "other-tag"])
artifact.tags[:] = ["new-tag", "other-tag"]
artifact.tags.remove("existing-tag")
artifact.tags.pop()
artifact.tags.clear()
```

## 아티팩트 버전에 속한 태그 보기

레지스트리에 연결된 아티팩트 버전에 속한 태그를 W&B App UI 또는 Python SDK를 사용하여 보세요.

<Tabs
  defaultValue="app_ui"
  values={[
    {label: 'W&B App UI', value: 'app_ui'},
    {label: 'Python SDK', value: 'python'},
  ]}>
  <TabItem value="app_ui">

1. https://wandb.ai/registry 에 있는 W&B Registry로 이동
2. 레지스트리 카드 클릭
3. 태그를 추가할 컬렉션 이름 옆에 **View details** 클릭
4. **Versions** 섹션으로 스크롤

아티팩트 버전에 하나 이상의 태그가 있으면 **Tags** 열 내에서 해당 태그를 볼 수 있습니다.

![](/images/registry/tag_artifact_version.png)

  </TabItem>
  <TabItem value="python">

태그를 보기 위해 아티팩트 버전을 가져오세요. 아티팩트 버전을 보유하면 아티팩트 오브젝트의 `tag` 속성을 보아 해당 아티팩트에 속한 태그를 볼 수 있습니다.

다른 아티팩트처럼, W&B에서 run을 생성하지 않고 아티팩트를 가져오거나, run을 생성하고 그 run 내에서 아티팩트를 가져올 수 있습니다.

유스 케이스에 따라, 아래 코드 셀 중 하나를 복사하여 아티팩트 버전의 태그를 보세요. `<>` 안의 값을 귀하의 값으로 교체하세요.

다음 코드조각은 새 run을 생성하지 않고 아티팩트 버전의 태그를 가져와 보는 방법을 보여줍니다:
```python title="Add a tag to an artifact version without creating a new run"
import wandb

ARTIFACT_TYPE = "<TYPE>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = wandb.Api().artifact(name = artifact_name, type = artifact_type)
print(artifact.tags)
```

다음 코드조각은 새 run을 생성하여 아티팩트를 가져와 태그를 보는 방법을 보여줍니다:
```python title="Add a tag to an artifact version during a run"
import wandb

ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

run = wandb.init(entity = "<entity>", project="<project>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
print(artifact.tags)
```

  </TabItem>
</Tabs>

## 아티팩트 버전에서 태그 제거하기

1. https://wandb.ai/registry 에 있는 W&B Registry로 이동
2. 레지스트리 카드 클릭
3. 태그를 추가할 컬렉션 이름 옆에 **View details** 클릭
4. **Versions**로 스크롤
5. 아티팩트 버전 옆에 **View** 클릭
6. **Version** 탭 내에서 태그 이름 위에 마우스 오버
7. 취소 버튼(**X** 아이콘) 클릭

## 기존 태그 검색하기

W&B App UI를 사용하여 컬렉션과 아티팩트 버전에서 기존 태그를 검색하세요:

1. https://wandb.ai/registry 에 있는 W&B Registry로 이동
2. 레지스트리 카드 클릭
3. 검색 바 내에서 태그 이름 입력

![](/images/registry/search_tags.gif)

## 특정 태그를 가진 아티팩트 버전 찾기

W&B Python SDK를 사용하여 특정 태그를 가진 아티팩트 버전을 찾으세요:

```python
import wandb

api = wandb.Api()
tagged_artifact_versions = api.artifacts(
    type_name = "<artifact_type>",
    name = "<artifact_name>",
    tags = ["<tag_1>", "<tag_2>"]
)

for artifact_version in tagged_artifact_versions:
    print(artifact_version.tags)
```