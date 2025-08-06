---
title: 태그로 버전 정리하기
description: 태그를 사용하여 컬렉션 또는 컬렉션 내의 artifact 버전을 효율적으로 정리할 수 있습니다. 태그는 Python SDK나
  W&B App UI를 통해 추가, 삭제, 편집할 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-registry-organize-with-tags
    parent: registry
weight: 7
---

컬렉션이나 아티팩트 버전을 체계적으로 관리할 때 태그를 생성하고 추가할 수 있습니다. 태그는 W&B App UI 또는 W&B Python SDK를 통해 컬렉션이나 아티팩트 버전에 추가, 수정, 확인 또는 삭제할 수 있습니다.

{{% alert title="태그와 에일리어스 중 언제 사용할지" %}}
특정 아티팩트 버전을 고유하게 참조해야 할 때는 에일리어스를 사용하세요. 예를 들어, `'production'`이나 `'latest'`와 같은 에일리어스를 사용하면 항상 `artifact_name:alias`가 단일 특정 버전을 가리키도록 보장할 수 있습니다.

반면, 여러 버전이나 컬렉션이 동일한 라벨을 공유할 수 있고, 특정 식별자에 단 하나의 버전만 연결되어야 하는 보장이 필요 없을 때는 태그를 사용하는 것이 더 유연합니다. 태그는 그룹화나 검색이 필요할 때 이상적입니다.
{{% /alert %}}


## 컬렉션에 태그 추가하기

W&B App UI 또는 Python SDK를 사용해 컬렉션에 태그를 추가할 수 있습니다:

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

W&B App UI에서 컬렉션에 태그를 추가하는 방법:

1. [W&B Registry App](https://wandb.ai/registry)으로 이동합니다.
2. 원하는 레지스트리 카드를 클릭합니다.
3. 컬렉션 이름 옆의 **View details**를 클릭합니다.
4. 컬렉션 카드 내에서 **Tags** 필드 옆의 플러스 아이콘(**+**)을 클릭하고 태그 이름을 입력합니다.
5. 키보드의 **Enter**를 누릅니다.

{{< img src="/images/registry/add_tag_collection.gif" alt="Registry 컬렉션에 태그 추가하기" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}

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

{{% /tab %}}
{{< /tabpane >}}



## 컬렉션에 속한 태그 업데이트

태그를 프로그래밍적으로 업데이트하려면 `tags` 속성을 다시 할당하거나 변경하면 됩니다. W&B에서는 Python에서 일반적으로 권장되는 방식인 직접 수정(in-place mutation)이 아닌, 속성을 재할당하는 방식을 권장합니다.

예를 들어, 다음 코드조각은 재할당을 이용해 리스트를 업데이트하는 방법을 보여줍니다. 자세한 내용은 [컬렉션에 태그 추가하기]({{< relref path="#add-a-tag-to-a-collection" lang="ko" >}}) 섹션의 예시를 참고하세요:

```python
collection.tags = [*collection.tags, "new-tag", "other-tag"]
collection.tags = collection.tags + ["new-tag", "other-tag"]

collection.tags = set(collection.tags) - set(tags_to_delete)
collection.tags = []  # 모든 태그 삭제
```

다음 코드조각은 직접 수정(in-place mutation) 방식을 사용하여 컬렉션의 태그를 업데이트하는 방법입니다:

```python
collection.tags += ["new-tag", "other-tag"]
collection.tags.append("new-tag")

collection.tags.extend(["new-tag", "other-tag"])
collection.tags[:] = ["new-tag", "other-tag"]
collection.tags.remove("existing-tag")
collection.tags.pop()
collection.tags.clear()
```

## 컬렉션에 속한 태그 확인

W&B App UI를 사용하여 컬렉션에 추가된 태그를 확인할 수 있습니다:

1. [W&B Registry App](https://wandb.ai/registry)으로 이동합니다.
2. 레지스트리 카드를 클릭합니다.
3. 컬렉션 이름 옆의 **View details**를 클릭합니다.

컬렉션에 하나 이상의 태그가 있을 경우, 컬렉션 카드 내 **Tags** 필드 옆에서 태그를 확인할 수 있습니다.

{{< img src="/images/registry/tag_collection_selected.png" alt="태그가 선택된 Registry 컬렉션" >}}

컬렉션에 추가된 태그는 컬렉션 이름 옆에도 표시됩니다.

예를 들어, 위 이미지에서는 "tag1"이라는 태그가 "zoo-dataset-tensors" 컬렉션에 추가된 것을 볼 수 있습니다.

{{< img src="/images/registry/tag_collection.png" alt="태그 관리" >}}


## 컬렉션에서 태그 삭제하기

W&B App UI를 사용하여 컬렉션에서 태그를 삭제하는 방법은 다음과 같습니다:

1. [W&B Registry App](https://wandb.ai/registry)으로 이동합니다.
2. 레지스트리 카드를 클릭합니다.
3. 컬렉션 이름 옆의 **View details**를 클릭합니다.
4. 컬렉션 카드 내에서 삭제하려는 태그 이름 위에 마우스를 올립니다.
5. 취소 버튼(**X** 아이콘)을 클릭합니다.

## 아티팩트 버전에 태그 추가하기

컬렉션과 연결된 아티팩트 버전에 태그를 추가하려면 W&B App UI 또는 Python SDK를 활용하세요.

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}
1. https://wandb.ai/registry에서 W&B Registry에 접속하세요.
2. 레지스트리 카드를 클릭합니다.
3. 태그를 추가하려는 컬렉션 이름 옆의 **View details**를 클릭합니다.
4. 아래로 스크롤하여 **Versions**를 찾습니다.
5. 아티팩트 버전 옆의 **View**를 클릭합니다.
6. **Version** 탭에서 **Tags** 필드 옆 **+** 아이콘을 클릭하고 태그 이름을 입력합니다.
7. 키보드의 **Enter**를 누릅니다.

{{< img src="/images/registry/add_tag_linked_artifact_version.gif" alt="아티팩트 버전에 태그 추가하기" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}
태그를 추가 또는 업데이트하려는 아티팩트 버전을 불러옵니다. 아티팩트 버전을 불러온 후 아티팩트 오브젝트의 `tag` 속성을 이용해 태그를 추가하거나 수정할 수 있습니다. 하나 이상의 태그를 list 형태로 `tag` 속성에 전달하세요.

다른 Artifacts와 마찬가지로, 별도의 run을 생성하지 않고 W&B에서 아티팩트를 불러올 수도 있고, 새로운 run을 생성한 후 그 안에서 아티팩트를 불러올 수도 있습니다. 어떤 방법이든, 반드시 아티팩트 오브젝트의 `save` 메소드를 호출하여 W&B 서버에 변경사항이 반영되도록 해야 합니다.

아래 코드조각을 참고해 알맞은 방식으로 태그를 추가하거나 수정할 수 있습니다. `<>` 안의 값은 사용자 환경에 맞게 바꿔주세요.


다음은 run을 생성하지 않고 아티팩트를 불러와서 태그를 추가하는 방법입니다:
```python title="새로운 run 없이 아티팩트 버전에 태그 추가"
import wandb

ARTIFACT_TYPE = "<TYPE>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = wandb.Api().artifact(name = artifact_name, type = ARTIFACT_TYPE)
artifact.tags = ["tag2"] # 하나 이상의 태그를 list로 입력
artifact.save()
```


다음은 새로운 run을 만들면서 아티팩트에 태그를 추가하는 방법입니다:

```python title="run 중에 아티팩트 버전에 태그 추가"
import wandb

ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

run = wandb.init(entity = "<entity>", project="<project>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
artifact.tags = ["tag2"] # 하나 이상의 태그를 list로 입력
artifact.save()
```

{{% /tab %}}
{{< /tabpane >}}



## 아티팩트 버전에 속한 태그 업데이트

태그를 프로그래밍적으로 업데이트하려면 `tags` 속성을 다시 할당하거나 변경하면 됩니다. W&B에서는 Python에서 일반적으로 권장되는 방식인 직접 수정(in-place mutation)이 아닌, 속성을 재할당하는 방식을 권장합니다.

예를 들어, 다음 코드조각은 재할당을 이용해 리스트를 업데이트하는 방법을 보여줍니다. 자세한 내용은 [아티팩트 버전에 태그 추가하기]({{< relref path="#add-a-tag-to-an-artifact-version" lang="ko" >}}) 섹션의 예시를 참고하세요:

```python
artifact.tags = [*artifact.tags, "new-tag", "other-tag"]
artifact.tags = artifact.tags + ["new-tag", "other-tag"]

artifact.tags = set(artifact.tags) - set(tags_to_delete)
artifact.tags = []  # 모든 태그 삭제
```

다음 코드조각은 직접 수정(in-place mutation) 방식을 사용하여 아티팩트 버전의 태그를 업데이트하는 방법입니다:

```python
artifact.tags += ["new-tag", "other-tag"]
artifact.tags.append("new-tag")

artifact.tags.extend(["new-tag", "other-tag"])
artifact.tags[:] = ["new-tag", "other-tag"]
artifact.tags.remove("existing-tag")
artifact.tags.pop()
artifact.tags.clear()
```


## 아티팩트 버전에 속한 태그 확인

W&B App UI 또는 Python SDK를 이용해 레지스트리에 연결된 아티팩트 버전의 태그를 확인할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

1. [W&B Registry App](https://wandb.ai/registry)으로 이동합니다.
2. 레지스트리 카드를 클릭합니다.
3. 태그를 추가한 컬렉션 이름 옆의 **View details**를 클릭합니다.
4. 아래로 스크롤하여 **Versions** 섹션을 확인합니다.

아티팩트 버전에 하나 이상의 태그가 있으면 **Tags** 열에서 해당 태그를 확인할 수 있습니다.

{{< img src="/images/registry/tag_artifact_version.png" alt="태그가 있는 아티팩트 버전" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}

아티팩트 버전을 불러와서 태그를 확인하세요. 아티팩트 버전을 불러온 후 아티팩트 오브젝트의 `tag` 속성을 확인하면 해당 아티팩트에 속한 태그를 볼 수 있습니다.

다른 Artifacts와 마찬가지로, 별도의 run 없이 W&B에서 아티팩트를 불러오거나 새로운 run을 만들고 그 run 안에서 아티팩트를 불러올 수도 있습니다.

아래 예시 코드조각을 참고해 태그를 추가하거나 수정할 수 있습니다. `<>` 안의 값은 환경에 맞게 변경하세요.

다음은 run을 생성하지 않고 아티팩트 버전의 태그를 불러오는 방법입니다:

```python title="새로운 run 없이 아티팩트 버전에 태그 추가"
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


아래는 run을 생성해서 아티팩트 버전의 태그를 조회하는 방법입니다:

```python title="run 중에 아티팩트 버전에 태그 추가"
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

{{% /tab %}}
{{< /tabpane >}}



## 아티팩트 버전에서 태그 삭제하기

1. [W&B Registry App](https://wandb.ai/registry)으로 이동합니다.
2. 레지스트리 카드를 클릭합니다.
3. 태그를 추가한 컬렉션 이름 옆의 **View details**를 클릭합니다.
4. 아래로 스크롤하여 **Versions**를 확인합니다.
5. 아티팩트 버전 옆의 **View**를 클릭합니다.
6. **Version** 탭에서 태그 이름 위에 마우스를 올립니다.
7. 취소 버튼(**X** 아이콘)을 클릭합니다.

## 기존 태그 검색

W&B App UI에서 컬렉션 및 아티팩트 버전 내의 기존 태그를 검색할 수 있습니다:

1. [W&B Registry App](https://wandb.ai/registry)으로 이동합니다.
2. 레지스트리 카드를 클릭합니다.
3. 상단의 검색창에 태그 이름을 입력하세요.

{{< img src="/images/registry/search_tags.gif" alt="태그 기반 검색" >}}


## 특정 태그가 있는 아티팩트 버전 찾기

W&B Python SDK를 이용하여 특정 태그가 있는 아티팩트 버전을 찾을 수 있습니다:

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