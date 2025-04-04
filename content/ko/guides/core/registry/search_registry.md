---
title: Find registry items
menu:
  default:
    identifier: ko-guides-core-registry-search_registry
    parent: registry
weight: 7
---

[W&B Registry App의 글로벌 검색 창]({{< relref path="./search_registry.md#search-for-registry-items" lang="ko" >}})을 사용하여 registry, collection, artifact version tag, collection tag 또는 에일리어스를 찾으세요. W&B Python SDK를 사용하여 특정 기준에 따라 [MongoDB 스타일 쿼리로 registries, collections 및 artifact versions을 필터링]({{< relref path="./search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ko" >}})할 수 있습니다.

보기 권한이 있는 항목만 검색 결과에 나타납니다.

## Registry 항목 검색

registry 항목을 검색하려면 다음을 수행하세요.

1. W&B Registry App으로 이동합니다.
2. 페이지 상단의 검색 창에 검색어를 지정합니다. Enter 키를 눌러 검색합니다.

지정한 용어가 기존 registry, collection 이름, artifact version tag, collection tag 또는 에일리어스와 일치하면 검색 결과가 검색 창 아래에 나타납니다.

{{< img src="/images/registry/search_registry.gif" alt=".gif of user typing text into registry search bar to filter registry items" >}}

## MongoDB 스타일 쿼리로 registry 항목 쿼리

[`wandb.Api().registries()`]({{< relref path="/ref/python/public-api/api.md#registries" lang="ko" >}}) 및 [쿼리 predicates](https://www.mongodb.com/docs/manual/reference/glossary/#std-term-query-predicate)를 사용하여 하나 이상의 [MongoDB 스타일 쿼리](https://www.mongodb.com/docs/compass/current/query/filter/)를 기반으로 registries, collections 및 artifact versions을 필터링합니다.

다음 표는 필터링하려는 항목 유형에 따라 사용할 수 있는 쿼리 이름을 나열합니다.

| | query name |
| ----- | ----- |
| registries | `name`, `description`, `created_at`, `updated_at` |
| collections | `name`, `tag`, `description`, `created_at`, `updated_at` |
| versions | `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

다음 코드 예제는 몇 가지 일반적인 검색 시나리오를 보여줍니다.

`wandb.Api().registries()` 메소드를 사용하려면 먼저 W&B Python SDK([`wandb`]({{< relref path="/ref/python/_index.md" lang="ko" >}})) 라이브러리를 가져옵니다.
```python
import wandb

# (선택 사항) 가독성을 위해 wandb.Api() 클래스의 인스턴스를 생성합니다.
api = wandb.Api()
```

문자열 `model`을 포함하는 모든 registries를 필터링합니다.

```python
# 문자열 `model`을 포함하는 모든 registries를 필터링합니다.
registry_filters = {
    "name": {"$regex": "model"}
}

# 필터와 일치하는 모든 registries의 iterable을 반환합니다.
registries = api.registries(filter=registry_filters)
```

collection 이름에 문자열 `yolo`를 포함하는 registry에 관계없이 모든 collections을 필터링합니다.

```python
# collection 이름에 문자열 `yolo`를 포함하는 registry에 관계없이
# 모든 collections을 필터링합니다.
collection_filters = {
    "name": {"$regex": "yolo"}
}

# 필터와 일치하는 모든 collections의 iterable을 반환합니다.
collections = api.registries().collections(filter=collection_filters)
```

collection 이름에 문자열 `yolo`를 포함하고 `cnn`을 태그로 갖는 registry에 관계없이 모든 collections을 필터링합니다.

```python
# collection 이름에 문자열 `yolo`를 포함하고 `cnn`을 태그로 갖는
# registry에 관계없이 모든 collections을 필터링합니다.
collection_filters = {
    "name": {"$regex": "yolo"},
    "tag": "cnn"
}

# 필터와 일치하는 모든 collections의 iterable을 반환합니다.
collections = api.registries().collections(filter=collection_filters)
```

문자열 `model`을 포함하고 태그 `image-classification` 또는 `latest` 에일리어스를 갖는 모든 artifact versions을 찾습니다.

```python
# 문자열 `model`을 포함하고
# 태그 `image-classification` 또는 `latest` 에일리어스를 갖는 모든 artifact versions을 찾습니다.
registry_filters = {
    "name": {"$regex": "model"}
}

# 논리적 $or 연산자를 사용하여 artifact versions을 필터링합니다.
version_filters = {
    "$or": [
        {"tag": "image-classification"},
        {"alias": "production"}
    ]
}

# 필터와 일치하는 모든 artifact versions의 iterable을 반환합니다.
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)
```

[논리적 쿼리 연산자](https://www.mongodb.com/docs/manual/reference/operator/query-logical/)에 대한 자세한 내용은 MongoDB 설명서를 참조하세요.

이전 코드 조각에서 `artifacts` iterable의 각 항목은 `Artifact` 클래스의 인스턴스입니다. 즉, 각 아티팩트의 속성 (예: `name`, `collection`, `aliases`, `tags`, `created_at` 등)에 엑세스할 수 있습니다.

```python
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```
아티팩트 오브젝트의 속성 전체 목록은 API Reference 문서의 [Artifacts Class]({{< relref path="/ref/python/artifact/_index.md" lang="ko" >}})를 참조하세요.

2024-01-08과 2025-03-04 13:10 UTC 사이에 생성된 registry 또는 collection에 관계없이 모든 artifact versions을 필터링합니다.

```python
# 2024-01-08과 2025-03-04 13:10 UTC 사이에 생성된 모든 artifact versions을 찾습니다.

artifact_filters = {
    "alias": "latest",
    "created_at" : {"$gte": "2024-01-08", "$lte": "2025-03-04 13:10:00"},
}

# 필터와 일치하는 모든 artifact versions의 iterable을 반환합니다.
artifacts = api.registries().collections().versions(filter=artifact_filters)
```

날짜 및 시간을 `YYYY-MM-DD HH:MM:SS` 형식으로 지정합니다. 날짜로만 필터링하려면 시간, 분, 초를 생략할 수 있습니다.

[쿼리 비교](https://www.mongodb.com/docs/manual/reference/operator/query-comparison/)에 대한 자세한 내용은 MongoDB 설명서를 참조하세요.
