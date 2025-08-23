---
title: Registry 아이템 찾기
menu:
  default:
    identifier: ko-guides-core-registry-search_registry
    parent: registry
weight: 7
---

[W&B Registry App의 글로벌 검색창]({{< relref path="./search_registry.md#search-for-registry-items" lang="ko" >}})을 이용해 registry, collection, artifact version tag, collection tag 또는 에일리어스를 찾아보세요. W&B Python SDK를 사용하면 MongoDB 스타일 쿼리로 [registry, collection, artifact version을 필터링]({{< relref path="./search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ko" >}})할 수 있습니다.

검색 결과에는 본인이 접근 권한이 있는 항목만 표시됩니다.

## Registry 항목 검색하기

Registry 항목을 검색하려면:

1. W&B Registry App으로 이동합니다.
2. 페이지 상단의 검색창에 검색어를 입력한 뒤 Enter 키를 누르세요.

입력한 검색어가 존재하는 registry, collection 이름, artifact version tag, collection tag, 또는 에일리어스와 일치하면 검색창 아래에 결과가 표시됩니다.

{{< img src="/images/registry/search_registry.gif" alt="Registry 내 검색하기" >}}

## MongoDB 스타일 쿼리로 registry 항목 조회하기

[`wandb.Api().registries()`]({{< relref path="/ref/python/public-api/api.md#registries" lang="ko" >}})와 [query predicates](https://www.mongodb.com/docs/manual/reference/glossary/#std-term-query-predicate)를 활용하여 MongoDB 스타일 쿼리로 registry, collection, artifact version을 다양한 필터 기준으로 조회할 수 있습니다.  
좀 더 상세한 MongoDB 스타일 쿼리 사용법은 [여기](https://www.mongodb.com/docs/compass/current/query/filter/)에서 확인할 수 있습니다.

아래 표는 필터링하려는 항목별로 사용할 수 있는 쿼리 이름을 정리한 것입니다.

| | 쿼리 이름 |
| ----- | ----- |
| registries | `name`, `description`, `created_at`, `updated_at` |
| collections | `name`, `tag`, `description`, `created_at`, `updated_at` |
| versions | `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

아래 코드 예시는 자주 사용되는 검색 상황을 보여줍니다.

`wandb.Api().registries()` 메소드를 사용하려면 먼저 W&B Python SDK([`wandb`]({{< relref path="/ref/python/_index.md" lang="ko" >}})) 라이브러리를 임포트해야 합니다:
```python
import wandb

# (선택) 가독성을 위해 wandb.Api() 클래스의 인스턴스를 생성합니다.
api = wandb.Api()
```

`model` 문자열이 포함된 모든 registry를 필터링하기:

```python
# 'model' 문자열이 포함된 모든 registry를 필터링합니다.
registry_filters = {
    "name": {"$regex": "model"}
}

# 해당 필터 조건을 만족하는 모든 registry의 iterable을 반환합니다.
registries = api.registries(filter=registry_filters)
```

registry와 관계없이 collection 이름에 `yolo`가 포함된 모든 collection을 필터링하기:

```python
# registry 상관없이 collection 이름에
# 'yolo'가 포함된 모든 collection을 필터링합니다.
collection_filters = {
    "name": {"$regex": "yolo"}
}

# 해당 필터에 맞는 모든 collection의 iterable을 반환합니다.
collections = api.registries().collections(filter=collection_filters)
```

registry에 상관없이 collection 이름에 `yolo`가 포함되고, 태그가 `cnn`인 collection만 필터링하기:

```python
# registry에 상관없이 collection 이름에 'yolo'가 포함되고
# 태그가 'cnn'인 collection만 필터링합니다.
collection_filters = {
    "name": {"$regex": "yolo"},
    "tag": "cnn"
}

# 해당 조건에 맞는 collection의 iterable을 반환합니다.
collections = api.registries().collections(filter=collection_filters)
```

`model`이 포함되고, 태그가 `image-classification`이거나 alias가 `production`인 artifact version 모두 찾기:

```python
# 'model' 문자열이 포함되고,
# 태그가 'image-classification'이거나 에일리어스가 'production'인
# 모든 artifact version을 찾습니다.
registry_filters = {
    "name": {"$regex": "model"}
}

# 논리 $or 연산자를 사용해 artifact version을 필터링합니다.
version_filters = {
    "$or": [
        {"tag": "image-classification"},
        {"alias": "production"}
    ]
}

# 해당 조건에 맞는 모든 artifact version의 iterable을 반환합니다.
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)
```

논리 연산자에 대한 더 자세한 내용은 MongoDB 문서의 [logical query operators](https://www.mongodb.com/docs/manual/reference/operator/query-logical/)를 참고하세요.

위 코드 예제의 `artifacts` iterable의 각 항목은 `Artifact` 클래스의 인스턴스입니다. 즉, 각 artifact의 속성(예: `name`, `collection`, `aliases`, `tags`, `created_at` 등)에 접근할 수 있습니다:

```python
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```
artifact 오브젝트 속성의 전체 목록은 API Reference 문서 내 [Artifacts Class]({{< relref path="/ref/python/sdk/classes/artifact/_index.md" lang="ko" >}})를 참고하세요.

2024-01-08부터 2025-03-04 13:10 UTC 기간에 생성된, registry나 collection 관계없이 모든 artifact version을 필터링하기:

```python
# 2024-01-08부터 2025-03-04 13:10 UTC 까지 생성된 artifact version 모두 찾기

artifact_filters = {
    "alias": "latest",
    "created_at" : {"$gte": "2024-01-08", "$lte": "2025-03-04 13:10:00"},
}

# 해당 조건에 맞는 모든 artifact version의 iterable을 반환합니다.
artifacts = api.registries().collections().versions(filter=artifact_filters)
```

날짜와 시간은 `YYYY-MM-DD HH:MM:SS` 형식으로 지정하세요. 날짜 기준만으로 필터할 땐 시간(시/분/초)은 생략해도 됩니다.

쿼리 비교 연산자에 대한 자세한 내용은 MongoDB 문서의 [query comparisons](https://www.mongodb.com/docs/manual/reference/operator/query-comparison/)를 참고하세요.