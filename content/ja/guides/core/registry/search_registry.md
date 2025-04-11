---
title: レジストリ項目を見つける
menu:
  default:
    identifier: ja-guides-core-registry-search_registry
    parent: registry
weight: 7
---

[W&B レジストリアプリのグローバル検索バー]({{< relref path="./search_registry.md#search-for-registry-items" lang="ja" >}})を使用して、レジストリ、コレクション、アーティファクトバージョンタグ、コレクションタグ、またはエイリアスを見つけます。W&B Python SDK を使用して、特定の条件に基づいて [レジストリ、コレクション、およびアーティファクトバージョンをフィルタリング]({{< relref path="./search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ja" >}}) するために、MongoDBスタイルのクエリを使用することができます。

表示権限がある項目のみが検索結果に表示されます。

## レジストリ項目の検索

レジストリ項目を検索するには:

1. W&B レジストリアプリに移動します。
2. ページ上部の検索バーに検索語を指定します。Enter を押して検索します。

指定した語が既存のレジストリ、コレクション名、アーティファクトバージョンタグ、コレクションタグ、またはエイリアスと一致する場合、検索バーの下に検索結果が表示されます。

{{< img src="/images/registry/search_registry.gif" alt="レジストリ検索バーにテキストを入力してレジストリ項目をフィルタリングするユーザーの.gif" >}}

## MongoDBスタイルのクエリでレジストリ項目をクエリ

[`wandb.Api().registries()`]({{< relref path="/ref/python/public-api/api.md#registries" lang="ja" >}}) と [query predicates](https://www.mongodb.com/docs/manual/reference/glossary/#std-term-query-predicate) を使用して、1つ以上の [MongoDBスタイルクエリ](https://www.mongodb.com/docs/compass/current/query/filter/) に基づいて、レジストリ、コレクション、およびアーティファクトバージョンをフィルタリングします。

以下の表は、フィルタリングしたい項目の種類に基づいて使用できるクエリの名前を示しています。

| | クエリ名 |
| ----- | ----- |
| registries | `name`, `description`, `created_at`, `updated_at` |
| collections | `name`, `tag`, `description`, `created_at`, `updated_at` |
| versions | `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

以下のコード例は、一般的な検索シナリオを示しています。

`wandb.Api().registries()` メソッドを使用するには、まず W&B Python SDK （[`wandb`]({{< relref path="/ref/python/_index.md" lang="ja" >}})） ライブラリをインポートします。
```python
import wandb

# （オプション）読みやすさのために、wandb.Api() クラスのインスタンスを作成します。
api = wandb.Api()
```

`model` という文字列を含むすべてのレジストリをフィルタリングします。

```python
# `model` という文字列を含むすべてのレジストリをフィルタリングします。
registry_filters = {
    "name": {"$regex": "model"}
}

# フィルタに一致するすべてのレジストリの反復可能なオブジェクトを返します
registries = api.registries(filter=registry_filters)
```

レジストリに関係なく、コレクション名に `yolo` という文字列を含むすべてのコレクションをフィルタリングします。

```python
# レジストリに関係なく、コレクション名に 
# `yolo` という文字列を含むすべてのコレクションをフィルタリングします。
collection_filters = {
    "name": {"$regex": "yolo"}
}

# フィルタに一致するすべてのコレクションの反復可能なオブジェクトを返します
collections = api.registries().collections(filter=collection_filters)
```

レジストリに関係なく、コレクション名に `yolo` という文字列を含み、`cnn` というタグを持つすべてのコレクションをフィルタリングします。

```python
# レジストリに関係なく、コレクション名に 
# `yolo` という文字列を含み、`cnn` というタグを持つすべてのコレクションをフィルタリングします。
collection_filters = {
    "name": {"$regex": "yolo"},
    "tag": "cnn"
}

# フィルタに一致するすべてのコレクションの反復可能なオブジェクトを返します
collections = api.registries().collections(filter=collection_filters)
```

`model` という文字列を含むすべてのアーティファクトバージョンを検索し、`image-classification` というタグまたは `latest` エイリアスを持っているもの：

```python
# `model` という文字列を含むすべてのアーティファクトバージョンを検索し、
# `image-classification` というタグまたは `latest` エイリアスを持っているもの。
registry_filters = {
    "name": {"$regex": "model"}
}

# 論理 $or 演算子を使用してアーティファクトバージョンをフィルタリングします
version_filters = {
    "$or": [
        {"tag": "image-classification"},
        {"alias": "production"}
    ]
}

# フィルタに一致するすべてのアーティファクトバージョンの反復可能なオブジェクトを返します
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)
```

詳細については、MongoDB ドキュメントの [logical query operators](https://www.mongodb.com/docs/manual/reference/operator/query-logical/) を参照してください。

前述のコードスニペット内の `artifacts` の反復可能なオブジェクトの各項目は、`Artifact` クラスのインスタンスです。つまり、各アーティファクトの属性（`name`、`collection`、`aliases`、`tags`、`created_at` など）にアクセスできます。

```python
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```

アーティファクトオブジェクトの属性の完全なリストについては、API Reference docs の [Artifacts Class]({{< relref path="/ref/python/artifact/_index.md" lang="ja" >}}) を参照してください。

レジストリやコレクションに関係なく、2024-01-08 から 2025-03-04 の 13:10 UTC の間に作成されたすべてのアーティファクトバージョンをフィルタリングします。

```python
# 2024-01-08 から 2025-03-04 の 13:10 UTC の間に作成された
# すべてのアーティファクトバージョンを検索します。

artifact_filters = {
    "alias": "latest",
    "created_at" : {"$gte": "2024-01-08", "$lte": "2025-03-04 13:10:00"},
}

# フィルタに一致するすべてのアーティファクトバージョンの反復可能なオブジェクトを返します
artifacts = api.registries().collections().versions(filter=artifact_filters)
```

日付と時刻は `YYYY-MM-DD HH:MM:SS` の形式で指定します。日付のみでフィルタリングしたい場合は、時間、分、秒を省略することができます。

詳細は、MongoDB ドキュメントの [query comparisons](https://www.mongodb.com/docs/manual/reference/operator/query-comparison/) を参照してください。