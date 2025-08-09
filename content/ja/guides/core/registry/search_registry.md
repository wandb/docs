---
title: Registry項目を見つける
menu:
  default:
    identifier: ja-guides-core-registry-search_registry
    parent: registry
weight: 7
---

[W&B Registry App のグローバル検索バー]({{< relref path="./search_registry.md#search-for-registry-items" lang="ja" >}}) を使って、Registry、コレクション、artifact バージョンのタグ、コレクションタグ、エイリアスを検索できます。W&B Python SDK では、MongoDB 形式のクエリを使って[Registry、コレクション、artifact バージョンをフィルタ]({{< relref path="./search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ja" >}}) することもできます。

自分が閲覧権限を持つアイテムだけが検索結果に表示されます。

## Registryアイテムの検索

Registryアイテムを検索するには：

1. W&B Registry App にアクセスします。
2. ページ上部の検索バーに検索したいキーワードを入力し、Enter キーを押します。

指定したキーワードが既存のRegistry、コレクション名、artifact バージョンタグ、コレクションタグ、エイリアスと一致する場合、検索結果が検索バーの下に表示されます。

{{< img src="/images/registry/search_registry.gif" alt="Searching within a Registry" >}}

## MongoDB形式のクエリでRegistryアイテムを検索

[`wandb.Api().registries()`]({{< relref path="/ref/python/public-api/api.md#registries" lang="ja" >}}) と [query predicates](https://www.mongodb.com/docs/manual/reference/glossary/#std-term-query-predicate) を使うことで、1つ以上の[MongoDB 形式のクエリ](https://www.mongodb.com/docs/compass/current/query/filter/)でRegistry、コレクション、artifact バージョンを柔軟に絞り込めます。

下記の表は、各アイテムタイプごとに使用できるクエリ名の一覧です：

| | クエリ名 |
| ----- | ----- |
| registries | `name`, `description`, `created_at`, `updated_at` |
| collections | `name`, `tag`, `description`, `created_at`, `updated_at` |
| versions | `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

以下のコード例では、よくある検索シナリオを紹介します。

`wandb.Api().registries()` メソッドの利用には、まず W&B Python SDK（[`wandb`]({{< relref path="/ref/python/_index.md" lang="ja" >}})）ライブラリをインポートします。

```python
import wandb

# （オプション）可読性向上のために wandb.Api() のインスタンスを生成
api = wandb.Api()
```

`model` という文字列を含むすべてのRegistryをフィルタ：

```python
# `model` という文字列を含む全てのRegistryをフィルタ
registry_filters = {
    "name": {"$regex": "model"}
}

# フィルタに一致する全てのRegistryがイテラブルで返る
registries = api.registries(filter=registry_filters)
```

Registryを問わず、コレクション名に `yolo` を含むすべてのコレクションをフィルタ：

```python
# Registryを問わず、コレクション名に `yolo` を含む
# すべてのコレクションをフィルタ
collection_filters = {
    "name": {"$regex": "yolo"}
}

# フィルタに一致する全てのコレクションがイテラブルで返る
collections = api.registries().collections(filter=collection_filters)
```

Registryを問わず、コレクション名に `yolo` を含み、タグとして `cnn` を持つすべてのコレクションをフィルタ：

```python
# Registryを問わず、コレクション名に `yolo` を含み、
# タグとして `cnn` を持つすべてのコレクションをフィルタ
collection_filters = {
    "name": {"$regex": "yolo"},
    "tag": "cnn"
}

# フィルタに一致する全てのコレクションがイテラブルで返る
collections = api.registries().collections(filter=collection_filters)
```

`model` という文字列を含み、タグ `image-classification` もしくは `production` エイリアスが付与された artifact バージョンをすべて取得：

```python
# `model` という文字列を含み、タグ `image-classification` または
# `production` エイリアスが付与されたすべての artifact バージョンを検索
registry_filters = {
    "name": {"$regex": "model"}
}

# 論理 $or 演算子で artifact バージョンをフィルタ
version_filters = {
    "$or": [
        {"tag": "image-classification"},
        {"alias": "production"}
    ]
}

# フィルタに一致するすべての artifact バージョンがイテラブルで返る
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)
```

[論理クエリ演算子についての詳細はMongoDBドキュメント](https://www.mongodb.com/docs/manual/reference/operator/query-logical/)を参照してください。

先程のコードスニペット内の `artifacts` イテラブルの各アイテムは `Artifact` クラスのインスタンスであり、`name` や `collection`、`aliases`、`tags`、`created_at` など様々な属性にアクセスできます。

```python
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```
artifact オブジェクトで利用可能な属性の一覧は [Artifacts Class]({{< relref path="/ref/python/sdk/classes/artifact/_index.md" lang="ja" >}}) をご覧ください。


Registryやコレクションに依存せず、2024-01-08 から 2025-03-04 13:10 UTC の間に作成されたすべての artifact バージョンをフィルタ：

```python
# 2024-01-08 から 2025-03-04 13:10 UTC の間に作成された
# すべての artifact バージョンを検索

artifact_filters = {
    "alias": "latest",
    "created_at" : {"$gte": "2024-01-08", "$lte": "2025-03-04 13:10:00"},
}

# フィルタに一致するすべての artifact バージョンがイテラブルで返る
artifacts = api.registries().collections().versions(filter=artifact_filters)
```

日時は `YYYY-MM-DD HH:MM:SS` 形式で指定してください。日付のみでフィルタしたい場合、時・分・秒は省略できます。

[比較クエリについての詳細はMongoDBドキュメント](https://www.mongodb.com/docs/manual/reference/operator/query-comparison/)をご覧ください。