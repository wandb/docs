---
title: レジストリ アイテムを検索する
menu:
  default:
    identifier: search_registry
    parent: registry
weight: 7
---

[W&B Registry App のグローバル検索バー]({{< relref "./search_registry.md#search-for-registry-items" >}}) を使って、レジストリ、コレクション、アーティファクトバージョンのタグ、コレクションタグ、エイリアスを検索できます。W&B Python SDK を使って、特定の条件に基づいて [レジストリ、コレクション、アーティファクトバージョンをフィルタリング]({{< relref "./search_registry.md#query-registry-items-with-mongodb-style-queries" >}}) するための MongoDB スタイルのクエリも利用可能です。

検索結果には、自分が閲覧権限を持つアイテムのみが表示されます。

## レジストリアイテムの検索

レジストリアイテムを検索する手順：

1. W&B Registry App にアクセスします。
2. ページ上部の検索バーに検索語を入力し、Enter キーを押します。

指定した語句が既存のレジストリ、コレクション名、アーティファクトバージョンのタグ、コレクションタグ、エイリアスのいずれかに一致すれば、検索結果が検索バーの下に表示されます。

{{< img src="/images/registry/search_registry.gif" alt="レジストリ内での検索" >}}

## MongoDB スタイルのクエリでレジストリアイテムを絞り込む

[`wandb.Api().registries()`]({{< relref "/ref/python/public-api/api.md#registries" >}}) と [クエリ述語](https://www.mongodb.com/docs/manual/reference/glossary/#std-term-query-predicate) を使うことで、1つ以上の [MongoDBスタイルクエリ](https://www.mongodb.com/docs/compass/current/query/filter/) に基づき、レジストリ、コレクション、アーティファクトバージョンを絞り込めます。

以下の表は、フィルタリング対象ごとに利用できるクエリ名を示しています：

| | クエリ名 |
| ----- | ----- |
| registries | `name`, `description`, `created_at`, `updated_at` |
| collections | `name`, `tag`, `description`, `created_at`, `updated_at` |
| versions | `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

以下のコード例では、よくある検索ケースを紹介します。

`wandb.Api().registries()` メソッドを利用するには、まず W&B Python SDK（[`wandb`]({{< relref "/ref/python/_index.md" >}})）ライブラリをインポートします。

```python
import wandb

# （任意）見やすくするために wandb.Api() クラスのインスタンスを作成します
api = wandb.Api()
```

`model` という文字列を含む全てのレジストリをフィルタ：

```python
# `model` という文字列を含む全てのレジストリをフィルタ
registry_filters = {
    "name": {"$regex": "model"}
}

# フィルタに一致する全レジストリのイテラブルを返す
registries = api.registries(filter=registry_filters)
```

コレクション名に `yolo` という文字列を含む、全レジストリ横断のコレクションをフィルタ：

```python
# 全レジストリ横断で、コレクション名に `yolo` を含むコレクションをフィルタ
collection_filters = {
    "name": {"$regex": "yolo"}
}

# フィルタに一致する全コレクションのイテラブルを返す
collections = api.registries().collections(filter=collection_filters)
```

コレクション名に `yolo` を含み、`cnn` タグをもつ全てのコレクションをフィルタ：

```python
# コレクション名に `yolo` を含み、`cnn` タグをもつ
# 全てのコレクションをフィルタ
collection_filters = {
    "name": {"$regex": "yolo"},
    "tag": "cnn"
}

# フィルタに一致する全コレクションのイテラブルを返す
collections = api.registries().collections(filter=collection_filters)
```

`model` を含み、タグ `image-classification` またはエイリアス `latest` をもつアーティファクトバージョンをすべて取得：

```python
# `model` を含み、タグ `image-classification` または
# エイリアス `production` をもつアーティファクトバージョンを検索
registry_filters = {
    "name": {"$regex": "model"}
}

# 論理 $or 演算子でアーティファクトバージョンをフィルタ
version_filters = {
    "$or": [
        {"tag": "image-classification"},
        {"alias": "production"}
    ]
}

# フィルタに一致する全アーティファクトバージョンのイテラブルを返す
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)
```

論理クエリオペレーターの詳細は [MongoDB ドキュメント](https://www.mongodb.com/docs/manual/reference/operator/query-logical/) を参照してください。

前述のコードスニペットの `artifacts` イテラブル内の各アイテムは `Artifact` クラスのインスタンスです。つまり、各アーティファクトの `name`、`collection`、`aliases`、`tags`、`created_at` などの属性にアクセスできます：

```python
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```
アーティファクトオブジェクトで利用可能な属性の一覧は、APIリファレンスの [Artifacts Class]({{< relref "/ref/python/sdk/classes/artifact/_index.md" >}}) をご覧ください。

2024-01-08 から 2025-03-04 13:10 UTC の間に作成された、全レジストリ・コレクション横断のアーティファクトバージョンをフィルタ：

```python
# 2024-01-08 から 2025-03-04 13:10 UTC の間に作成された
# 全アーティファクトバージョンを検索

artifact_filters = {
    "alias": "latest",
    "created_at" : {"$gte": "2024-01-08", "$lte": "2025-03-04 13:10:00"},
}

# フィルタに一致する全アーティファクトバージョンのイテラブルを返す
artifacts = api.registries().collections().versions(filter=artifact_filters)
```

日時は `YYYY-MM-DD HH:MM:SS` 形式で指定します。日付だけでフィルタしたい場合、時間・分・秒は省略可能です。

クエリの比較演算子など詳しくは、[MongoDB ドキュメント](https://www.mongodb.com/docs/manual/reference/operator/query-comparison/) も参考にしてください。