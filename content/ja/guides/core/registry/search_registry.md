---
title: Find registry items
menu:
  default:
    identifier: ja-guides-core-registry-search_registry
    parent: registry
weight: 7
---

[W&B Registry App のグローバル検索バー]({{< relref path="./search_registry.md#search-for-registry-items" lang="ja" >}}) を使用して、レジストリ、コレクション、アーティファクト バージョン タグ、コレクション タグ、または エイリアス を検索します。MongoDB スタイルのクエリを使用して、W&B Python SDK を使用して特定の条件に基づいて[レジストリ、コレクション、およびアーティファクト バージョンをフィルタリング]({{< relref path="./search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ja" >}})できます。

表示する権限を持つアイテムのみが検索結果に表示されます。

## レジストリ アイテムの検索

レジストリ アイテムを検索するには:

1. W&B Registry App に移動します。
2. ページ上部の検索バーに検索語句を指定します。Enter キーを押して検索します。

指定した用語が既存のレジストリ、コレクション名、アーティファクト バージョン タグ、コレクション タグ、または エイリアス と一致する場合、検索結果は検索バーの下に表示されます。

{{< img src="/images/registry/search_registry.gif" alt="レジストリ検索バーにテキストを入力してレジストリ アイテムをフィルタリングするユーザーの.gif" >}}

## MongoDB スタイルのクエリでレジストリ アイテムをクエリする

[`wandb.Api().registries()`]({{< relref path="/ref/python/public-api/api.md#registries" lang="ja" >}}) と [クエリ述語](https://www.mongodb.com/docs/manual/reference/glossary/#std-term-query-predicate) を使用して、1 つ以上の [MongoDB スタイルのクエリ](https://www.mongodb.com/docs/compass/current/query/filter/) に基づいて、レジストリ、コレクション、およびアーティファクト バージョンをフィルタリングします。

次の表に、フィルタリングするアイテムのタイプに基づいて使用できるクエリ名を示します。

| | クエリ名 |
| ----- | ----- |
| registries | `name`, `description`, `created_at`, `updated_at` |
| collections | `name`, `tag`, `description`, `created_at`, `updated_at` |
| versions | `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

次の コードスニペット は、一般的な検索シナリオを示しています。

`wandb.Api().registries()` メソッドを使用するには、まず W&B Python SDK ([`wandb`]({{< relref path="/ref/python/_index.md" lang="ja" >}})) ライブラリをインポートします。
```python
import wandb

# (オプション) 可読性を高めるために wandb.Api() クラスのインスタンスを作成します
api = wandb.Api()
```

文字列 `model` を含むすべての registries をフィルタリングします。

```python
# 文字列 `model` を含むすべての registries をフィルタリングします
registry_filters = {
    "name": {"$regex": "model"}
}

# フィルタに一致するすべての registries のイテラブルを返します
registries = api.registries(filter=registry_filters)
```

コレクション名に文字列 `yolo` を含む、registry に関係なく、すべての collections をフィルタリングします。

```python
# コレクション名に文字列 `yolo` を含む、registry に関係なく、
# すべての collections をフィルタリングします
collection_filters = {
    "name": {"$regex": "yolo"}
}

# フィルタに一致するすべての collections のイテラブルを返します
collections = api.registries().collections(filter=collection_filters)
```

コレクション名に文字列 `yolo` を含み、`cnn` をタグとして持つ、registry に関係なく、すべての collections をフィルタリングします。

```python
# コレクション名に文字列 `yolo` を含み、
# `cnn` をタグとして持つ、registry に関係なく、すべての collections をフィルタリングします
collection_filters = {
    "name": {"$regex": "yolo"},
    "tag": "cnn"
}

# フィルタに一致するすべての collections のイテラブルを返します
collections = api.registries().collections(filter=collection_filters)
```

文字列 `model` を含み、タグ `image-classification` または `latest` エイリアス のいずれかを持つすべてのアーティファクト バージョンを検索します。

```python
# 文字列 `model` を含み、
# タグ `image-classification` または `latest` エイリアス のいずれかを持つすべてのアーティファクト バージョンを検索します
registry_filters = {
    "name": {"$regex": "model"}
}

# 論理 $or 演算子を使用してアーティファクト バージョンをフィルタリングします
version_filters = {
    "$or": [
        {"tag": "image-classification"},
        {"alias": "production"}
    ]
}

# フィルタに一致するすべてのアーティファクト バージョンのイテラブルを返します
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)
```

[論理クエリオペレーター](https://www.mongodb.com/docs/manual/reference/operator/query-logical/) の詳細については、MongoDB のドキュメントを参照してください。

前の コードスニペット の `artifacts` イテラブルの各アイテムは、`Artifact` クラスのインスタンスです。つまり、各アーティファクトの属性 ( `name` 、 `collection` 、 `aliases` 、 `tags` 、 `created_at` など) にアクセスできます。

```python
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```
アーティファクト オブジェクトの属性の完全なリストについては、API Reference ドキュメントの [Artifacts Class]({{< relref path="/ref/python/artifact/_index.md" lang="ja" >}}) を参照してください。

2024-01-08 から 2025-03-04 の 13:10 UTC の間に作成された、registry または collection に関係なく、すべてのアーティファクト バージョンをフィルタリングします。

```python
# 2024-01-08 から 2025-03-04 の 13:10 UTC の間に作成されたすべてのアーティファクト バージョンを検索します。

artifact_filters = {
    "alias": "latest",
    "created_at" : {"$gte": "2024-01-08", "$lte": "2025-03-04 13:10:00"},
}

# フィルタに一致するすべてのアーティファクト バージョンのイテラブルを返します
artifacts = api.registries().collections().versions(filter=artifact_filters)
```

日付と時刻を `YYYY-MM-DD HH:MM:SS` 形式で指定します。日付のみでフィルタリングする場合は、時間、分、秒を省略できます。

[クエリ比較](https://www.mongodb.com/docs/manual/reference/operator/query-comparison/) の詳細については、MongoDB のドキュメントを参照してください。
