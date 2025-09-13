---
title: レジストリ内のエントリを検索
menu:
  default:
    identifier: ja-guides-core-registry-search_registry
    parent: registry
weight: 7
---

W&B Registry App の[グローバル検索バー]({{< relref path="./search_registry.md#search-for-registry-items" lang="ja" >}}) を使用して、registry、コレクション、アーティファクト バージョン タグ、コレクション タグ、またはエイリアスを検索します。W&B Python SDK を使用して、特定の条件に基づいて [registry、コレクション、およびアーティファクト バージョンをフィルタリング]({{< relref path="./search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ja" >}})するために、MongoDB スタイルのクエリを使用できます。

表示権限のある項目のみが検索結果に表示されます。

## registry アイテムの検索

registry アイテムを検索するには：

1. W&B Registry App に移動します。
2. ページ上部の検索バーに検索語句を指定します。Enter キーを押して検索します。

指定した語句が既存の registry、コレクション名、アーティファクト バージョン タグ、コレクション タグ、またはエイリアスと一致する場合、検索結果は検索バーの下に表示されます。

{{< img src="/images/registry/search_registry.gif" alt="Searching within a Registry" >}}

## MongoDB スタイルのクエリによる registry アイテムのクエリ

[`wandb.Api().registries()`]({{< relref path="/ref/python/public-api/api.md#registries" lang="ja" >}}) と [クエリ述語](https://www.mongodb.com/docs/manual/reference/glossary/#std-term-query-predicate) を使用して、1 つ以上の [MongoDB スタイルのクエリ](https://www.mongodb.com/docs/compass/current/query/filter/) に基づいて registry、コレクション、およびアーティファクト バージョンをフィルタリングします。

次の表は、フィルタリングしたいアイテムの種類に基づいて使用できるクエリ名をリストしています。

| | クエリ名 |
| ----- | ----- |
| registries | `name`, `description`, `created_at`, `updated_at` |
| collections | `name`, `tag`, `description`, `created_at`, `updated_at` |
| versions | `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

以下のコード例は、いくつかの一般的な検索シナリオを示しています。

`wandb.Api().registries()` メソッドを使用するには、まず W&B Python SDK ([`wandb`]({{< relref path="/ref/python/_index.md" lang="ja" >}})) ライブラリをインポートします。
```python
import wandb

# (オプション) 読みやすくするために wandb.Api() クラスのインスタンスを作成します
api = wandb.Api()
```

文字列 `model` を含むすべての registry をフィルタリングします。

```python
# 文字列 `model` を含むすべての registry をフィルタリングします
registry_filters = {
    "name": {"$regex": "model"}
}

# フィルタに一致するすべての registry のイテラブルを返します
registries = api.registries(filter=registry_filters)
```

registry とは無関係に、コレクション名に文字列 `yolo` を含むすべてのコレクションをフィルタリングします。

```python
# registry とは無関係に、
# コレクション名に文字列 `yolo` を含むすべてのコレクションをフィルタリングします
collection_filters = {
    "name": {"$regex": "yolo"}
}

# フィルタに一致するすべてのコレクションのイテラブルを返します
collections = api.registries().collections(filter=collection_filters)
```

registry とは無関係に、コレクション名に文字列 `yolo` を含み、`cnn` をタグとして持つすべてのコレクションをフィルタリングします。

```python
# registry とは無関係に、
# コレクション名に文字列 `yolo` を含み、`cnn` をタグとして持つすべてのコレクションをフィルタリングします
collection_filters = {
    "name": {"$regex": "yolo"},
    "tag": "cnn"
}

# フィルタに一致するすべてのコレクションのイテラブルを返します
collections = api.registries().collections(filter=collection_filters)
```

文字列 `model` を含み、`image-classification` タグまたは `latest` エイリアスのいずれかを持つすべてのアーティファクト バージョンを見つけます。

```python
# 文字列 `model` を含み、
# `image-classification` タグまたは `latest` エイリアスのいずれかを持つすべてのアーティファクト バージョンを見つけます
registry_filters = {
    "name": {"$regex": "model"}
}

# アーティファクト バージョンをフィルタリングするために論理 $or 演算子を使用します
version_filters = {
    "$or": [
        {"tag": "image-classification"},
        {"alias": "production"}
    ]
}

# フィルタに一致するすべてのアーティファクト バージョンのイテラブルを返します
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)
```

[論理クエリ演算子](https://www.mongodb.com/docs/manual/reference/operator/query-logical/) の詳細については、MongoDB ドキュメントを参照してください。

前のコードスニペットの `artifacts` イテラブル内の各アイテムは `Artifact` クラスのインスタンスです。これは、`name`、`collection`、`aliases`、`tags`、`created_at` など、各アーティファクトの属性にアクセスできることを意味します。

```python
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```
アーティファクト オブジェクトの属性の完全なリストについては、API リファレンス ドキュメントの [Artifacts クラス]({{< relref path="/ref/python/sdk/classes/artifact/_index.md" lang="ja" >}}) を参照してください。

registry またはコレクションとは無関係に、2024-01-08 から 2025-03-04 の 13:10 UTC の間に作成されたすべてのアーティファクト バージョンをフィルタリングします。

```python
# 2024-01-08 から 2025-03-04 の 13:10 UTC の間に作成されたすべてのアーティファクト バージョンを見つけます。

artifact_filters = {
    "alias": "latest",
    "created_at" : {"$gte": "2024-01-08", "$lte": "2025-03-04 13:10:00"},
}

# フィルタに一致するすべてのアーティファクト バージョンのイテラブルを返します
artifacts = api.registries().collections().versions(filter=artifact_filters)
```

日付と時刻は `YYYY-MM-DD HH:MM:SS` 形式で指定します。日付のみでフィルタリングしたい場合は、時、分、秒を省略できます。

[クエリ比較](https://www.mongodb.com/docs/manual/reference/operator/query-comparison/) の詳細については、MongoDB ドキュメントを参照してください。