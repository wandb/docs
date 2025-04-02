---
title: Export table data
description: テーブルからデータをエクスポートする方法。
menu:
  default:
    identifier: ja-guides-models-tables-tables-download
    parent: tables
---

すべての W&B Artifacts と同様に、Tables は pandas のデータフレームに変換して、簡単にデータのエクスポートができます。

## `table` を `artifact` に変換する
まず、テーブルを artifact に変換する必要があります。これを行うには、`artifact.get(table, "table_name")` を使用するのが最も簡単です。

```python
# 新しいテーブルを作成してログに記録します。
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 作成した artifact を使用して、作成したテーブルを取得します。
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact` を Dataframe に変換する
次に、テーブルをデータフレームに変換します。

```python
# 前のコード例から続けて:
df = table.get_dataframe()
```

## データのエクスポート
これで、データフレームがサポートする任意の方法を使用してエクスポートできます。

```python
# テーブルデータを .csv に変換する
df.to_csv("example.csv", encoding="utf-8")
```

# 次のステップ
- `artifacts` の [リファレンスドキュメント]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) を確認してください。
- [Tables Walktrough]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) ガイドをご覧ください。
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) のリファレンスドキュメントを確認してください。
