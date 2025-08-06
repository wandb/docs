---
title: テーブル データのエクスポート
description: テーブルからデータをエクスポートする方法
menu:
  default:
    identifier: tables-download
    parent: tables
---

すべての W&B Artifacts と同様に、Tables は pandas の dataframe に変換して、簡単にデータをエクスポートできます。

## `table` を `artifact` に変換する
まず、テーブルを artifact に変換します。最も簡単な方法は `artifact.get(table, "table_name")` を使うことです。

```python
# 新しいテーブルを作成し、ログに記録します。
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 作成した artifact を使用して、テーブルを取得します。
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact` を Dataframe に変換する
次に、テーブルを dataframe に変換します。

```python
# 前のコード例から続けます:
df = table.get_dataframe()
```

## データのエクスポート
これで、dataframe のサポートする任意のメソッドでデータをエクスポートできます。

```python
# テーブルデータを .csv に変換します
df.to_csv("example.csv", encoding="utf-8")
```

# 次のステップ
- `artifacts` の[リファレンスドキュメント]({{< relref "/guides/core/artifacts/construct-an-artifact.md" >}})をチェックしてください。
- [Tables Walkthrough]({{< relref "/guides/models/tables/tables-walkthrough.md" >}})ガイドを確認してみてください。
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) のリファレンスドキュメントもご覧ください。