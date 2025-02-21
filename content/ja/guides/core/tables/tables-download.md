---
title: Export table data
description: テーブルからデータをエクスポートする方法。
menu:
  default:
    identifier: ja-guides-core-tables-tables-download
    parent: tables
---

W&B のすべての Artifacts と同様に、Tables は pandas のデータフレームに変換して簡単にデータをエクスポートできます。

## `table` を `artifact` に変換する
まず、テーブルをアーティファクトに変換する必要があります。これを行う最も簡単な方法は `artifact.get(table, "table_name")` を使用することです。

```python
# 新しいテーブルを作成してログに記録する。
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 作成したアーティファクトを使用してテーブルを取得する。
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact` を Dataframe に変換する
次に、テーブルをデータフレームに変換します。

```python
# 前のコード例に続いて：
df = table.get_dataframe()
```

## データのエクスポート
これで、データフレームがサポートする任意のメソッドを使用してエクスポートできます。

```python
# テーブルデータを .csv に変換する
df.to_csv("example.csv", encoding="utf-8")
```

# 次のステップ
- `artifacts` に関する[参考ドキュメント]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}})をチェックしてください。
- [Tables Walktrough]({{< relref path="/guides/core/tables/tables-walkthrough.md" lang="ja" >}}) ガイドを確認してください。
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) の参考ドキュメントをチェックしてください。