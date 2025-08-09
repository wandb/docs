---
title: Tablesデータをエクスポートする
description: Tableからデータをエクスポートする方法
menu:
  default:
    identifier: ja-guides-models-tables-tables-download
    parent: tables
---

すべての W&B Artifacts と同様に、Tables は pandas のデータフレームへ変換することで、簡単にデータをエクスポートできます。

## `table` を `artifact` に変換する
まず、table を artifact に変換します。一番簡単な方法は `artifact.get(table, "table_name")` を使うことです。

```python
# 新しいTableを作成してログします。
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 作成した artifact からTableを取得します。
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact` をデータフレームに変換する
次に、table をデータフレームへ変換します。

```python
# 前のコード例の続き:
df = table.get_dataframe()
```

## データのエクスポート
これで、データフレームがサポートする任意の方法でエクスポートできます。

```python
# Tableデータを .csv へ変換
df.to_csv("example.csv", encoding="utf-8")
```

# 次のステップ
- `artifacts` の[リファレンスドキュメント]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}})をチェックしましょう。
- [Tables Walkthrough]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}})ガイドも参考にしてください。
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) のリファレンスドキュメントもご覧ください。