---
title: テーブル データをエクスポート
description: テーブルからデータをエクスポートする方法。
menu:
  default:
    identifier: ja-guides-models-tables-tables-download
    parent: tables
---

すべての W&B Artifacts 同様に、Tables は pandas データフレームに変換して、データのエクスポートを簡単に行うことができます。

## `table` を `artifact` に変換する
まず、テーブルをアーティファクトに変換する必要があります。これを行う最も簡単な方法は `artifact.get(table, "table_name")` を使用することです：

```python
# 新しいテーブルを作成してログします。
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 作成したアーティファクトを使用してテーブルを取得します。
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact` をデータフレームに変換する
次に、テーブルをデータフレームに変換します：

```python
# 前のコード例から続けて：
df = table.get_dataframe()
```

## データをエクスポート
現在、データフレームがサポートする任意のメソッドを使用してエクスポートできます：

```python
# テーブルデータを .csv に変換
df.to_csv("example.csv", encoding="utf-8")
```

# 次のステップ
- `artifacts` に関する [リファレンスドキュメント]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) をチェックしてください。
- [Tables Walkthrough]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) ガイドを確認してください。
- [データフレーム](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) リファレンスドキュメントを参照してください。