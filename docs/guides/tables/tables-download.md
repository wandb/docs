---
description: テーブルからデータをエクスポートする方法
displayed_sidebar: default
---


# テーブルデータのエクスポート
すべての W&B Artifacts と同様に、Tables は簡単にデータをエクスポートできるように pandas データフレームに変換できます。

## `table` を `artifact` に変換
まず、テーブルをアーティファクトに変換します。これを行う最も簡単な方法は `artifact.get(table, "table_name")` を使用することです。

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

## `artifact` をデータフレームに変換
次に、テーブルをデータフレームに変換します。

```python
# 前のコード例に続いて
df = table.get_dataframe()
```

## データのエクスポート
これで、データフレームがサポートする任意の方法でデータをエクスポートできます。

```python
# テーブルデータを .csv に変換する
df.to_csv("example.csv", encoding="utf-8")
```

# 次のステップ
- `artifacts` に関する [リファレンスドキュメント](../artifacts/construct-an-artifact.md) をチェックしてください。
- [Tables Walktrough](../tables/tables-walkthrough.md) ガイドをご覧ください。
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) リファレンスドキュメントをチェックしてください。