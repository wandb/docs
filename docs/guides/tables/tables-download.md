---
description: テーブルからデータをエクスポートする方法
displayed_sidebar: default
---

# テーブルデータのエクスポート
すべてのW&B Artifactsと同様に、Tablesもpandasデータフレームに変換してデータを簡単にエクスポートできます。

## `table`を`artifact`に変換
まず、テーブルをartifactに変換する必要があります。これを行う最も簡単な方法は、`artifact.get(table, "table_name")`を使用することです。

```python
# 新しいテーブルを作成してログする。
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 作成したartifactを使用してテーブルを取得する。
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact`をデータフレームに変換
次に、テーブルをデータフレームに変換します。

```python
# 前のコード例から続ける:
df = table.get_dataframe()
```

## データのエクスポート
これで、データフレームがサポートする任意の方法でエクスポートできます。

```python
# テーブルデータを.csvに変換
df.to_csv("example.csv", encoding="utf-8")
```

# 次のステップ
- `artifacts`に関する[リファレンスドキュメント](../artifacts/construct-an-artifact.md)をチェックしてください。
- [Tables Walkthrough](../tables/tables-walkthrough.md)ガイドを参照してください。
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)のリファレンスドキュメントをチェックしてください。