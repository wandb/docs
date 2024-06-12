---
description: "\u30C6\u30FC\u30D6\u30EB\u304B\u3089\u30C7\u30FC\u30BF\u3092\u30A8\u30AF\
  \u30B9\u30DD\u30FC\u30C8\u3059\u308B\u65B9\u6CD5"
displayed_sidebar: default
---

# テーブルデータのエクスポート
すべてのW&B Artifactsと同様に、テーブルは簡単にデータをエクスポートするためにpandasのデータフレームに変換できます。

## `table`を`artifact`に変換
まず、テーブルをartifactに変換する必要があります。これを行う最も簡単な方法は`artifact.get(table, "table_name")`を使用することです。

```python
# 新しいテーブルを作成してログを取る
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 作成したartifactを使用してテーブルを取得する
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact`をデータフレームに変換
次に、テーブルをデータフレームに変換します。

```python
# 前のコード例から続けて:
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
- [Tables Walkthrough](../tables/tables-walkthrough.md)ガイドを確認してください。
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)のリファレンスドキュメントをチェックしてください。