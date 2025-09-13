---
title: テーブルのデータをエクスポート
description: テーブルからデータをエクスポートする方法。
menu:
  default:
    identifier: ja-guides-models-tables-tables-download
    parent: tables
---

すべての W&B Artifacts と 同様に、Tables は データのエクスポートを 容易にするため pandas の DataFrame に 変換できます。
## `table` を `artifact` に 変換する
まず、table を artifact に 変換する必要が あります。最も簡単な方法は `artifact.get(table, "table_name")` を 使うことです:
```python
# 新しい table を作成してログに記録します。
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 作成した artifact を使って、作成済みの table を取得します。
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```
## `artifact` を DataFrame に 変換する
次に、table を DataFrame に 変換します:
```python
# 直前のコード例から続きます:
df = table.get_dataframe()
```
## データのエクスポート
これで、DataFrame がサポートする任意のメソッドでエクスポートできます:
```python
# table のデータを .csv に変換する
df.to_csv("example.csv", encoding="utf-8")
```
# 次のステップ
- `artifacts` に関する [リファレンス ドキュメント]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) をご覧ください。
- 私たちの [Tables ウォークスルー]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) ガイドを参照してください。
- [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) のリファレンス ドキュメントをご覧ください。