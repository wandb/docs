---
title: アーティファクトを更新する
description: W&B Run の内外で既存の Artifact を更新する方法をご紹介します。
menu:
  default:
    identifier: ja-guides-core-artifacts-update-an-artifact
    parent: artifacts
weight: 4
---

`description`、`metadata`、`alias` に更新したい値を設定します。`save()` メソッドを呼び出すことで、W&B サーバー上の artifact を更新できます。artifact の更新は W&B Run の実行中でも、Run の外でも可能です。

Run の外から artifact を更新する場合は W&B Public API（[`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})）を使用してください。Run 実行中に artifact を更新する場合は Artifact API（[`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}})）を利用します。

{{% alert color="secondary" %}}
Model Registry でモデルに紐づけられた artifact の alias は更新できません。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Run 実行中の場合" %}}

以下のコード例は、[`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) API を使って artifact の description を更新する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```
  {{% /tab %}}
  {{% tab header="Run の外の場合" %}}
以下のコード例は、`wandb.Api` API を使って artifact の description を更新する方法を示しています。

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# description を更新
artifact.description = "My new description"

# メタデータの一部 key を個別に更新
artifact.metadata["oldKey"] = "new value"

# メタデータをまとめて置き換え
artifact.metadata = {"newKey": "new value"}

# alias を追加
artifact.aliases.append("best")

# alias を削除
artifact.aliases.remove("latest")

# alias を完全に置き換え
artifact.aliases = ["replaced"]

# すべての artifact の変更を保存する
artifact.save()
```

詳しくは Weights and Biases の [Artifact API]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) をご覧ください。
  {{% /tab %}}
  {{% tab header="コレクションの場合" %}}
アーティファクトのコレクションも、単一の artifact と同じ方法で更新できます。

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<This is where you'd describe the purpose of your collection.>"
artifact.save()
```
詳しくは [Artifacts Collection]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) のリファレンスをご確認ください。
  {{% /tab %}}
{{% /tabpane %}}