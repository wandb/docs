---
title: Update an artifact
description: W&B Run の内外で既存の Artifact を更新します。
menu:
  default:
    identifier: ja-guides-core-artifacts-update-an-artifact
    parent: artifacts
weight: 4
---

Artifact の `description`、`metadata`、および `alias` を更新するために希望する value を渡します。W&B サーバー上の Artifact を更新するには、`save()` メソッドを呼び出します。W&B の Run 中または Run の外部で Artifact を更新できます。

Run の外部で Artifact を更新するには、W&B Public API（[`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})）を使用します。Run 中に Artifact を更新するには、Artifact API（[`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}})）を使用します。

{{% alert color="secondary" %}}
モデルレジストリ内のモデルにリンクされている Artifact の エイリアス を更新することはできません。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Run 中" %}}

次のコード例は、[`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用して、Artifact の description を更新する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```
  {{% /tab %}}
  {{% tab header="Run の外部" %}}
次のコード例は、`wandb.Api` API を使用して、Artifact の description を更新する方法を示しています。

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# description を更新
artifact.description = "My new description"

# 選択的に metadata の キー を更新
artifact.metadata["oldKey"] = "new value"

# metadata を完全に置き換え
artifact.metadata = {"newKey": "new value"}

# エイリアス を追加
artifact.aliases.append("best")

# エイリアス を削除
artifact.aliases.remove("latest")

# エイリアス を完全に置き換え
artifact.aliases = ["replaced"]

# すべての Artifact の変更を永続化
artifact.save()
```

詳細については、Weights and Biases [Artifact API]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) を参照してください。
  {{% /tab %}}
  {{% tab header="コレクションを使用" %}}
単一の Artifact と同じ方法で、Artifact のコレクションを更新することもできます。

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<This is where you'd describe the purpose of your collection.>"
artifact.save()
```
詳細については、[Artifacts Collection]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) のリファレンスを参照してください。
  {{% /tab %}}
{{% /tabpane %}}
