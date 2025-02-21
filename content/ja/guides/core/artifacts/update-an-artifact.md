---
title: Update an artifact
description: W&B の Run の内外で、既存の Artifact を更新します。
menu:
  default:
    identifier: ja-guides-core-artifacts-update-an-artifact
    parent: artifacts
weight: 4
---

Artifacts の `description` 、 `metadata` 、および `alias` を更新するために、必要な値を渡します。W&B サーバー上の Artifacts を更新するには、 `save()` メソッドを呼び出します。W&B の Run 中または Run の外部で Artifacts を更新できます。

Run の外部で Artifacts を更新するには、W&B Public API ( [`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})) を使用します。Run 中に Artifacts を更新するには、Artifact API ( [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}})) を使用します。

{{% alert color="secondary" %}}
モデルレジストリ内のモデルにリンクされている artifact のエイリアスは更新できません。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Run 中" %}}

次のコード例は、[`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用して artifact の description を更新する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```
  {{% /tab %}}
  {{% tab header="Run の外部" %}}
次のコード例は、 `wandb.Api` API を使用して artifact の description を更新する方法を示しています。

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# Update the description
artifact.description = "My new description"

# Selectively update metadata keys
artifact.metadata["oldKey"] = "new value"

# Replace the metadata entirely
artifact.metadata = {"newKey": "new value"}

# Add an alias
artifact.aliases.append("best")

# Remove an alias
artifact.aliases.remove("latest")

# Completely replace the aliases
artifact.aliases = ["replaced"]

# Persist all artifact modifications
artifact.save()
```

詳細については、Weights and Biases の [Artifact API]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) を参照してください。
  {{% /tab %}}
  {{% tab header="コレクションを使用する" %}}
単一の artifact と同じ方法で、Artifact のコレクションを更新することもできます。

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
