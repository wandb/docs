---
title: Update an artifact
description: W&B Run 内外で既存のアーティファクトを更新します。
menu:
  default:
    identifier: ja-guides-core-artifacts-update-an-artifact
    parent: artifacts
weight: 4
---

指定した値を渡して、アーティファクトの `description`、`metadata`、および `alias` を更新します。`save()` メソッドを呼び出して、W&B サーバー上のアーティファクトを更新します。アーティファクトは、W&B Run 中または Run 外で更新できます。

Run 外でアーティファクトを更新するには、W&B Public API ([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})) を使用します。Run 中にアーティファクトを更新するには、Artifact API ([`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}})) を使用します。

{{% alert color="secondary" %}}
モデルレジストリでモデルにリンクされているアーティファクトのエイリアスは更新できません。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="During a run" %}}

以下のコード例は、[`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用してアーティファクトの説明を更新する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
以下のコード例は、`wandb.Api` API を使用してアーティファクトの説明を更新する方法を示しています。

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# 説明を更新
artifact.description = "My new description"

# メタデータのキーを選択的に更新
artifact.metadata["oldKey"] = "new value"

# メタデータを完全に置き換え
artifact.metadata = {"newKey": "new value"}

# エイリアスを追加
artifact.aliases.append("best")

# エイリアスを削除
artifact.aliases.remove("latest")

# エイリアスを完全に置き換え
artifact.aliases = ["replaced"]

# すべてのアーティファクト変更を保存
artifact.save()
```

詳細は、Weights and Biases の [Artifact API]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) を参照してください。  
  {{% /tab %}}
  {{% tab header="With collections" %}}
単一のアーティファクトと同じ方法で、Artifact コレクションを更新することもできます。

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<This is where you'd describe the purpose of your collection.>"
artifact.save()
```
詳細は、[Artifacts Collection]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) のリファレンスを参照してください。
  {{% /tab %}}
{{% /tabpane %}}